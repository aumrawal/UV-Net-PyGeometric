"""
Fixed harness for UV-Net autoresearch on Fusion 360 Gallery segmentation. DO NOT MODIFY.

Holds everything that must stay constant across experiments so that results are comparable:
the data (preprocessed Fusion 360 Gallery s2.0.0 graphs, official split, UV-Net's 80/20
train/val split), the time budget, and the evaluation (`evaluate`, per-face mean IoU on the
full validation split). The test split is never loaded here: it is reserved for the final
check of a promoted model, outside the loop.

train.py imports from this file:

    from prepare import TIME_BUDGET, NUM_CLASSES, make_train_dataset, evaluate

Data layout (written by the preprocessing kernel):
    <data_dir>/split.json, <data_dir>/cache/{train,val,test}.pt
Each graph:
    x          : F x 7 x 10 x 10   face UV-grids (xyz, normal, trim mask), channels-first, float32
    edge_attr  : E x 6 x 10        edge U-grids (xyz, tangent), channels-first, float32
    edge_index : 2 x E             directed face-adjacency edges (both directions present)
    y          : F                 face labels, 0..7 (see CLASS_NAMES)
    face_attr  : F x 4             exact face area + bounding-box extent (xyz), float32, if present
"""
import json
import os
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

T_IMPORT = time.time()

# ---------------------------------------------------------------------------------------------
# Constants (the runner may set AR_TIME_BUDGET for smoke tests; experiments must not)
# ---------------------------------------------------------------------------------------------
TIME_BUDGET = int(os.environ.get("AR_TIME_BUDGET", 600))   # training wall clock, seconds
EVAL_BATCH_SIZE = 64
CLASS_NAMES = ["ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd", "Fillet", "Chamfer", "RevolveSide", "RevolveEnd"]
NUM_CLASSES = len(CLASS_NAMES)


def data_dir():
    """AR_DATA_DIR if set, else the preprocessing kernel's output attached under /kaggle/input."""
    if os.environ.get("AR_DATA_DIR"):
        return pathlib.Path(os.environ["AR_DATA_DIR"])
    found = sorted(pathlib.Path("/kaggle/input").rglob("fusion360_processed/split.json"))
    assert found, "Fusion 360 data not found: set AR_DATA_DIR or attach the preprocessing kernel"
    return found[0].parent


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def random_rotation():
    """Random rotation in 90 degree steps about a canonical axis (UV-Net's augmentation)."""
    axis = random.choice([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])])
    angle = np.radians(random.choice([0.0, 90.0, 180.0, 270.0]))
    return torch.tensor(Rotation.from_rotvec(angle * axis).as_matrix(), dtype=torch.float32)


def _rotate(grid, R):
    grid = grid.clone()
    shape = grid[..., :3].shape
    grid[..., :3] = (grid[..., :3].reshape(-1, 3) @ R).reshape(shape)
    grid[..., 3:6] = (grid[..., 3:6].reshape(-1, 3) @ R).reshape(shape)
    return grid


class Fusion360Graphs(Dataset):
    """One PyG graph per part, read from the packed, memory-mapped split cache."""

    def __init__(self, split, random_rotate=False):
        assert split in ("train", "val"), "the test split is not available inside the loop"
        pack = torch.load(data_dir() / "cache" / f"{split}.pt", weights_only=False, mmap=True)
        self.names = pack["names"]
        self.x, self.edge_attr, self.edge_index, self.y = pack["x"], pack["edge_attr"], pack["edge_index"], pack["y"]
        self.face_attr = pack.get("face_attr")
        self.face_ptr, self.edge_ptr = pack["face_ptr"], pack["edge_ptr"]
        self.random_rotate = random_rotate

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        f0, f1 = int(self.face_ptr[idx]), int(self.face_ptr[idx + 1])
        e0, e1 = int(self.edge_ptr[idx]), int(self.edge_ptr[idx + 1])
        d = Data(edge_index=self.edge_index[:, e0:e1].clone(), y=self.y[f0:f1].clone(), num_nodes=f1 - f0)
        x, e = self.x[f0:f1].float(), self.edge_attr[e0:e1].float()   # stored channels-last, float16
        if self.random_rotate:
            R = random_rotation()
            x, e = _rotate(x, R), _rotate(e, R)
        d.x = x.permute(0, 3, 1, 2).contiguous()         # F x 7 x U x V
        d.edge_attr = e.permute(0, 2, 1).contiguous()    # E x 6 x U
        if self.face_attr is not None:
            d.face_attr = self.face_attr[f0:f1].float()
        return d


def make_train_dataset(random_rotate=False):
    """The training split. Wrap it in any DataLoader / sampler you like."""
    return Fusion360Graphs("train", random_rotate=random_rotate)


# ---------------------------------------------------------------------------------------------
# Evaluation: the ground-truth metric
# ---------------------------------------------------------------------------------------------
_VAL = None


@torch.no_grad()
def evaluate(model, device):
    """
    Per-face mean IoU over the 8 classes on the full validation split (higher is better).

    `model(batch)` must return per-face logits of shape (num_faces, NUM_CLASSES) for a PyG Batch.
    The model is put in eval mode. The last call's result is the experiment's score; it is also
    written to $AR_RESULT_PATH for the runner.
    """
    global _VAL
    if _VAL is None:
        _VAL = Fusion360Graphs("val")
    loader = DataLoader(_VAL, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    model.eval()
    t0 = time.time()
    mat = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    total_loss, total_faces = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch).float()
        assert logits.shape == (batch.y.numel(), NUM_CLASSES), f"bad logits shape {tuple(logits.shape)}"
        total_loss += F.cross_entropy(logits, batch.y, reduction="sum").item()
        total_faces += batch.y.numel()
        mat += torch.bincount((batch.y * NUM_CLASSES + logits.argmax(-1)).cpu(),
                              minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    tp = mat.diag().float()
    support, predicted = mat.sum(1).float(), mat.sum(0).float()
    iou = tp / (support + predicted - tp).clamp(min=1)
    present = support > 0
    result = {
        "val_miou": iou[present].mean().item(),
        "val_acc": (tp.sum() / mat.sum().clamp(min=1)).item(),
        "val_loss": total_loss / max(total_faces, 1),
        "per_class_iou": dict(zip(CLASS_NAMES, iou.tolist())),
        "eval_seconds": time.time() - t0,
        "seconds_since_start": time.time() - T_IMPORT,
        "time_budget": TIME_BUDGET,
    }
    if os.environ.get("AR_RESULT_PATH"):
        pathlib.Path(os.environ["AR_RESULT_PATH"]).write_text(json.dumps(result, indent=1))
    return result
