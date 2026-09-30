"""MFCAD++ dataset of preprocessed PyG graphs (see process/step_to_pyg.py)."""
import json
import pathlib
import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

# MFCAD++ feature classes (Colligan et al., Hierarchical CADNet, 2022). Label 24 is the stock face.
MFCADPP_CLASSES = [
    "chamfer", "through_hole", "triangular_passage", "rectangular_passage", "6sides_passage",
    "triangular_through_slot", "rectangular_through_slot", "circular_through_slot",
    "rectangular_through_step", "2sides_through_step", "slanted_through_step", "Oring",
    "blind_hole", "triangular_pocket", "rectangular_pocket", "6sides_pocket", "circular_end_pocket",
    "rectangular_blind_slot", "v_circular_end_blind_slot", "h_circular_end_blind_slot",
    "triangular_blind_step", "circular_blind_step", "rectangular_blind_step", "round", "stock",
]


def random_rotation():
    """Random rotation in 90 degree increments about a canonical axis (as in UV-Net)."""
    axis = random.choice([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])])
    angle = np.radians(random.choice([0.0, 90.0, 180.0, 270.0]))
    return torch.tensor(Rotation.from_rotvec(angle * axis).as_matrix(), dtype=torch.float32)


def _rotate(grid, R):
    grid = grid.clone()
    shape = grid[..., :3].shape
    grid[..., :3] = (grid[..., :3].reshape(-1, 3) @ R).reshape(shape)
    grid[..., 3:6] = (grid[..., 3:6].reshape(-1, 3) @ R).reshape(shape)
    return grid


class MFCADPPDataset(Dataset):
    num_classes = len(MFCADPP_CLASSES)

    def __init__(self, root_dir, split="train", random_rotate=False, max_samples=None):
        """
        Args:
            root_dir: folder produced by process/step_to_pyg.py (contains graphs/ and split.json)
            split: "train", "val" or "test"
            random_rotate: apply random 90-degree rotations (augmentation)
            max_samples: cap the number of models (quick prototyping on a subset)

        The per-model graphs are packed into one contiguous cache file per split
        (<root_dir>/cache/<split>.pt) the first time a split is loaded; later runs read
        that single file instead of tens of thousands of small ones.
        """
        assert split in ("train", "val", "test")
        root = pathlib.Path(root_dir)
        names = json.loads((root / "split.json").read_text())[split]
        pack = _load_or_build_cache(root, split, names)
        n = len(pack["names"]) if not max_samples else min(max_samples, len(pack["names"]))
        self.names = pack["names"][:n]
        self.x, self.edge_attr, self.edge_index, self.y = pack["x"], pack["edge_attr"], pack["edge_index"], pack["y"]
        self.face_ptr, self.edge_ptr = pack["face_ptr"], pack["edge_ptr"]
        self.random_rotate = random_rotate
        print(f"{split}: {n} models, {int(self.face_ptr[n])} faces")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        f0, f1 = int(self.face_ptr[idx]), int(self.face_ptr[idx + 1])
        e0, e1 = int(self.edge_ptr[idx]), int(self.edge_ptr[idx + 1])
        d = Data(edge_index=self.edge_index[:, e0:e1].clone(), y=self.y[f0:f1].clone(), num_nodes=f1 - f0)
        d.name = self.names[idx]
        # Stored channels-last in float16; the CNN encoders want float32 channels-first
        x, e = self.x[f0:f1].float(), self.edge_attr[e0:e1].float()
        if self.random_rotate:
            R = random_rotation()
            x, e = _rotate(x, R), _rotate(e, R)
        d.x = x.permute(0, 3, 1, 2).contiguous()          # F x 7 x U x V
        d.edge_attr = e.permute(0, 2, 1).contiguous()     # E x 6 x U
        return d


def _load_or_build_cache(root, split, names):
    """Pack all graphs of a split into contiguous tensors; rebuilt if split.json changes."""
    cache = root / "cache" / f"{split}.pt"
    if cache.exists():
        # mmap: tensors are paged in from the file on demand instead of being copied into RAM,
        # which keeps the ~3 GB train split out of swap on 16 GB machines
        pack = torch.load(cache, weights_only=False, mmap=True)
        if pack["names_requested"] == list(names):
            return pack
        print(f"Split list changed; rebuilding {cache}")
    xs, es, eis, ys, kept = [], [], [], [], []
    for name in tqdm(names, desc=f"Packing {split} cache"):
        d = torch.load(root / "graphs" / f"{name}.pt", weights_only=False)
        if d.edge_index.size(1) == 0:
            continue
        xs.append(d.x); es.append(d.edge_attr); eis.append(d.edge_index); ys.append(d.y); kept.append(name)
    face_counts = torch.tensor([0] + [x.size(0) for x in xs])
    edge_counts = torch.tensor([0] + [e.size(0) for e in es])
    pack = {
        "names": kept,                      # models actually packed
        "names_requested": list(names),     # split.json entry, used to detect stale caches
        "x": torch.cat(xs),                 # sumF x U x V x 7 float16
        "edge_attr": torch.cat(es),         # sumE x U x 6 float16
        "edge_index": torch.cat(eis, 1),    # 2 x sumE, node ids local to each model
        "y": torch.cat(ys),
        "face_ptr": face_counts.cumsum(0),
        "edge_ptr": edge_counts.cumsum(0),
    }
    cache.parent.mkdir(exist_ok=True)
    torch.save(pack, cache)
    del pack, xs, es, eis, ys
    return torch.load(cache, weights_only=False, mmap=True)
