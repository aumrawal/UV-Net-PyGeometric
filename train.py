"""
Train / test UV-Net (PyTorch Geometric port) for per-face segmentation on MFCAD++ (machining
features, the default) or the Fusion 360 Gallery segmentation dataset (modeling operations).

    python train.py train --data_dir data/processed --epochs 100 --batch_size 64
    python train.py test  --data_dir data/processed --checkpoint results/<run>/best.pt
    python train.py train --dataset fusion360 --data_dir data/fusion360_processed

Device selection: CUDA if available, else Apple-silicon MPS, else CPU (override with --device).
"""
import argparse
import json
import pathlib
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from datasets.fusion360 import Fusion360Dataset, FUSION360_CLASSES
from datasets.mfcadpp import MFCADPPDataset, MFCADPP_CLASSES
from uvnet_pyg.models import UVNetSegmenter

DATASETS = {
    "mfcadpp": (MFCADPPDataset, MFCADPP_CLASSES),
    "fusion360": (Fusion360Dataset, FUSION360_CLASSES),
}


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bucket_sizes(args, device):
    """Shape bucketing (see UVNetSegmenter.forward); on by default only on MPS."""
    if args.pad_nodes is not None:
        return args.pad_nodes, args.pad_edges
    return (128, 512) if device.type == "mps" else (0, 0)


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.n = num_classes
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, preds, labels):
        idx = labels.cpu() * self.n + preds.cpu()
        self.mat += torch.bincount(idx, minlength=self.n * self.n).view(self.n, self.n)

    def compute(self):
        tp = self.mat.diag().float()
        support = self.mat.sum(1).float()
        union = support + self.mat.sum(0).float() - tp
        present = support > 0
        iou = torch.where(union > 0, tp / union.clamp(min=1), torch.zeros_like(tp))
        return {
            "accuracy": (tp.sum() / self.mat.sum().clamp(min=1)).item(),
            "mean_iou": iou[present].mean().item() if present.any() else 0.0,
            "per_class_iou": iou.tolist(),
        }


def run_epoch(model, loader, device, optimizer=None, pad=(0, 0)):
    training = optimizer is not None
    model.train(training)
    cm = ConfusionMatrix(model.seg.linear3.out_features)
    total_loss, total_faces = 0.0, 0
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader):
            if device.type == "mps" and step % 200 == 0:
                torch.mps.empty_cache()  # drop buffers cached for batch shapes we no longer see
            batch = batch.to(device)
            logits = model(batch, *pad)
            loss = F.cross_entropy(logits, batch.y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.y.numel()
            total_faces += batch.y.numel()
            cm.update(logits.argmax(-1), batch.y)
    metrics = cm.compute()
    metrics["loss"] = total_loss / max(total_faces, 1)
    return metrics


def train(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    if args.resume:
        run_dir = pathlib.Path(args.resume).parent
    else:
        run_dir = pathlib.Path(args.results_dir) / args.experiment_name / time.strftime("%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=1))
    print(f"Device: {device} | run dir: {run_dir}")

    dataset_cls = DATASETS[args.dataset][0]
    train_set = dataset_cls(args.data_dir, "train", random_rotate=args.random_rotate,
                            max_samples=args.max_train_samples)
    # --val_split train (with --max_val_samples == --max_train_samples) gives an overfitting check
    val_set = dataset_cls(args.data_dir, args.val_split, max_samples=args.max_val_samples)
    # drop_last avoids a size-1 BatchNorm batch, matching the original UV-Net loader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    model = UVNetSegmenter(num_classes=args.num_classes).to(device)
    pad = bucket_sizes(args, device)
    print(f"Shape bucketing (faces, edges): {pad}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(run_dir)

    best_val, start_epoch = float("inf"), 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_val, start_epoch = ckpt["best_val"], ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer, pad)
        va = run_epoch(model, val_loader, device, pad=pad)
        for split, m in (("train", tr), ("val", va)):
            for k in ("loss", "accuracy", "mean_iou"):
                writer.add_scalar(f"{split}/{k}", m[k], epoch)
        print(f"epoch {epoch:3d} | {time.time() - t0:5.1f}s | "
              f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} iou {tr['mean_iou']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} iou {va['mean_iou']:.4f}")
        is_best = va["loss"] < best_val
        best_val = min(best_val, va["loss"])
        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "best_val": best_val, "num_classes": args.num_classes, "val": va}
        torch.save(ckpt, run_dir / "last.pt")
        if is_best:
            torch.save(ckpt, run_dir / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            torch.save(ckpt, run_dir / f"epoch_{epoch:03d}.pt")
    writer.close()
    print(f"Best val loss {best_val:.4f}. Checkpoint: {run_dir / 'best.pt'}")


def test(args):
    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = UVNetSegmenter(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model"])
    dataset_cls, class_names = DATASETS[args.dataset]
    test_set = dataset_cls(args.data_dir, "test", max_samples=args.max_test_samples)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    m = run_epoch(model, loader, device, pad=bucket_sizes(args, device))
    print(f"Test: loss {m['loss']:.4f} | face accuracy {m['accuracy']:.4f} | mean IoU {m['mean_iou']:.4f}")
    names = class_names if ckpt["num_classes"] == len(class_names) else range(ckpt["num_classes"])
    for name, iou in zip(names, m["per_class_iou"]):
        print(f"  {str(name):28s} IoU {iou:.4f}")
    out = pathlib.Path(args.checkpoint).with_name("test_metrics.json")
    out.write_text(json.dumps(m, indent=1))


def main():
    p = argparse.ArgumentParser("UV-Net (PyG) segmentation on MFCAD++ / Fusion 360 Gallery")
    p.add_argument("mode", choices=("train", "test"))
    p.add_argument("--dataset", default="mfcadpp", choices=tuple(DATASETS))
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--resume", default=None, help="last.pt of an interrupted run; continues in the same run dir")
    p.add_argument("--num_classes", type=int, default=None, help="Default: the dataset's number of classes")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_every", type=int, default=10,
                   help="Also keep a checkpoint every N epochs as epoch_NNN.pt (0: only best.pt and last.pt)")
    p.add_argument("--val_split", default="val", choices=("train", "val", "test"),
                   help="Split used for validation; 'train' for an overfitting sanity check")
    p.add_argument("--random_rotate", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--pad_nodes", type=int, default=None, help="Bucket face count to this multiple (default: 128 on MPS, off elsewhere)")
    p.add_argument("--pad_edges", type=int, default=512, help="Bucket edge count to this multiple (used with --pad_nodes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--experiment_name", default="mfcadpp_segmentation")
    args = p.parse_args()
    if args.num_classes is None:
        args.num_classes = DATASETS[args.dataset][0].num_classes
    if args.mode == "train":
        train(args)
    else:
        assert args.checkpoint, "--checkpoint is required for test"
        test(args)


if __name__ == "__main__":
    main()
