"""Dump per-face logits for a checkpoint: python analysis/dump_logits.py <dataset> <data_dir> <ckpt> <split> <out.npz>"""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from torch_geometric.loader import DataLoader
from datasets.fusion360 import Fusion360Dataset
from datasets.mfcadpp import MFCADPPDataset
from uvnet_pyg.models import UVNetSegmenter

name, data_dir, ckpt, split, out = sys.argv[1:6]
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
model = UVNetSegmenter(num_classes=ck["num_classes"]).to(dev)
model.load_state_dict(ck["model"]); model.eval()
ds = {"fusion360": Fusion360Dataset, "mfcadpp": MFCADPPDataset}[name](data_dir, split)
L, Y, G = [], [], []
off = 0
with torch.no_grad():
    for b in DataLoader(ds, batch_size=64, shuffle=False):
        b = b.to(dev)
        L.append(model(b, 128, 512).float().cpu()); Y.append(b.y.cpu()); G.append(b.batch.cpu() + off)
        off += b.num_graphs
L, Y, G = (torch.cat(t).numpy() for t in (L, Y, G))
print(split, "epoch", ck["epoch"], "faces", len(Y), "models", off, "acc", (L.argmax(1) == Y).mean())
np.savez(out, logits=L, y=Y, graph=G)
