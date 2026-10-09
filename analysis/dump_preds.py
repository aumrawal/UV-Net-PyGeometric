import sys, numpy as np, torch
sys.path.insert(0, ".")
from torch_geometric.loader import DataLoader
from datasets.mfcadpp import MFCADPPDataset
from uvnet_pyg.models import UVNetSegmenter

run = "results/mfcadpp_full/0930_000433"
dev = torch.device("mps")
ck = torch.load(f"{run}/best.pt", map_location="cpu", weights_only=False)
model = UVNetSegmenter(num_classes=ck["num_classes"]).to(dev)
model.load_state_dict(ck["model"]); model.eval()
ds = MFCADPPDataset("data/processed", "test")
loader = DataLoader(ds, batch_size=64, shuffle=False)
P, Y, G, C = [], [], [], []
off = 0
with torch.no_grad():
    for b in loader:
        b = b.to(dev)
        lg = model(b, 128, 512)
        pr = lg.softmax(-1)
        P.append(pr.argmax(-1).cpu()); Y.append(b.y.cpu()); G.append(b.batch.cpu() + off)
        C.append(pr.max(-1).values.cpu())
        off += b.num_graphs
P, Y, G, C = (torch.cat(t).numpy() for t in (P, Y, G, C))
print("epoch", ck["epoch"], "faces", len(Y), "models", off, "acc", (P == Y).mean())
np.savez(sys.argv[1], pred=P, y=Y, graph=G, conf=C)
