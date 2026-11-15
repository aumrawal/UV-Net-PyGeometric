"""
UV-Net face segmentation on Fusion 360 Gallery: the single file the agent edits.

Baseline = the UV-Net recipe of UV-Net-PyGeometric (Jayaraman et al., CVPR 2021, ported to PyG):
surface 2D CNN + curve 1D CNN, two rounds of edge-conditioned message passing (NNConv + edge
update), max-pooled part embedding, per-face MLP. Adam, lr 1e-3, batch 64, no augmentation.
Trains for TIME_BUDGET seconds of wall clock, then scores with prepare.evaluate.

    python train.py
"""
import math
import os
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_max_pool

from prepare import TIME_BUDGET, NUM_CLASSES, make_train_dataset, evaluate

# ---------------------------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------------------------
SEED = int(os.environ.get("AR_SEED", 0))   # set by the runner (ar.py --seeds)
BATCH_SIZE = 64
LR = 2e-3          # peak; warmup then cosine decay to 0 over the time budget
WARMUP_STEPS = 200
RANDOM_ROTATE = False
NUM_WORKERS = 0

# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------
def _conv1d(cin, cout, k=3, padding=0):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=padding, bias=False), nn.BatchNorm1d(cout), nn.LeakyReLU())


def _conv2d(cin, cout, k=3, padding=0):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=padding, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU())


def _fc(cin, cout):
    return nn.Sequential(nn.Linear(cin, cout, bias=False), nn.BatchNorm1d(cout), nn.LeakyReLU())


def _kaiming_init(module, types):
    for m in module.modules():
        if isinstance(m, types):
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class MLP(nn.Module):
    """MLP with linear output (BatchNorm + ReLU between hidden layers)."""

    def __init__(self, num_layers, din, dhidden, dout):
        super().__init__()
        dims = [din] + [dhidden] * (num_layers - 1) + [dout]
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.BatchNorm1d(d) for d in dims[1:-1])

    def forward(self, x):
        for lin, bn in zip(self.linears[:-1], self.norms):
            x = F.relu(bn(lin(x)))
        return self.linears[-1](x)


class CurveEncoder(nn.Module):
    """1D CNN over edge U-grids: E x 6 x U -> E x dout."""

    def __init__(self, cin=6, dout=64):
        super().__init__()
        self.convs = nn.Sequential(_conv1d(cin, 64, padding=1), _conv1d(64, 128, padding=1), _conv1d(128, 256, padding=1))
        self.fc = _fc(256, dout)
        _kaiming_init(self, (nn.Linear, nn.Conv1d))

    def forward(self, x):
        return self.fc(self.convs(x).mean(-1))


class SurfaceEncoder(nn.Module):
    """2D CNN over face UV-grids: F x 7 x U x V -> F x dout."""

    def __init__(self, cin=7, dout=64):
        super().__init__()
        self.convs = nn.Sequential(_conv2d(cin, 64, padding=1), _conv2d(64, 128, padding=1), _conv2d(128, 256, padding=1))
        self.fc = _fc(256, dout)
        _kaiming_init(self, (nn.Linear, nn.Conv2d))

    def forward(self, x):
        return self.fc(self.convs(x).mean((-2, -1)))


class EdgeConv(nn.Module):
    """Eq. 2 of the paper: update edge features from their two faces."""

    def __init__(self, edge_feats, out_feats, node_feats, hidden=64):
        super().__init__()
        self.proj = MLP(1, node_feats, hidden, edge_feats)
        self.mlp = MLP(2, edge_feats, hidden, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, edge_index, h, he):
        src, dst = edge_index
        return F.leaky_relu(self.bn(self.mlp((1 + self.eps) * he + self.proj(h[src]) + self.proj(h[dst]))))


class NodeConv(nn.Module):
    """Eq. 1 of the paper: edge-conditioned convolution (NNConv) over the face-adjacency graph."""

    def __init__(self, node_feats, out_feats, edge_feats, hidden=64):
        super().__init__()
        self.gconv = NNConv(node_feats, out_feats, nn=nn.Linear(edge_feats, node_feats * out_feats),
                            aggr="add", root_weight=False, bias=False)
        self.bn = nn.BatchNorm1d(out_feats)
        self.mlp = MLP(2, out_feats, hidden, out_feats)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, edge_index, h, he):
        return F.leaky_relu(self.bn(self.mlp(self.gconv((1 + self.eps) * h, edge_index, he))))


class GraphEncoder(nn.Module):
    """Message passing over faces; returns per-face embeddings and a pooled part embedding."""

    def __init__(self, din, dedge, dgraph, hidden=64, num_layers=3):
        super().__init__()
        self.node_convs, self.edge_convs = nn.ModuleList(), nn.ModuleList()
        for i in range(num_layers - 1):
            nf, ef = (din, dedge) if i == 0 else (hidden, hidden)
            self.node_convs.append(NodeConv(nf, hidden, ef, hidden))
            self.edge_convs.append(EdgeConv(ef, hidden, hidden, hidden))  # its node input is the node conv's output
        self.readouts = nn.ModuleList(nn.Linear(din if i == 0 else hidden, dgraph) for i in range(num_layers))
        self.drop1, self.drop = nn.Dropout(0.3), nn.Dropout(0.5)

    def forward(self, edge_index, h, he, batch):
        reps = [h]
        for node_conv, edge_conv in zip(self.node_convs, self.edge_convs):
            h = node_conv(edge_index, h, he)
            he = edge_conv(edge_index, h, he)
            reps.append(h)
        graph = sum(self.drop(lin(global_max_pool(r, batch))) for lin, r in zip(self.readouts, reps))
        return self.drop1(reps[-1]), graph


class Classifier(nn.Module):
    def __init__(self, din, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        _kaiming_init(self, (nn.Linear,))

    def forward(self, x):
        return self.net(x)


def edge_convexity(x, edge_index):
    """Per directed edge: cos of the angle between the two faces' mean normals, and the convexity
    signs (c_dst - c_src) . n_src and (c_src - c_dst) . n_dst on unit offsets (< 0: convex)."""
    mask = x[:, 6:7]
    w = mask.sum((-2, -1)).clamp(min=1e-6)
    n = F.normalize((x[:, 3:6] * mask).sum((-2, -1)) / w, dim=1)
    c = (x[:, 0:3] * mask).sum((-2, -1)) / w
    src, dst = edge_index
    off = F.normalize(c[dst] - c[src], dim=1)
    return torch.stack([(n[src] * n[dst]).sum(1), (off * n[src]).sum(1), (-off * n[dst]).sum(1)], dim=1)


def edge_local_convexity(x, edge_attr, edge_index):
    """Convexity at the edge itself, for curved faces too: the normals of the two faces at their grid
    points nearest the edge midpoint, their cosine, and the sign of each face's centroid offset
    from the midpoint along the other face's local normal (< 0: convex)."""
    mid = edge_attr[:, 0:3, edge_attr.size(2) // 2]                       # E x 3
    pts = x[:, 0:3].flatten(2)                                            # F x 3 x UV
    nrm = x[:, 3:6].flatten(2)
    mask = x[:, 6].flatten(1) > 0.5                                       # F x UV
    w = x[:, 6:7].sum((-2, -1)).clamp(min=1e-6)
    cen = (x[:, 0:3] * x[:, 6:7]).sum((-2, -1)) / w                       # F x 3
    src, dst = edge_index

    def nearest_normal(f):
        d = (pts[f] - mid.unsqueeze(-1)).square().sum(1)                  # E x UV
        d = d.masked_fill(~mask[f], float("inf"))
        idx = d.argmin(1)
        return F.normalize(nrm[f].gather(2, idx.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1), dim=1)

    ns, nd = nearest_normal(src), nearest_normal(dst)
    od = F.normalize(cen[dst] - mid, dim=1)
    os_ = F.normalize(cen[src] - mid, dim=1)
    return torch.stack([(ns * nd).sum(1), (od * ns).sum(1), (os_ * nd).sum(1)], dim=1)


class UVNetSegmenter(nn.Module):
    def __init__(self, num_classes, crv_dim=64, srf_dim=64, graph_dim=128):
        super().__init__()
        self.edge_geo = nn.Sequential(nn.Linear(6, crv_dim), nn.LeakyReLU(), nn.Linear(crv_dim, crv_dim))
        self.curve_encoder = CurveEncoder(6, crv_dim)
        self.surface_encoder = SurfaceEncoder(7, srf_dim)
        self.graph_encoder = GraphEncoder(srf_dim, crv_dim, graph_dim, hidden=128)
        self.seg = Classifier(graph_dim + 128, num_classes)

    def forward(self, data):
        he = self.curve_encoder(data.edge_attr) + self.edge_geo(torch.cat([edge_convexity(data.x, data.edge_index),
            edge_local_convexity(data.x, data.edge_attr, data.edge_index)], 1))
        h = self.surface_encoder(data.x)
        node_emb, graph_emb = self.graph_encoder(data.edge_index, h, he, data.batch)
        return self.seg(torch.cat((node_emb, graph_emb[data.batch]), dim=1))


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------
def main():
    t_start = time.time()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = make_train_dataset(random_rotate=RANDOM_ROTATE)
    # drop_last avoids a size-1 BatchNorm batch, as in the original UV-Net loader
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                        num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    model = UVNetSegmenter(NUM_CLASSES).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print(f"device {device} | {len(train_set)} training parts | {num_params:,} parameters | budget {TIME_BUDGET}s")

    # The budget is wall clock from the first batch on, data loading included
    step, epoch, train_time, done = 0, 0, 0.0, False
    t_train = time.time()
    while not done:
        epoch += 1
        model.train()
        loss_sum, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(batch).float(), batch.y)
            progress = min(train_time / TIME_BUDGET, 1.0)
            for g in optimizer.param_groups:
                g["lr"] = LR * min(1.0, (step + 1) / WARMUP_STEPS) * 0.5 * (1 + math.cos(math.pi * progress))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum, n = loss_sum + loss.item(), n + 1   # .item() syncs with the GPU
            train_time = time.time() - t_train
            step += 1
            if train_time >= TIME_BUDGET:
                done = True
                break
        print(f"epoch {epoch:3d} | step {step:6d} | train loss {loss_sum / max(n, 1):.4f} | "
              f"train time {train_time:6.0f}s", flush=True)

    result = evaluate(model, device)
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    print("---")
    print(f"val_miou:         {result['val_miou']:.6f}")
    print(f"val_acc:          {result['val_acc']:.6f}")
    print(f"val_loss:         {result['val_loss']:.6f}")
    print(f"training_seconds: {train_time:.1f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_mb:.1f}")
    print(f"num_epochs:       {epoch}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.2f}")
    for name, iou in result["per_class_iou"].items():
        print(f"iou_{name}: {iou:.4f}")


if __name__ == "__main__":
    main()
