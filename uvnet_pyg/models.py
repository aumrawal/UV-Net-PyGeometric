"""UV-Net face segmentation model (per-face classification) on PyTorch Geometric batches."""
import torch
from torch import nn
import torch.nn.functional as F

from uvnet_pyg.encoders import UVNetCurveEncoder, UVNetSurfaceEncoder, UVNetGraphEncoder


class _NonLinearClassifier(nn.Module):
    """3-layer MLP with linear outputs."""

    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.dp1(F.relu(self.bn1(self.linear1(x))))
        x = self.dp2(F.relu(self.bn2(self.linear2(x))))
        return self.linear3(x)


class UVNetSegmenter(nn.Module):
    """
    Expects a torch_geometric Batch with:
        x          : num_faces x 7 x U x V   face UV-grids (xyz, normal, trim mask), channels-first
        edge_attr  : num_edges x 6 x U       edge U-grids (xyz, tangent), channels-first
        edge_index : 2 x num_edges
        batch      : num_faces
    Returns per-face logits (num_faces x num_classes).
    """

    def __init__(
        self,
        num_classes,
        crv_in_channels=6,
        srf_in_channels=7,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
    ):
        super().__init__()
        self.curv_encoder = UVNetCurveEncoder(in_channels=crv_in_channels, output_dims=crv_emb_dim)
        self.surf_encoder = UVNetSurfaceEncoder(in_channels=srf_in_channels, output_dims=srf_emb_dim)
        self.graph_encoder = UVNetGraphEncoder(srf_emb_dim, crv_emb_dim, graph_emb_dim)
        # Node embeddings (hidden_dim=64 == srf_emb_dim) concatenated with the global embedding
        self.seg = _NonLinearClassifier(graph_emb_dim + srf_emb_dim, num_classes, dropout=dropout)

    def forward(self, data, pad_nodes_to=0, pad_edges_to=0):
        """
        pad_nodes_to / pad_edges_to (optional): round the face/edge counts up to a multiple of
        these values by appending a disconnected dummy graph, whose outputs are dropped. On MPS
        every new tensor shape triggers a kernel recompile (~0.5 s), so bucketing shapes makes
        training several times faster. Real-face outputs are unchanged in eval mode; in train
        mode only the BatchNorm statistics see the few extra (duplicated) rows.
        """
        x, edge_attr, edge_index, batch = data.x, data.edge_attr, data.edge_index, data.batch
        num_faces = x.size(0)
        if pad_nodes_to or pad_edges_to:
            x, edge_attr, edge_index, batch = _pad_with_dummy_graph(
                x, edge_attr, edge_index, batch, data.num_graphs, pad_nodes_to, pad_edges_to
            )

        hidden_crv_feat = self.curv_encoder(edge_attr)
        hidden_srf_feat = self.surf_encoder(x)
        node_emb, graph_emb = self.graph_encoder(edge_index, hidden_srf_feat, hidden_crv_feat, batch)
        # Broadcast each graph's global embedding to its faces (replaces DGL repeat_interleave)
        local_global_feat = torch.cat((node_emb, graph_emb[batch]), dim=1)
        return self.seg(local_global_feat)[:num_faces]


def _pad_with_dummy_graph(x, edge_attr, edge_index, batch, num_graphs, node_mult, edge_mult):
    """Append a dummy graph (copies of random real faces/edges) so sizes hit bucket multiples."""
    n, e = x.size(0), edge_attr.size(0)
    pad_n = (-n) % node_mult if node_mult else 0
    pad_e = (-e) % edge_mult if edge_mult else 0
    if pad_n == 0 and pad_e == 0:
        return x, edge_attr, edge_index, batch
    if pad_n < 2:  # dummy edges need at least two dummy endpoints
        pad_n += node_mult if node_mult else 2
    dev = x.device
    node_src = torch.randint(0, n, (pad_n,), device=dev)
    edge_src = torch.randint(0, e, (pad_e,), device=dev)
    k = torch.arange(pad_e, device=dev)
    dummy_edges = torch.stack((n + k % pad_n, n + (k + 1) % pad_n))
    return (
        torch.cat((x, x[node_src])),
        torch.cat((edge_attr, edge_attr[edge_src])),
        torch.cat((edge_index, dummy_edges), dim=1),
        torch.cat((batch, torch.full((pad_n,), num_graphs, dtype=batch.dtype, device=dev))),
    )
