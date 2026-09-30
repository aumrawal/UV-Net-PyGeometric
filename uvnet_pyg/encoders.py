"""
UV-Net encoders ported from DGL to PyTorch Geometric.

The curve/surface CNNs are identical to the original UV-Net
(github.com/AutodeskAILab/UV-Net, uvnet/encoders.py). The graph encoder
replaces dgl.nn.NNConv / dgl MaxPooling with torch_geometric equivalents:

    DGL NNConv(aggregator="sum", bias=False)  ->  PyG NNConv(aggr="add", root_weight=False, bias=False)
    DGL MaxPooling()                          ->  PyG global_max_pool(x, batch)

Both compute h_i' = sum_{j->i} h_j @ W(e_ji), so the math is unchanged.
All ops used here run on CUDA and on Apple-silicon MPS.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_max_pool


def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )


class _MLP(nn.Module):
    """MLP with linear output (BatchNorm + ReLU between hidden layers)."""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        self.num_layers = num_layers
        if num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)]
            )

    def forward(self, x):
        if self.num_layers == 1:
            return self.linear(x)
        h = x
        for i in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[i](self.linears[i](h)))
        return self.linears[-1](h)


def _kaiming_init(module, types):
    for m in module.modules():
        if isinstance(m, types):
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class UVNetCurveEncoder(nn.Module):
    """1D CNN over edge U-grids: (num_edges, 6, num_u) -> (num_edges, output_dims)."""

    def __init__(self, in_channels=6, output_dims=64):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims)
        _kaiming_init(self, (nn.Linear, nn.Conv1d))

    def forward(self, x):
        assert x.size(1) == self.in_channels
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.final_pool(x).flatten(1)
        return self.fc(x)


class UVNetSurfaceEncoder(nn.Module):
    """2D CNN over face UV-grids: (num_faces, 7, num_u, num_v) -> (num_faces, output_dims)."""

    def __init__(self, in_channels=7, output_dims=64):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = _conv2d(64, 128, 3, padding=1)
        self.conv3 = _conv2d(128, 256, 3, padding=1)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims)
        _kaiming_init(self, (nn.Linear, nn.Conv2d))

    def forward(self, x):
        assert x.size(1) == self.in_channels
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.final_pool(x).flatten(1)
        return self.fc(x)


class _EdgeConv(nn.Module):
    """Eq. 2 of the paper: update edge features from their endpoint node features."""

    def __init__(self, edge_feats, out_feats, node_feats, num_mlp_layers=2, hidden_mlp_dim=64):
        super().__init__()
        self.proj = _MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = _MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, edge_index, nfeat, efeat):
        src, dst = edge_index
        agg = self.proj(nfeat[src]) + self.proj(nfeat[dst])
        h = self.mlp((1 + self.eps) * efeat + agg)
        return F.leaky_relu(self.batchnorm(h))


class _NodeConv(nn.Module):
    """Eq. 1 of the paper: edge-conditioned convolution (NNConv) over the face-adjacency graph."""

    def __init__(self, node_feats, out_feats, edge_feats, num_mlp_layers=2, hidden_mlp_dim=64):
        super().__init__()
        self.gconv = NNConv(
            in_channels=node_feats,
            out_channels=out_feats,
            nn=nn.Linear(edge_feats, node_feats * out_feats),
            aggr="add",
            root_weight=False,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_feats)
        # Kept identical to the original (input_dim=node_feats even though gconv outputs
        # out_feats); UV-Net always uses node_feats == out_feats == hidden_dim here.
        self.mlp = _MLP(num_mlp_layers, node_feats, hidden_mlp_dim, out_feats)
        self.eps = nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, edge_index, nfeat, efeat):
        h = (1 + self.eps) * nfeat
        h = self.gconv(h, edge_index, efeat)
        h = self.mlp(h)
        return F.leaky_relu(self.batchnorm(h))


class UVNetGraphEncoder(nn.Module):
    """Message passing over the face-adjacency graph (Section 3.2 of the paper)."""

    def __init__(
        self,
        input_dim,
        input_edge_dim,
        output_dim,
        hidden_dim=64,
        num_layers=3,
        num_mlp_layers=2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.node_conv_layers = nn.ModuleList()
        self.edge_conv_layers = nn.ModuleList()
        for layer in range(num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                _NodeConv(node_feats, hidden_dim, edge_feats, num_mlp_layers, hidden_dim)
            )
            self.edge_conv_layers.append(
                _EdgeConv(edge_feats, hidden_dim, node_feats, num_mlp_layers, hidden_dim)
            )

        self.linears_prediction = nn.ModuleList(
            [nn.Linear(input_dim if layer == 0 else hidden_dim, output_dim) for layer in range(num_layers)]
        )
        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)

    def forward(self, edge_index, h, efeat, batch):
        """
        Args:
            edge_index (LongTensor): 2 x num_edges (directed, both directions present)
            h (Tensor): num_nodes x input_dim face embeddings
            efeat (Tensor): num_edges x input_edge_dim edge embeddings
            batch (LongTensor): num_nodes graph-assignment vector
        Returns:
            (node embeddings num_nodes x hidden_dim, graph embeddings num_graphs x output_dim)
        """
        hidden_rep = [h]
        he = efeat
        for i in range(self.num_layers - 1):
            h = self.node_conv_layers[i](edge_index, h, he)
            he = self.edge_conv_layers[i](edge_index, h, he)
            hidden_rep.append(h)

        out = self.drop1(hidden_rep[-1])
        score_over_layer = 0
        for i, hr in enumerate(hidden_rep):
            pooled = global_max_pool(hr, batch)
            score_over_layer = score_over_layer + self.drop(self.linears_prediction[i](pooled))
        return out, score_over_layer
