from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, heads: int = 2) -> None:
        super().__init__(); self.heads = heads; self.out_features = out_features
        self.projection = nn.Linear(in_features, heads * out_features, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, 2 * out_features)); nn.init.xavier_uniform_(self.attention)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        projected = self.projection(x).view(x.size(0), self.heads, self.out_features)
        source, target = edge_index
        scores = F.leaky_relu((torch.cat([projected[source], projected[target]], dim=-1) * self.attention).sum(-1))
        result = torch.zeros_like(projected)
        for node in range(x.size(0)):
            mask = target == node
            if mask.any():
                weights = torch.softmax(scores[mask], dim=0).unsqueeze(-1)
                result[node] = (projected[source[mask]] * weights).sum(0)
        return result.reshape(x.size(0), self.heads * self.out_features)


class OCRGATEncoder(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, heads: int = 2) -> None:
        super().__init__(); self.layer1 = GATLayer(input_dim, hidden_dim, heads); self.layer2 = GATLayer(hidden_dim * heads, hidden_dim, 1)

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        return self.layer2(torch.relu(self.layer1(node_features, edge_index)), edge_index)


class PairwiseOrderModel(nn.Module):
    """GAT encoder followed by an explicit pairwise precedence scorer."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32) -> None:
        super().__init__(); self.input_dim = input_dim; self.encoder = OCRGATEncoder(input_dim, hidden_dim, heads=2); self.scorer = nn.Sequential(nn.Linear(hidden_dim * 2 + input_dim * 2 + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        embeddings = self.encoder(node_features, edge_index); count = embeddings.size(0)
        left = embeddings[:, None, :].expand(count, count, -1); right = embeddings[None, :, :].expand(count, count, -1)
        left_features = node_features[:, None, :].expand(count, count, -1)
        right_features = node_features[None, :, :].expand(count, count, -1)
        native_rank_diff = left_features[..., -1:] - right_features[..., -1:]
        raw = self.scorer(torch.cat([left, right, left_features, right_features, native_rank_diff], dim=-1)).squeeze(-1)
        return raw - raw.transpose(0, 1)
