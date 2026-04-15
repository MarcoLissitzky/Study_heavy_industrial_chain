from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 617,
        hidden_channels: int = 64,
        out_channels: int = 16,
        heads: int = 4,
        gat_dropout: float = 0.2,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=float(gat_dropout),
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=float(gat_dropout),
        )
        if activation == "elu":
            self.activation: nn.Module = nn.ELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected x to have shape [N, F], got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected x feature dim {self.in_channels}, got {x.size(1)}")
        h = self.conv1(x, edge_index)
        h = self.activation(h)
        z = self.conv2(h, edge_index)
        return z


class EdgeMLPDecoder(nn.Module):
    def __init__(self, node_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        pair = torch.cat([z[src], z[dst]], dim=1)
        logits = self.mlp(pair).squeeze(-1)
        return logits
