"""GAT (main) and GCN (baseline) with per-attack curve heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

from gnn.constants import ATTACK_ORDER, N_CURVE_POINTS, parse_attacks
from gnn.edge_utils import align_edge_tensors_for_gat


class ResilienceGNN(nn.Module):
    """
    Graph-level regression: pool -> one linear head per attack type.
    Default: betweenness + capacity + random (30-dim); use attacks=('betweenness','random') for 20-dim.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 32,
        gat_heads: int = 4,
        edge_dim: int = 1,
        model_type: str = "gat",
        dropout: float = 0.1,
        n_out_points: int | None = None,
        global_dim: int = 0,
        attacks: str | tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.edge_dim = edge_dim
        self.global_dim = int(global_dim or 0)
        self.n_out_points = int(n_out_points or N_CURVE_POINTS)
        self.attacks: tuple[str, ...] = (
            parse_attacks(attacks) if isinstance(attacks, str) else tuple(attacks or ATTACK_ORDER)
        )
        self._last_attn = None

        if self.model_type == "gat":
            self.conv1 = GATConv(
                in_channels,
                hidden_channels // gat_heads,
                heads=gat_heads,
                edge_dim=edge_dim,
                dropout=dropout,
            )
            self.conv2 = GATConv(
                hidden_channels,
                hidden_channels,
                heads=1,
                concat=False,
                edge_dim=edge_dim,
                dropout=dropout,
            )
        elif self.model_type == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif self.model_type == "mlp":
            # Graph-agnostic: per-node MLP + global mean pool (no edge_index)
            self.node_encoder = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.dropout = nn.Dropout(dropout)
        k = self.n_out_points
        head_in = hidden_channels + self.global_dim
        self.heads = nn.ModuleDict({a: nn.Linear(head_in, k) for a in self.attacks})
        # Legacy attribute names (may be absent when attack dropped)
        self.head_betweenness = self.heads["betweenness"] if "betweenness" in self.heads else None
        self.head_capacity = self.heads["capacity"] if "capacity" in self.heads else None
        self.head_random = self.heads["random"] if "random" in self.heads else None

    def _fuse_global(self, graph_embed: torch.Tensor, data) -> torch.Tensor:
        """graph_embed = pool(GAT); combined = cat([graph_embed, global_features])."""
        if self.global_dim <= 0:
            return graph_embed
        gf = getattr(data, "gf", None)
        if gf is None:
            raise ValueError("global_dim>0 but batch has no data.gf (use step6 --global-features)")
        gf = gf.float()
        b = int(graph_embed.size(0))
        f = self.global_dim
        if gf.dim() == 1:
            if gf.numel() == b * f:
                gf = gf.view(b, f)
            elif gf.numel() == f and b == 1:
                gf = gf.view(1, f)
            else:
                raise ValueError(f"data.gf 1D numel={gf.numel()} for batch={b}, global_dim={f}")
        elif gf.dim() == 2:
            if gf.size(0) != b:
                raise ValueError(f"data.gf rows {gf.size(0)} != batch size {b}")
        else:
            raise ValueError(f"data.gf must be 1D or 2D, got shape {tuple(gf.shape)}")
        if gf.size(-1) != f:
            raise ValueError(f"data.gf last dim {gf.size(-1)} != global_dim {f}")
        return torch.cat([graph_embed, gf], dim=1)

    def _predict_heads(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts = {a: self.heads[a](h) for a in self.attacks}
        y = torch.cat([parts[a] for a in self.attacks], dim=-1)
        return y, parts

    def forward(self, data, return_attention: bool = False):
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        edge_attr = getattr(data, "edge_attr", None)

        if self.model_type == "gat":
            if edge_attr is None:
                raise ValueError("GAT model requires edge_attr (normalized distance)")
            edge_index, edge_attr = align_edge_tensors_for_gat(edge_index, edge_attr)
            h, (ei1, alpha1) = self.conv1(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            h = self.dropout(h.relu())
            h, (ei2, alpha2) = self.conv2(
                h, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            self._last_attn = {
                "edge_index": ei2.detach().cpu(),
                "alpha": alpha2.detach().cpu(),
            }
        elif self.model_type == "gcn":
            h = self.conv1(x, edge_index).relu()
            h = self.dropout(h)
            h = self.conv2(h, edge_index)
            self._last_attn = None
        elif self.model_type == "mlp":
            self._last_attn = None
            h = self.node_encoder(x)
            h = self.dropout(h)
            h = global_mean_pool(h, batch)
            h = self._fuse_global(h, data)
            y, parts = self._predict_heads(h)
            y_b = parts.get("betweenness")
            y_c = parts.get("capacity")
            y_r = parts.get("random")
            if return_attention:
                return y, y_b, y_c, y_r, None
            return y, y_b, y_c, y_r
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        h = global_mean_pool(h, batch)
        h = self._fuse_global(h, data)
        y, parts = self._predict_heads(h)
        y_b = parts.get("betweenness")
        y_c = parts.get("capacity")
        y_r = parts.get("random")

        if return_attention:
            return y, y_b, y_c, y_r, self._last_attn
        return y, y_b, y_c, y_r

    def predict_heads(self, data):
        y, y_b, y_c, y_r = self.forward(data)
        return y, y_b, y_c, y_r
