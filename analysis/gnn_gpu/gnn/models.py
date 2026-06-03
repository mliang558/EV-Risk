"""GAT / Graph Transformer / GCN / MLP with per-attack curve heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, TransformerConv, global_mean_pool

from gnn.constants import ATTACK_ORDER, N_CURVE_POINTS, parse_attacks
from gnn.edge_utils import align_edge_tensors_for_gat


def model_uses_edge_attr(model_type: str) -> bool:
    return str(model_type).lower() in ("gat", "transformer", "gt", "graph_transformer")


def resilience_kwargs_from_ckpt(ckpt: dict) -> dict:
    """Rebuild ResilienceGNN kwargs from a saved checkpoint dict."""
    mt = str(ckpt.get("model_type", "gat")).lower()
    return {
        "in_channels": 4,
        "hidden_channels": int(ckpt.get("hidden", 32)),
        "gat_heads": int(ckpt.get("gat_heads", 4)),
        "edge_dim": 1 if model_uses_edge_attr(mt) else 0,
        "model_type": mt,
        "n_out_points": int(ckpt.get("curve_points", 10)),
        "global_dim": int(ckpt.get("global_dim", 0)),
        "num_layers": int(ckpt.get("num_layers", 2)),
        "transformer_heads": int(ckpt.get("transformer_heads", ckpt.get("gat_heads", 4))),
        "attacks": tuple(ckpt.get("attacks", ATTACK_ORDER)),
    }


class ResilienceGNN(nn.Module):
    """
    Graph-level regression: message-passing stack -> pool -> linear head(s).

    - gat: stacked GATConv (default 2 layers; use num_layers=4|5 for deeper)
    - transformer: stacked TransformerConv (graph transformer style)
    - gcn / mlp: unchanged baselines
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
        num_layers: int = 2,
        transformer_heads: int | None = None,
    ):
        super().__init__()
        self.model_type = model_type.lower()
        if self.model_type in ("gt", "graph_transformer"):
            self.model_type = "transformer"
        self.edge_dim = edge_dim
        self.global_dim = int(global_dim or 0)
        self.n_out_points = int(n_out_points or N_CURVE_POINTS)
        self.hidden_channels = int(hidden_channels)
        self.gat_heads = int(gat_heads)
        self.num_layers = max(2, int(num_layers))
        self.transformer_heads = int(transformer_heads or gat_heads)
        self.attacks: tuple[str, ...] = (
            parse_attacks(attacks) if isinstance(attacks, str) else tuple(attacks or ATTACK_ORDER)
        )
        self._last_attn = None
        self.dropout = nn.Dropout(dropout)

        if self.model_type == "gat":
            self.convs = self._build_gat_stack(in_channels, edge_dim, dropout)
        elif self.model_type == "transformer":
            self.convs = self._build_transformer_stack(in_channels, edge_dim, dropout)
        elif self.model_type == "gcn":
            self.convs = self._build_gcn_stack(in_channels)
        elif self.model_type == "mlp":
            self.convs = nn.ModuleList()
            self.node_encoder = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
            )
        else:
            raise ValueError(
                f"Unknown model_type: {model_type!r}; use gat | transformer | gcn | mlp"
            )

        k = self.n_out_points
        head_in = hidden_channels + self.global_dim
        self.heads = nn.ModuleDict({a: nn.Linear(head_in, k) for a in self.attacks})
        # ModuleDict has no .get() on older PyTorch — use membership check
        self.head_betweenness = self.heads["betweenness"] if "betweenness" in self.heads else None
        self.head_capacity = self.heads["capacity"] if "capacity" in self.heads else None
        self.head_random = self.heads["random"] if "random" in self.heads else None

    def _build_gat_stack(self, in_channels: int, edge_dim: int, dropout: float) -> nn.ModuleList:
        h = self.hidden_channels
        heads = self.gat_heads
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            is_first = i == 0
            is_last = i == self.num_layers - 1
            in_dim = in_channels if is_first else h
            if is_last:
                layers.append(
                    GATConv(
                        in_dim,
                        h,
                        heads=1,
                        concat=False,
                        edge_dim=edge_dim,
                        dropout=dropout,
                    )
                )
            elif is_first:
                layers.append(
                    GATConv(
                        in_dim,
                        h // heads,
                        heads=heads,
                        edge_dim=edge_dim,
                        dropout=dropout,
                    )
                )
            else:
                layers.append(
                    GATConv(
                        in_dim,
                        h,
                        heads=1,
                        concat=False,
                        edge_dim=edge_dim,
                        dropout=dropout,
                    )
                )
        return layers

    def _build_transformer_stack(
        self, in_channels: int, edge_dim: int, dropout: float
    ) -> nn.ModuleList:
        h = self.hidden_channels
        heads = self.transformer_heads
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            is_first = i == 0
            is_last = i == self.num_layers - 1
            in_dim = in_channels if is_first else h
            if is_last:
                layers.append(
                    TransformerConv(
                        in_dim,
                        h,
                        heads=1,
                        concat=False,
                        dropout=dropout,
                        edge_dim=edge_dim if edge_dim > 0 else None,
                        beta=True,
                    )
                )
            elif is_first:
                layers.append(
                    TransformerConv(
                        in_dim,
                        h // heads,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                        edge_dim=edge_dim if edge_dim > 0 else None,
                        beta=True,
                    )
                )
            else:
                layers.append(
                    TransformerConv(
                        in_dim,
                        h // heads,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                        edge_dim=edge_dim if edge_dim > 0 else None,
                        beta=True,
                    )
                )
        return layers

    def _build_gcn_stack(self, in_channels: int) -> nn.ModuleList:
        h = self.hidden_channels
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = in_channels if i == 0 else h
            layers.append(GCNConv(in_dim, h))
        return layers

    def _fuse_global(self, graph_embed: torch.Tensor, data) -> torch.Tensor:
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
        elif gf.dim() == 2 and gf.size(0) != b:
            raise ValueError(f"data.gf rows {gf.size(0)} != batch size {b}")
        if gf.size(-1) != f:
            raise ValueError(f"data.gf last dim {gf.size(-1)} != global_dim {f}")
        return torch.cat([graph_embed, gf], dim=1)

    def _predict_heads(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts = {a: self.heads[a](h) for a in self.attacks}
        y = torch.cat([parts[a] for a in self.attacks], dim=-1)
        return y, parts

    def _forward_conv_stack(self, x, edge_index, edge_attr):
        h = x
        last_attn = None
        n = len(self.convs)

        if self.model_type == "gat":
            if edge_attr is None:
                raise ValueError("GAT requires edge_attr")
            edge_index, edge_attr = align_edge_tensors_for_gat(edge_index, edge_attr)
            for i, conv in enumerate(self.convs):
                is_last = i == n - 1
                if is_last:
                    h, (ei2, alpha2) = conv(
                        h,
                        edge_index,
                        edge_attr=edge_attr,
                        return_attention_weights=True,
                    )
                    last_attn = {"edge_index": ei2.detach().cpu(), "alpha": alpha2.detach().cpu()}
                else:
                    h = conv(h, edge_index, edge_attr=edge_attr)
                    h = self.dropout(h.relu())
            return h, last_attn

        if self.model_type == "transformer":
            edge_index, edge_attr = align_edge_tensors_for_gat(edge_index, edge_attr)
            for i, conv in enumerate(self.convs):
                is_last = i == n - 1
                h = conv(h, edge_index, edge_attr)
                if not is_last:
                    h = self.dropout(h.relu())
            return h, None

        if self.model_type == "gcn":
            for i, conv in enumerate(self.convs):
                h = conv(h, edge_index)
                if i < n - 1:
                    h = self.dropout(h.relu())
            return h, None

        raise ValueError(f"_forward_conv_stack: {self.model_type}")

    def forward(self, data, return_attention: bool = False):
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        edge_attr = getattr(data, "edge_attr", None)

        if self.model_type == "mlp":
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

        h, last_attn = self._forward_conv_stack(x, edge_index, edge_attr)
        self._last_attn = last_attn

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
