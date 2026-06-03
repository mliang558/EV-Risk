"""Align edge_index and edge_attr for PyG GAT (fixes npz / batch mismatches)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops


def align_npz_edges(edge_index: np.ndarray, edge_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned (2, E) and (E,) arrays."""
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must be (2, E), got {ei.shape}")
    ew = np.asarray(edge_weight, dtype=np.float32).reshape(-1)
    n_e = ei.shape[1]

    if ew.size != n_e:
        if ew.size * 2 == n_e:
            ew = np.concatenate([ew, ew])
        elif n_e == 2 * ew.size and ew.size > 0:
            ei = np.concatenate([ei, ei[:, ::-1]], axis=1)
        elif n_e == 0:
            ew = np.zeros(0, dtype=np.float32)
        else:
            m = min(n_e, ew.size)
            ei = ei[:, :m]
            ew = ew[:m]

    if ei.shape[1] == 0:
        return ei, ew

    t_ei = torch.from_numpy(ei)
    t_ew = torch.from_numpy(ew)
    t_ei, t_ew = remove_self_loops(t_ei, t_ew)
    t_ei, t_ew = coalesce(t_ei, t_ew, reduce="mean")
    return t_ei.numpy(), t_ew.numpy()


def npz_to_pyg_edges(edge_index: np.ndarray, edge_weight: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    ei, ew = align_npz_edges(edge_index, edge_weight)
    return torch.from_numpy(ei), torch.from_numpy(ew).view(-1, 1)


def align_edge_tensors_for_gat(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align edge_index.size(1) with edge_attr.size(0) before GATConv.
    Safe to call on every forward pass (Step 6/7/8).
    """
    if edge_index is None or edge_index.numel() == 0:
        z = edge_index.new_zeros((2, 0))
        a = torch.zeros((0, 1), dtype=torch.float32, device=edge_index.device)
        return z, a

    ne = int(edge_index.size(1))
    if edge_attr is None:
        return edge_index, torch.ones((ne, 1), dtype=torch.float32, device=edge_index.device)

    rows = int(edge_attr.size(0))
    if rows == ne:
        return edge_index, edge_attr.reshape(-1)[:ne].reshape(ne, 1).float()

    ew = edge_attr.detach().reshape(-1).float()
    if ew.numel() == ne:
        return edge_index, ew.view(ne, 1)
    if ew.numel() * 2 == ne:
        return edge_index, torch.cat([ew, ew]).view(ne, 1)
    if ne == 2 * ew.numel() and ew.numel() > 0:
        return edge_index[:, : ew.numel()], ew.view(-1, 1)

    m = min(ne, ew.numel())
    return edge_index[:, :m], ew[:m].view(m, 1)


def edge_attr_row_count(data: Data) -> int:
    """PyG GAT needs edge_attr.size(0) == edge_index.size(1) (not numel)."""
    if data.edge_attr is None:
        return 0
    return int(data.edge_attr.size(0))


def force_sync_edge_attr(data: Data) -> Data:
    """Last-resort torch-side align (handles [E,1] vs [E/2,2] false numel match)."""
    if data.edge_index is None or data.edge_index.numel() == 0:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        return data

    ne = int(data.edge_index.size(1))
    if data.edge_attr is None:
        data.edge_attr = torch.ones((ne, 1), dtype=torch.float32)
        return data

    rows = edge_attr_row_count(data)
    if rows == ne:
        data.edge_attr = data.edge_attr.reshape(-1)[:ne].view(-1, 1).float()
        return data

    ew = data.edge_attr.detach().cpu().reshape(-1).float()
    if ew.numel() == ne:
        data.edge_attr = ew.view(-1, 1)
        return data
    if ew.numel() * 2 == ne:
        data.edge_attr = torch.cat([ew, ew]).view(-1, 1)
        return data
    if ne == 2 * ew.numel() and ew.numel() > 0:
        data.edge_index = data.edge_index[:, : ew.numel()]
        data.edge_attr = ew.view(-1, 1)
        return data

    m = min(ne, ew.numel())
    data.edge_index = data.edge_index[:, :m]
    data.edge_attr = ew[:m].view(-1, 1)
    return data


def reload_edges_from_npz(data: Data, npz_path: str | Path) -> Data:
    """Rebuild edge_index/edge_attr from subgraph npz (same as Step 5)."""
    z = np.load(str(npz_path))
    ei, ew = align_npz_edges(z["edge_index"], z["edge_weight"])
    data.edge_index = torch.from_numpy(ei).long()
    data.edge_attr = torch.from_numpy(np.asarray(ew, dtype=np.float32)).view(-1, 1)
    return force_sync_edge_attr(data)


def sanitize_graph_edges(data: Data) -> Data:
    """In-place fix for a single PyG Data object (e.g. loaded from pyg_dataset.pt)."""
    if data.x is not None:
        data.num_nodes = int(data.x.size(0))

    if data.edge_index is None or data.edge_index.numel() == 0:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        return data

    ei = data.edge_index.cpu().numpy()
    if data.edge_attr is not None:
        ew = data.edge_attr.cpu().numpy().reshape(-1)
    else:
        ew = np.ones(ei.shape[1], dtype=np.float32)

    # 1) edge_index columns must match edge_attr length
    ei, ew = align_npz_edges(ei, ew)

    # 2) drop edges that reference node ids >= num_nodes
    n = int(data.num_nodes)
    if n > 0 and ei.shape[1] > 0:
        valid = (ei[0] >= 0) & (ei[0] < n) & (ei[1] >= 0) & (ei[1] < n)
        ei = ei[:, valid]
        ew = ew[valid]

    # 3) coalesce duplicate directed edges
    ei2, ew2 = align_npz_edges(ei, ew)
    data.edge_index = torch.from_numpy(ei2)
    ew_t = torch.from_numpy(np.asarray(ew2, dtype=np.float32)).view(-1, 1)
    if ew_t.size(0) != data.edge_index.size(1):
        ew_t = torch.ones((data.edge_index.size(1), 1), dtype=torch.float32)
    data.edge_attr = ew_t
    return force_sync_edge_attr(data)


def ensure_graph_level_y(data: Data, y_dim: int = 30) -> None:
    """PyG batches 1D y as concat [B*D]; use shape [1, D] -> batched [B, D]."""
    if data.y is None:
        return
    y = data.y.view(-1)
    if y.numel() != y_dim:
        y = y[:y_dim]
    data.y = y.view(1, -1)
    if hasattr(data, "y_betweenness") and data.y_betweenness is not None:
        data.y_betweenness = data.y_betweenness.view(-1)[:10].view(1, -1)
    if hasattr(data, "y_capacity") and data.y_capacity is not None:
        data.y_capacity = data.y_capacity.view(-1)[:10].view(1, -1)
    if hasattr(data, "y_random") and data.y_random is not None:
        data.y_random = data.y_random.view(-1)[:10].view(1, -1)


def sanitize_graph_list(graphs: list[Data], *, show_progress: bool = True) -> int:
    """Sanitize all graphs; return count of graphs that needed fixes."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    fixed = 0
    it = graphs
    if show_progress and tqdm is not None and len(graphs) > 100:
        it = tqdm(graphs, desc="Sanitize graphs", unit="graph")

    for g in it:
        needs = False
        if g.x is not None:
            n_x = int(g.x.size(0))
            stored = int(g.num_nodes) if getattr(g, "num_nodes", None) is not None else n_x
            if stored != n_x:
                needs = True
        if g.edge_index is not None:
            n_e = g.edge_index.size(1)
            n_a = edge_attr_row_count(g)
            if g.edge_attr is None or n_a != n_e:
                needs = True
        if needs:
            fixed += 1
        sanitize_graph_edges(g)
        ensure_graph_level_y(g)
    return fixed
