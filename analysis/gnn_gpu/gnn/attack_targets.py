"""Select / reshape multi-attack curve labels (30-dim full -> subset)."""

from __future__ import annotations

import numpy as np
import torch

from gnn.constants import ATTACK_INDEX, ATTACK_ORDER, N_CURVE_POINTS


def match_target_to_pred(
    target: torch.Tensor,
    pred: torch.Tensor,
    attacks: tuple[str, ...] = ATTACK_ORDER,
) -> torch.Tensor:
    if target.dim() == 1:
        target = target.view(pred.size(0), -1)
    n_att = len(attacks)
    k = pred.size(1) // n_att
    out_dim = n_att * k

    if target.size(1) == out_dim:
        return target

    # Full 3×10 labels in storage order [B, 30]
    if target.size(1) == len(ATTACK_ORDER) * N_CURVE_POINTS:
        full = target.view(target.size(0), len(ATTACK_ORDER), N_CURVE_POINTS)
        idx = [ATTACK_INDEX[a] for a in attacks]
        return full[:, idx, :k].reshape(target.size(0), out_dim)

    if target.size(1) > out_dim:
        return target[:, :out_dim]
    return target


def select_attacks_numpy(y: np.ndarray, attacks: tuple[str, ...], k: int | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    k = k or (y.shape[1] // len(attacks) if y.shape[1] % len(attacks) == 0 else N_CURVE_POINTS)
    if y.shape[1] == len(attacks) * k:
        return y
    if y.shape[1] == len(ATTACK_ORDER) * N_CURVE_POINTS:
        full = y.reshape(len(y), len(ATTACK_ORDER), N_CURVE_POINTS)
        idx = [ATTACK_INDEX[a] for a in attacks]
        return full[:, idx, :k].reshape(len(y), len(attacks) * k)
    raise ValueError(f"Cannot select attacks {attacks} from y shape {y.shape}")
