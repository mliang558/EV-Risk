"""Regression metrics for attack-curve heads (variable # of removal points)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from gnn import constants as _c
from gnn.constants import ATTACK_ORDER, ATTACK_INDEX, N_CURVE_POINTS as _NCP

N_CURVE_POINTS = int(getattr(_c, "N_CURVE_POINTS", 10))
PCT_REMOVAL = list(getattr(_c, "PCT_REMOVAL", [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]))


def _n_points_from_shape(y: np.ndarray, n_attacks: int | None = None) -> int:
    """Points per attack head from column count (10 / 20 / 30, not total cols)."""
    y = np.asarray(y)
    if y.ndim == 1:
        n = y.size
        if n == len(ATTACK_ORDER) * N_CURVE_POINTS:
            return N_CURVE_POINTS
        if n_attacks and n % n_attacks == 0:
            return n // n_attacks
        return N_CURVE_POINTS if n % 30 == 0 and n >= 30 else (n // 3 if n % 3 == 0 else n)
    d = int(y.shape[1])
    if d == len(ATTACK_ORDER) * N_CURVE_POINTS:
        return N_CURVE_POINTS
    if n_attacks and d % n_attacks == 0:
        return d // n_attacks
    if d in (10, 20):
        return N_CURVE_POINTS
    if d % 3 == 0 and d <= len(ATTACK_ORDER) * N_CURVE_POINTS:
        return d // 3
    return d


def split_attack_heads(
    y: np.ndarray,
    attacks: tuple[str, ...] = ATTACK_ORDER,
    n_points: int | None = None,
) -> dict[str, np.ndarray]:
    """Split (n, len(attacks)*k) or full (n, 30) into per-attack arrays."""
    from gnn.attack_targets import select_attacks_numpy

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    k = n_points
    if k is None:
        if y.shape[1] % len(attacks) == 0:
            k = y.shape[1] // len(attacks)
        else:
            k = _n_points_from_shape(y, n_attacks=len(attacks))
    if y.shape[1] == len(ATTACK_ORDER) * N_CURVE_POINTS:
        y = select_attacks_numpy(y, attacks, k)
    elif y.shape[1] != len(attacks) * k:
        raise ValueError(f"split_attack_heads: shape {y.shape}, attacks={attacks}, k={k}")
    out = {}
    for i, a in enumerate(attacks):
        out[a] = y[:, i * k : (i + 1) * k]
    return out


def split_three_heads(y: np.ndarray, n_points: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy: always returns (B, C, R) slices; missing attacks are empty (0 cols)."""
    parts = split_attack_heads(y, ATTACK_ORDER, n_points)
    k = n_points or _n_points_from_shape(y)
    z = np.zeros((y.shape[0] if y.ndim > 1 else 1, k), dtype=float)
    if y.ndim == 1:
        z = z.reshape(1, -1)
    return parts.get("betweenness", z), parts.get("capacity", z), parts.get("random", z)


def metrics_by_attack(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attacks: tuple[str, ...] = ATTACK_ORDER,
    n_points: int | None = None,
) -> dict[str, dict[str, float]]:
    k = n_points
    if k is None:
        if y_true.shape[1] % len(attacks) == 0:
            k = y_true.shape[1] // len(attacks)
        else:
            k = _n_points_from_shape(y_true, n_attacks=len(attacks))
    t_parts = split_attack_heads(y_true, attacks, k)
    p_parts = split_attack_heads(y_pred, attacks, k)
    return {a: curve_metrics(t_parts[a], p_parts[a], n_points=k) for a in attacks}


def slice_labels_to_k(
    y: np.ndarray,
    k: int,
    attacks: tuple[str, ...] = ATTACK_ORDER,
) -> np.ndarray:
    """Take first k removal points per attack from full 30-dim labels."""
    from gnn.attack_targets import select_attacks_numpy

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.shape[1] == len(attacks) * k:
        return y
    if y.shape[1] == len(ATTACK_ORDER) * N_CURVE_POINTS:
        return select_attacks_numpy(y, attacks, k)
    if y.shape[1] != len(ATTACK_ORDER) * N_CURVE_POINTS:
        return y[:, : len(attacks) * k]
    full = y.reshape(len(y), len(ATTACK_ORDER), N_CURVE_POINTS)
    return full[:, :, :k].reshape(len(y), len(ATTACK_ORDER) * k)


def curve_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_points: int | None = None,
) -> dict[str, float]:
    """Metrics for one head (n, k) or three heads (n, 3*k)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.ndim == 1:
        yt = yt.reshape(1, -1)
    if yp.ndim == 1:
        yp = yp.reshape(1, -1)

    dim = yt.shape[1]
    if n_points is not None:
        k = int(n_points)
    elif dim == len(ATTACK_ORDER) * N_CURVE_POINTS:
        k = N_CURVE_POINTS
    elif dim in (20, 2 * N_CURVE_POINTS):
        k = dim // 2
    elif dim % 3 == 0 and dim <= len(ATTACK_ORDER) * N_CURVE_POINTS:
        k = dim // 3
    else:
        k = dim  # single head: (n, k)

    if dim == len(ATTACK_ORDER) * N_CURVE_POINTS and yp.shape[1] in (20, 2 * k, len(ATTACK_ORDER) * k):
        from gnn.attack_targets import select_attacks_numpy

        n_att = yp.shape[1] // k if yp.shape[1] % k == 0 else 2
        att = ATTACK_ORDER if n_att == 3 else ("betweenness", "random")
        yt = select_attacks_numpy(yt, att, k)
        yp = select_attacks_numpy(yp, att, k) if yp.shape[1] == len(ATTACK_ORDER) * N_CURVE_POINTS else yp
    elif dim == len(ATTACK_ORDER) * N_CURVE_POINTS and k < N_CURVE_POINTS:
        yt = slice_labels_to_k(yt, k)
    if yp.shape[1] != yt.shape[1]:
        yp = slice_labels_to_k(yp, k) if yp.shape[1] == len(ATTACK_ORDER) * N_CURVE_POINTS else yp[:, : yt.shape[1]]

    n_cols = yt.shape[1]
    n_att = n_cols // k if k > 0 and n_cols % k == 0 else 1
    if n_cols == n_att * k and n_att > 1:
        yt = yt.reshape(-1, n_cols)
        yp = yp.reshape(-1, n_cols)
    elif n_cols == k:
        yt = yt.reshape(-1, k)
        yp = yp.reshape(-1, k)
    else:
        raise ValueError(f"curve_metrics: expected {k} or {3 * k} cols, got {n_cols}")

    per_r2 = [float(r2_score(yt[:, i], yp[:, i])) for i in range(yt.shape[1])]
    return {
        "n_points": k,
        "r2": float(r2_score(yt.ravel(), yp.ravel())),
        "mae": float(mean_absolute_error(yt.ravel(), yp.ravel())),
        "r2_mean_per_point": float(np.mean(per_r2)),
    }


def per_point_metrics_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_points: int = N_CURVE_POINTS,
) -> pd.DataFrame:
    """One row per (attack, removal %): R² and MAE at that curve point."""
    rows = []
    parts_t = split_attack_heads(y_true, ATTACK_ORDER, n_points)
    parts_p = split_attack_heads(y_pred, ATTACK_ORDER, n_points)
    for attack in ATTACK_ORDER:
        if attack not in parts_t:
            continue
        yt, yp = parts_t[attack], parts_p[attack]
        for i in range(n_points):
            rows.append(
                {
                    "attack": attack,
                    "point_index": i,
                    "removal_pct": PCT_REMOVAL[i],
                    "r2": float(r2_score(yt[:, i], yp[:, i])),
                    "mae": float(mean_absolute_error(yt[:, i], yp[:, i])),
                }
            )
    return pd.DataFrame(rows)


def per_point_table_vs_full_labels(
    y_true_30: np.ndarray,
    y_pred: np.ndarray,
    k_model: int,
) -> pd.DataFrame:
    """R² at all 10 label points; NaN where model does not predict (k_model < 10)."""
    rows = []
    attacks = ("betweenness", "capacity", "random")
    for a_idx, attack in enumerate(attacks):
        for i in range(N_CURVE_POINTS):
            yt = y_true_30[:, a_idx * N_CURVE_POINTS + i]
            row = {
                "attack": attack,
                "point_index": i,
                "removal_pct": PCT_REMOVAL[i],
                "model_predicts": i < k_model,
            }
            if i < k_model:
                yp = y_pred[:, a_idx * k_model + i]
                row["r2"] = float(r2_score(yt, yp))
                row["mae"] = float(mean_absolute_error(yt, yp))
            else:
                row["r2"] = np.nan
                row["mae"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def metrics_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    min_n: int = 5,
    n_points: int | None = None,
) -> pd.DataFrame:
    yt = slice_labels_to_k(y_true, n_points or _n_points_from_shape(y_true))
    yp = np.asarray(y_pred, dtype=float)
    if yp.shape[-1] != yt.shape[-1]:
        yp = slice_labels_to_k(yp, n_points or _n_points_from_shape(yt))
    groups = np.asarray(groups)
    rows = []
    for g in sorted(pd.unique(groups), key=str):
        mask = groups == g
        n = int(mask.sum())
        if n < min_n:
            continue
        m = curve_metrics(yt[mask], yp[mask])
        rows.append({"group": str(g), "n": n, **m})
    return pd.DataFrame(rows)
