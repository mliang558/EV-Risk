"""Shared constants for attack-curve GNN."""

from __future__ import annotations

PCT_LABELS = ["p5", "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]
PCT_REMOVAL = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
N_CURVE_POINTS = len(PCT_LABELS)

# If per-point R² drops after ~30% removal, train with --curve-points 4 (5–30% only).
EARLY_CURVE_POINTS = 4

Y_BETWEENNESS_COLS = [f"y_betweenness_{p}" for p in PCT_LABELS]
Y_CAPACITY_COLS = [f"y_capacity_{p}" for p in PCT_LABELS]
Y_RANDOM_COLS = [f"y_random_{p}" for p in PCT_LABELS]
Y_ALL_COLS = Y_BETWEENNESS_COLS + Y_CAPACITY_COLS + Y_RANDOM_COLS

ATTACK_ORDER = ("betweenness", "capacity", "random")
ATTACK_INDEX = {a: i for i, a in enumerate(ATTACK_ORDER)}
ATTACK_Y_COLS = {
    "betweenness": Y_BETWEENNESS_COLS,
    "capacity": Y_CAPACITY_COLS,
    "random": Y_RANDOM_COLS,
}


def parse_attacks(spec: str | None) -> tuple[str, ...]:
    """e.g. 'all' | 'no-capacity' | 'betweenness,random'."""
    if spec is None or not str(spec).strip():
        return ATTACK_ORDER
    s = str(spec).strip().lower().replace(" ", "")
    if s in ("all", "3", "bcr", "betweenness,capacity,random"):
        return ATTACK_ORDER
    if s in ("no-capacity", "no_capacity", "nocapacity", "br", "betweenness,random"):
        return ("betweenness", "random")
    parts = tuple(p.strip() for p in s.split(",") if p.strip())
    for p in parts:
        if p not in ATTACK_INDEX:
            raise ValueError(f"Unknown attack {p!r}; choose from {ATTACK_ORDER}")
    if not parts:
        return ATTACK_ORDER
    return parts


def y_cols_for_attacks(attacks: tuple[str, ...] | None = None) -> list[str]:
    attacks = attacks or ATTACK_ORDER
    cols: list[str] = []
    for a in attacks:
        cols.extend(ATTACK_Y_COLS[a])
    return cols

NODE_FEATURE_NAMES = ["lon", "lat", "capacity", "dcfc_ratio"]

# Default: Mississippi–Rockies style split (window center longitude)
DEFAULT_LON_THRESHOLD = -100.0
