"""Train/test splits: east–west geography and leave-one-state-out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gnn.constants import DEFAULT_LON_THRESHOLD


def window_center_lon(meta: pd.DataFrame) -> np.ndarray:
    if "center_lon" in meta.columns:
        return meta["center_lon"].astype(float).values
    if "center_lat" in meta.columns and "lon" in meta.columns:
        return meta["lon"].astype(float).values
    raise ValueError("windows_meta needs center_lon for geographic split")


def east_west_mask(meta: pd.DataFrame, lon_threshold: float = DEFAULT_LON_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """East (lon > threshold) -> train; West -> test."""
    lon = window_center_lon(meta)
    train_mask = lon > lon_threshold
    test_mask = ~train_mask
    return train_mask, test_mask


def primary_state(manifest_row: pd.Series) -> str:
    raw = str(manifest_row.get("states", ""))
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    return parts[0] if parts else "UNK"


def normalize_state_code(name: str) -> str:
    """Map state name or abbreviation to 2-letter code when possible."""
    s = str(name).strip()
    if not s or s == "UNK":
        return "UNK"
    if len(s) == 2:
        return s.upper()
    key = s.lower().replace(".", "")
    _FULL_TO_ABBR = {
        "alabama": "AL",
        "alaska": "AK",
        "arizona": "AZ",
        "arkansas": "AR",
        "california": "CA",
        "colorado": "CO",
        "connecticut": "CT",
        "delaware": "DE",
        "florida": "FL",
        "georgia": "GA",
        "iowa": "IA",
        "maine": "ME",
        "maryland": "MD",
        "massachusetts": "MA",
        "michigan": "MI",
        "minnesota": "MN",
        "mississippi": "MS",
        "missouri": "MO",
        "montana": "MT",
        "nebraska": "NE",
        "nevada": "NV",
        "new hampshire": "NH",
        "new jersey": "NJ",
        "new mexico": "NM",
        "new york": "NY",
        "north carolina": "NC",
        "north dakota": "ND",
        "ohio": "OH",
        "oklahoma": "OK",
        "oregon": "OR",
        "pennsylvania": "PA",
        "rhode island": "RI",
        "south carolina": "SC",
        "south dakota": "SD",
        "tennessee": "TN",
        "texas": "TX",
        "utah": "UT",
        "vermont": "VT",
        "virginia": "VA",
        "washington": "WA",
        "west virginia": "WV",
        "wisconsin": "WI",
        "wyoming": "WY",
        "district of columbia": "DC",
    }
    return _FULL_TO_ABBR.get(key, s.upper()[:2] if len(s) >= 2 else s.upper())


def state_codes_series(manifest: pd.DataFrame) -> pd.Series:
    return manifest.apply(lambda r: normalize_state_code(primary_state(r)), axis=1)


# Census-style regions (2-letter state codes)
REGION_STATES: dict[str, frozenset[str]] = {
    "northeast": frozenset({"CT", "DE", "MA", "MD", "ME", "NH", "NJ", "NY", "PA", "RI", "VT", "DC"}),
    "midwest": frozenset(
        {"IA", "IL", "IN", "KS", "MI", "MN", "MO", "ND", "NE", "OH", "SD", "WI"}
    ),
    "south": frozenset(
        {
            "AL",
            "AR",
            "FL",
            "GA",
            "KY",
            "LA",
            "MS",
            "NC",
            "OK",
            "SC",
            "TN",
            "TX",
            "VA",
            "WV",
        }
    ),
    "west": frozenset({"AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY"}),
}

# Paper appendix: 5 representative states
REPRESENTATIVE_LOO_STATES = ("DE", "CA", "TX", "ME", "IA")


def holdout_region_masks(
    manifest: pd.DataFrame, holdout_region: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """Train on all windows NOT in holdout_region; test on windows in that region."""
    region = str(holdout_region).strip().lower()
    if region not in REGION_STATES:
        raise ValueError(f"Unknown region {holdout_region!r}; choose from {list(REGION_STATES)}")
    codes = state_codes_series(manifest)
    in_region = codes.isin(REGION_STATES[region]).values
    test_mask = in_region
    train_mask = ~test_mask
    label = f"holdout_{region}"
    return train_mask, test_mask, label


def east_west_retrain_masks(
    meta: pd.DataFrame, lon_threshold: float = DEFAULT_LON_THRESHOLD
) -> tuple[np.ndarray, np.ndarray, str]:
    """Same as east_west but explicit name for geographic generalization script."""
    train_mask, test_mask = east_west_mask(meta, lon_threshold)
    return train_mask, test_mask, f"east_west_lon>{lon_threshold}"


STRATUM_ORDER = ("urban", "suburban", "rural")


def window_stratum_series(meta: pd.DataFrame) -> pd.Series:
    if "window_stratum" in meta.columns:
        return meta["window_stratum"].astype(str).str.strip().str.lower()
    if "radius_km" in meta.columns:
        r = meta["radius_km"].astype(float)
        out = pd.Series("unknown", index=meta.index, dtype=object)
        out[np.isclose(r, 10.0, atol=0.5)] = "urban"
        out[np.isclose(r, 30.0, atol=0.5)] = "suburban"
        out[np.isclose(r, 80.0, atol=0.5)] = "rural"
        return out
    raise ValueError("meta needs window_stratum or radius_km for stratified split")


def stratum_distribution(meta: pd.DataFrame, mask: np.ndarray | None = None) -> dict[str, float]:
    s = window_stratum_series(meta)
    if mask is not None:
        s = s.iloc[np.asarray(mask, dtype=bool)]
    vc = s.value_counts(normalize=True)
    return {k: float(vc.get(k, 0.0)) for k in STRATUM_ORDER}


def stratified_stratum_mask(
    meta: pd.DataFrame,
    test_frac: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
  Per-window stratified split: within each urban/suburban/rural stratum,
  hold out test_frac for test. Train and test match global stratum mix.
    """
    rng = np.random.default_rng(seed)
    strata = window_stratum_series(meta).values
    test_mask = np.zeros(len(meta), dtype=bool)
    for label in STRATUM_ORDER:
        idx = np.where(strata == label)[0]
        if len(idx) == 0:
            continue
        idx = idx.copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(test_frac * len(idx))))
        test_mask[idx[:n_test]] = True
    train_mask = ~test_mask
    return train_mask, test_mask


def stratified_state_holdout_mask(
    manifest: pd.DataFrame,
    meta: pd.DataFrame,
    test_frac: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
  Assign whole states to test until ~test_frac windows, greedily matching
  urban/suburban/rural proportions to the full dataset.
    """
    if "window_id" not in meta.columns or "window_id" not in manifest.columns:
        raise ValueError("stratified_state_holdout requires window_id in meta and manifest")
    man = manifest.copy()
    man["window_id"] = man["window_id"].astype(str)
    man["_state"] = state_codes_series(man).values
    m = meta.copy()
    m["window_id"] = m["window_id"].astype(str)
    m = m.merge(man[["window_id", "_state"]], on="window_id", how="left")
    m["_pos"] = np.arange(len(m))

    global_props = stratum_distribution(m)
    target_n = int(round(test_frac * len(m)))

    states = [s for s in sorted(m["_state"].dropna().unique()) if s and s != "UNK"]
    rng = np.random.default_rng(seed)
    rng.shuffle(states)

    state_to_pos: dict[str, list[int]] = {}
    for st in states:
        pos = m.loc[m["_state"] == st, "_pos"].astype(int).tolist()
        if pos:
            state_to_pos[st] = pos

    test_pos: set[int] = set()
    while len(test_pos) < target_n and state_to_pos:
        best_add: list[int] = []
        best_score = -np.inf
        for _st, pos_list in state_to_pos.items():
            if all(p in test_pos for p in pos_list):
                continue
            add = [p for p in pos_list if p not in test_pos]
            if not add:
                continue
            cand = list(test_pos | set(add))
            if len(cand) > target_n * 1.2:
                continue
            mask = np.zeros(len(m), dtype=bool)
            mask[cand] = True
            props = stratum_distribution(m, mask=mask)
            score = -sum(abs(props.get(k, 0) - global_props.get(k, 0)) for k in STRATUM_ORDER)
            if score > best_score:
                best_score = score
                best_add = add
        if not best_add:
            break
        test_pos.update(best_add)

    test_mask = np.zeros(len(meta), dtype=bool)
    for p in test_pos:
        if 0 <= p < len(test_mask):
            test_mask[p] = True
    train_mask = ~test_mask
    return train_mask, test_mask


def build_split_masks(
    meta: pd.DataFrame,
    *,
    manifest: pd.DataFrame | None = None,
    lon_threshold: float = DEFAULT_LON_THRESHOLD,
    test_frac: float = 0.3,
    seed: int = 42,
    include: tuple[str, ...] = ("east_west", "stratified_stratum", "stratified_state"),
) -> dict[str, np.ndarray]:
    """Build named train/test masks for pyg_dataset.pt split_masks."""
    out: dict[str, np.ndarray] = {}
    if "east_west" in include:
        tr, te = east_west_mask(meta, lon_threshold)
        out["east_west_train"] = tr
        out["east_west_test"] = te
        out["lon_threshold"] = np.array([lon_threshold])
    if "stratified_stratum" in include:
        tr, te = stratified_stratum_mask(meta, test_frac=test_frac, seed=seed)
        out["stratified_stratum_train"] = tr
        out["stratified_stratum_test"] = te
    if "stratified_state" in include:
        if manifest is None:
            raise ValueError("stratified_state split requires manifest.csv")
        tr, te = stratified_state_holdout_mask(manifest, meta, test_frac=test_frac, seed=seed)
        out["stratified_state_train"] = tr
        out["stratified_state_test"] = te
    out["stratified_test_frac"] = np.array([test_frac])
    out["stratified_seed"] = np.array([seed])
    return out


def split_key_prefix(split: str) -> str:
    s = str(split).strip().lower().replace("-", "_")
    if s in ("east_west", "ew", "geo"):
        return "east_west"
    if s in ("stratified_stratum", "stratum", "strata"):
        return "stratified_stratum"
    if s in ("stratified_state", "state_stratified"):
        return "stratified_state"
    raise ValueError(f"Unknown split {split!r}")


def resolve_train_test_masks(
    meta: pd.DataFrame,
    split_masks: dict[str, np.ndarray] | None,
    split: str,
    *,
    manifest: pd.DataFrame | None = None,
    test_frac: float = 0.3,
    seed: int = 42,
    lon_threshold: float = DEFAULT_LON_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Load masks from bundle or compute on the fly."""
    prefix = split_key_prefix(split)
    n = len(meta)
    if split_masks and f"{prefix}_train" in split_masks and f"{prefix}_test" in split_masks:
        tr, te = split_masks[f"{prefix}_train"], split_masks[f"{prefix}_test"]
        if len(tr) == n and len(te) == n:
            return tr, te
    if prefix == "east_west":
        return east_west_mask(meta, lon_threshold)
    if prefix == "stratified_stratum":
        return stratified_stratum_mask(meta, test_frac=test_frac, seed=seed)
    if prefix == "stratified_state":
        if manifest is None:
            raise ValueError("stratified_state split needs manifest.csv")
        return stratified_state_holdout_mask(manifest, meta, test_frac=test_frac, seed=seed)
    raise ValueError(f"Unhandled split {split}")


def format_stratum_report(meta: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray) -> str:
    g = stratum_distribution(meta)
    tr = stratum_distribution(meta, train_mask)
    te = stratum_distribution(meta, test_mask)
    lines = [
        f"  n_train={int(train_mask.sum())} n_test={int(test_mask.sum())}",
        "  stratum%  global | train | test",
    ]
    for k in STRATUM_ORDER:
        lines.append(f"    {k:9s} {g.get(k,0):.3f} | {tr.get(k,0):.3f} | {te.get(k,0):.3f}")
    return "\n".join(lines)


def leave_one_state_folds(
    manifest: pd.DataFrame,
    min_windows: int = 20,
    states_only: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    codes = state_codes_series(manifest)
    want = None
    if states_only:
        want = {normalize_state_code(s) for s in states_only}
    folds = {}
    for st in sorted(codes.unique()):
        if st == "UNK":
            continue
        if want is not None and st not in want:
            continue
        test_mask = (codes == st).values
        if test_mask.sum() < min_windows:
            continue
        folds[st] = test_mask
    return folds
