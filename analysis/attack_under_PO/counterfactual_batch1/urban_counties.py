#!/usr/bin/env python3
"""Urban county flags for scenario U (urban R_c scaling)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def urban_fips_from_mcc(
    county_geom_mcc: pd.DataFrame,
    *,
    top_frac: float = 0.25,
) -> set[str]:
    """
    Mark urban counties as top `top_frac` by MCC / area_km2 (customer density proxy).

    county_geom_mcc must contain fips_str, MCC, area_km2 (from compute_impact_radius).
    """
    df = county_geom_mcc.copy()
    if "fips_str" not in df.columns:
        if "fips" in df.columns:
            df["fips_str"] = df["fips"].astype(int).astype(str).str.zfill(5)
        else:
            raise ValueError("county_geom_mcc needs fips_str or fips")
    for col in ("MCC", "area_km2"):
        if col not in df.columns:
            raise ValueError(f"county_geom_mcc missing {col}")
    area = df["area_km2"].astype(float).clip(lower=1e-6)
    dens = df["MCC"].astype(float) / area
    thr = float(np.nanquantile(dens.values, 1.0 - top_frac))
    urban = df.loc[dens >= thr, "fips_str"].astype(str).str.zfill(5)
    return set(urban.tolist())


def save_urban_fips(fips: set[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"fips_str": sorted(fips)}).to_csv(path, index=False)


def load_urban_fips(path: Path) -> set[str]:
    df = pd.read_csv(path)
    return set(df["fips_str"].astype(str).str.zfill(5).tolist())
