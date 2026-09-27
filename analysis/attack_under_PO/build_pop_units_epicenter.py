#!/usr/bin/env python3
"""
Build census-tract population units for network-independent epicenters.

Writes: data/processed/pop_units_epicenter.gpkg
Columns: tract_geoid, county_fips, population, geometry (EPSG:4326)

Sources
-------
- Geometry: TIGER/Line 2020 tracts (HTTPS per state)
- Population: Census 2020 Decennial PL P1_001N (optional CENSUS_API_KEY),
  else TIGER land area is NOT used as pop — require API or a local
  census_tract_strata.csv with a filled population column.

Usage:
  export CENSUS_API_KEY=...   # recommended
  python analysis/attack_under_PO/build_pop_units_epicenter.py
  python analysis/attack_under_PO/build_pop_units_epicenter.py --states TX,CA
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "data" / "processed" / "pop_units_epicenter.gpkg"

STATE_ABBR_TO_FIPS = {
    "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10",
    "DC": "11", "FL": "12", "GA": "13", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26",
    "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
    "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48",
    "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56",
}


def _read_shp_from_zip_url(url: str, timeout: int = 300):
    import geopandas as gpd

    req = urllib.request.Request(url, headers={"User-Agent": "ev-pop-units/1.0"})
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "tiger.zip"
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            zpath.write_bytes(resp.read())
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(td)
        shps = list(Path(td).glob("*.shp"))
        if not shps:
            raise FileNotFoundError(f"No .shp in {url}")
        return gpd.read_file(shps[0])


def load_state_tract_geom(abbr: str):
    import geopandas as gpd

    fips = STATE_ABBR_TO_FIPS[abbr]
    # Prefer local parquet cache from GNN pipeline if present
    local = ROOT / "data" / "processed" / "tract_shp_2020" / f"{abbr}_tract2020.parquet"
    if local.is_file():
        gdf = gpd.read_parquet(local)
        if "tract_geoid" not in gdf.columns and "GEOID" in gdf.columns:
            gdf["tract_geoid"] = gdf["GEOID"].astype(str).str.zfill(11)
        return gdf

    url = f"https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_{fips}_tract.zip"
    print(f"  download {abbr}: {url}", flush=True)
    tr = _read_shp_from_zip_url(url)
    tr = tr[tr["STATEFP"].astype(str).str.zfill(2) == fips].copy()
    tr["tract_geoid"] = tr["GEOID"].astype(str).str.zfill(11)
    return tr.to_crs("EPSG:4326")


def fetch_state_pop(api_key: str, state_fips: str, sleep_s: float = 0.15):
    import pandas as pd

    url = (
        f"https://api.census.gov/data/2020/dec/pl"
        f"?get=P1_001N"
        f"&for=tract:*"
        f"&in=state:{state_fips}"
        f"&key={api_key}"
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as e:
            if attempt == 2:
                raise
            time.sleep(1.5)
    header, *body = data
    rows = []
    for rec in body:
        total = float(rec[0])
        st, county, tract = rec[1], rec[2], rec[3]
        rows.append(
            {
                "tract_geoid": f"{st}{county}{tract}".zfill(11),
                "population": total,
            }
        )
    time.sleep(sleep_s)
    return pd.DataFrame(rows)


def load_pop_from_strata_csv() -> "pd.DataFrame | None":
    import pandas as pd

    path = ROOT / "data" / "processed" / "census_tract_strata.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path, dtype={"tract_geoid": str})
    if "population" not in df.columns:
        return None
    df["tract_geoid"] = df["tract_geoid"].astype(str).str.zfill(11)
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    if df["population"].fillna(0).le(0).all():
        return None
    return df[["tract_geoid", "population"]].dropna()


def main() -> None:
    import geopandas as gpd
    import pandas as pd

    parser = argparse.ArgumentParser(description="Build pop_units_epicenter.gpkg")
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    parser.add_argument(
        "--states",
        type=str,
        default="",
        help="Comma abbrs (default: all CONUS EV states)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("CENSUS_API_KEY", ""),
    )
    args = parser.parse_args()

    abbrs = (
        [a.strip().upper() for a in args.states.split(",") if a.strip()]
        if args.states
        else sorted(STATE_ABBR_TO_FIPS.keys())
    )
    # Drop AK/HI if somehow present
    abbrs = [a for a in abbrs if a in STATE_ABBR_TO_FIPS]

    strata_pop = load_pop_from_strata_csv()
    api_key = args.api_key.strip()
    if strata_pop is None and not api_key:
        raise SystemExit(
            "Need CENSUS_API_KEY or a census_tract_strata.csv with filled population.\n"
            "  export CENSUS_API_KEY=...\n"
            "  python analysis/attack_under_PO/build_pop_units_epicenter.py"
        )

    parts = []
    for i, abbr in enumerate(abbrs, 1):
        print(f"[{i}/{len(abbrs)}] {abbr}", flush=True)
        geom = load_state_tract_geom(abbr)
        if "tract_geoid" not in geom.columns:
            raise SystemExit(f"{abbr}: no tract_geoid")
        geom["tract_geoid"] = geom["tract_geoid"].astype(str).str.zfill(11)
        geom["county_fips"] = geom["tract_geoid"].str[:5]

        if strata_pop is not None:
            pop = strata_pop
        else:
            pop = fetch_state_pop(api_key, STATE_ABBR_TO_FIPS[abbr])

        merged = geom.merge(pop, on="tract_geoid", how="left")
        merged["population"] = merged["population"].fillna(0.0).clip(lower=0.0)
        # keep only tracts with geometry
        merged = merged[merged.geometry.notna()].copy()
        parts.append(merged[["tract_geoid", "county_fips", "population", "geometry"]])

    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gpkg":
        out.to_file(out_path, driver="GPKG")
    else:
        out.to_parquet(out_path)
    n_pos = int((out["population"] > 0).sum())
    print(
        f"Wrote {len(out)} tracts ({n_pos} with pop>0) → {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
