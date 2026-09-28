#!/usr/bin/env python3
"""
Build census-tract population units for network-independent epicenters.

Writes: data/processed/pop_units_epicenter.gpkg
Columns: tract_geoid, county_fips, population, geometry (EPSG:4326)

Geography lock (must match EAGLE-I / MC county FIPS)
----------------------------------------------------
- Tract boundaries: TIGER/Line 2020 (stable through the 2020s).
- Population (in order of preference):
  1. ``--pop-csv`` or filled ``data/processed/census_tract_strata.csv``
  2. Census 2020 PL P1_001N via ``CENSUS_API_KEY`` (api.census.gov)
  3. **Census Reporter ACS B01003** (no key; 2019–2023 style latest release)
- county_fips = first 5 digits of 2020 tract GEOID (legacy CT 09001–09015).
- County polygons for radius stay on tl_2021 (see GEOGRAPHY_FIPS_LOCK.md).

Usage:
  # No Census key needed (ACS via Census Reporter):
  python analysis/attack_under_PO/build_pop_units_epicenter.py --states TX
  python analysis/attack_under_PO/build_pop_units_epicenter.py

  # Prefer 2020 Decennial if you have a key:
  export CENSUS_API_KEY=...
  python analysis/attack_under_PO/build_pop_units_epicenter.py --prefer-decennial
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


def _http_get(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ev-pop-units/1.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _read_shp_from_zip_url(url: str, timeout: int = 300):
    import geopandas as gpd

    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "tiger.zip"
        zpath.write_bytes(_http_get(url, timeout=timeout))
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(td)
        shps = list(Path(td).glob("*.shp"))
        if not shps:
            raise FileNotFoundError(f"No .shp in {url}")
        return gpd.read_file(shps[0])


def load_state_tract_geom(abbr: str):
    import geopandas as gpd

    fips = STATE_ABBR_TO_FIPS[abbr]
    local = ROOT / "data" / "processed" / "tract_shp_2020" / f"{abbr}_tract2020.parquet"
    if local.is_file():
        gdf = gpd.read_parquet(local)
        if "tract_geoid" not in gdf.columns and "GEOID" in gdf.columns:
            gdf["tract_geoid"] = gdf["GEOID"].astype(str).str.zfill(11)
        return gdf

    url = f"https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_{fips}_tract.zip"
    print(f"  download geom {abbr}: {url}", flush=True)
    tr = _read_shp_from_zip_url(url)
    tr = tr[tr["STATEFP"].astype(str).str.zfill(2) == fips].copy()
    tr["tract_geoid"] = tr["GEOID"].astype(str).str.zfill(11)
    return tr.to_crs("EPSG:4326")


def fetch_state_pop_decennial(api_key: str, state_fips: str, sleep_s: float = 0.2):
    """2020 PL P1_001N via api.census.gov (requires valid key)."""
    import pandas as pd

    if not api_key:
        raise ValueError("Decennial fetch requires CENSUS_API_KEY")
    url = (
        f"https://api.census.gov/data/2020/dec/pl"
        f"?get=P1_001N&for=tract:*&in=state:{state_fips}&key={api_key}"
    )
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = _http_get(url, timeout=120)
            text = raw.decode("utf-8", errors="replace")
            if text.lstrip().startswith("<") or "Missing Key" in text:
                raise RuntimeError(
                    "Census API returned HTML (Missing Key / blocked). "
                    "Check CENSUS_API_KEY, or omit --prefer-decennial to use "
                    "Census Reporter ACS (no key)."
                )
            data = json.loads(text)
            break
        except Exception as e:
            last_err = e
            if attempt == 2:
                raise RuntimeError(f"Decennial API failed for state {state_fips}: {e}") from e
            time.sleep(1.5)
    else:
        raise RuntimeError(str(last_err))

    header, *body = data
    rows = []
    for rec in body:
        total = float(rec[0]) if rec[0] not in (None, "", "null") else 0.0
        st, county, tract = rec[1], rec[2], rec[3]
        rows.append(
            {"tract_geoid": f"{st}{county}{tract}".zfill(11), "population": total}
        )
    time.sleep(sleep_s)
    return pd.DataFrame(rows)


def fetch_state_pop_acs_reporter(state_fips: str, sleep_s: float = 0.3):
    """
    ACS total population (B01003) via Census Reporter — no API key.

    GEOIDs look like 14000US48001950100 → tract_geoid = last 11 chars.
    """
    import pandas as pd

    url = (
        "https://api.censusreporter.org/1.0/data/show/latest"
        f"?table_ids=B01003&geo_ids=140|04000US{state_fips}"
    )
    print(f"  population ACS (Census Reporter) state={state_fips}", flush=True)
    raw = _http_get(url, timeout=180)
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("<"):
        raise RuntimeError(
            f"Census Reporter returned HTML for state {state_fips} "
            f"(blocked or down). Head: {text[:160]!r}"
        )
    payload = json.loads(text)
    data = payload.get("data") or {}
    rows = []
    for geo_id, tables in data.items():
        # 14000US48001950100 → 48001950100
        if "US" in geo_id:
            geoid = geo_id.split("US", 1)[-1].zfill(11)
        else:
            geoid = str(geo_id)[-11:].zfill(11)
        est = (
            tables.get("B01003", {})
            .get("estimate", {})
            .get("B01003001")
        )
        if est is None:
            continue
        rows.append({"tract_geoid": geoid, "population": float(est)})
    time.sleep(sleep_s)
    if not rows:
        raise RuntimeError(f"Census Reporter returned 0 tracts for state {state_fips}")
    return pd.DataFrame(rows)


def load_pop_csv(path: Path):
    import pandas as pd

    df = pd.read_csv(path, dtype={"tract_geoid": str})
    if "tract_geoid" not in df.columns or "population" not in df.columns:
        raise SystemExit(f"{path} needs columns tract_geoid, population")
    df["tract_geoid"] = df["tract_geoid"].astype(str).str.zfill(11)
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["population"])
    if df.empty or df["population"].fillna(0).le(0).all():
        raise SystemExit(f"{path}: no positive population values")
    return df[["tract_geoid", "population"]]


def load_pop_from_strata_csv() -> "pd.DataFrame | None":
    path = ROOT / "data" / "processed" / "census_tract_strata.csv"
    if not path.is_file():
        return None
    try:
        return load_pop_csv(path)
    except SystemExit:
        return None


def main() -> None:
    import geopandas as gpd
    import pandas as pd

    parser = argparse.ArgumentParser(description="Build pop_units_epicenter.gpkg")
    parser.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    parser.add_argument("--states", type=str, default="", help="Comma abbrs (default: all)")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("CENSUS_API_KEY", ""),
        help="Census API key for 2020 Decennial (optional)",
    )
    parser.add_argument(
        "--prefer-decennial",
        action="store_true",
        help="Use 2020 PL via Census API when key is set (default: ACS Reporter, no key)",
    )
    parser.add_argument(
        "--pop-csv",
        type=str,
        default="",
        help="Optional CSV with tract_geoid,population (skips API)",
    )
    args = parser.parse_args()

    abbrs = (
        [a.strip().upper() for a in args.states.split(",") if a.strip()]
        if args.states
        else sorted(STATE_ABBR_TO_FIPS.keys())
    )
    abbrs = [a for a in abbrs if a in STATE_ABBR_TO_FIPS]

    if args.pop_csv:
        strata_pop = load_pop_csv(Path(args.pop_csv))
        print(f"Using --pop-csv ({len(strata_pop)} rows)", flush=True)
    else:
        strata_pop = load_pop_from_strata_csv()
        if strata_pop is not None:
            print(f"Using census_tract_strata.csv ({len(strata_pop)} rows)", flush=True)

    api_key = (args.api_key or "").strip()
    use_decennial = bool(args.prefer_decennial and api_key)
    if args.prefer_decennial and not api_key:
        print(
            "[warn] --prefer-decennial set but no CENSUS_API_KEY; using ACS Reporter",
            flush=True,
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
        elif use_decennial:
            pop = fetch_state_pop_decennial(api_key, STATE_ABBR_TO_FIPS[abbr])
        else:
            pop = fetch_state_pop_acs_reporter(STATE_ABBR_TO_FIPS[abbr])

        merged = geom.merge(pop, on="tract_geoid", how="left")
        merged["population"] = merged["population"].fillna(0.0).clip(lower=0.0)
        merged = merged[merged.geometry.notna()].copy()
        n_pos = int((merged["population"] > 0).sum())
        print(f"  tracts={len(merged)} pop>0={n_pos}", flush=True)
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
    print(f"Wrote {len(out)} tracts ({n_pos} with pop>0) → {out_path}", flush=True)


if __name__ == "__main__":
    main()
