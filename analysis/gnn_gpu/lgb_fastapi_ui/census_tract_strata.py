#!/usr/bin/env python3
"""
Census tract urban / suburban / rural labels for charging nodes.

Priority
--------
1. Cached tract table: data/processed/census_tract_strata.csv
2. Census 2020 PL API (P2 urban pop / P1 total pop) if CENSUS_API_KEY is set
3. Fallback: tract x CBSA (M1 metro / M2 micropolitan) + tract land area

Node assignment: spatial join lat/lon -> 2020 census tract (pygris, per state).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from states import STATES_ORDER

from project_paths import resolve_project_root

STATE_ABBR_TO_FIPS = {
    "AL": "01", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10",
    "DC": "11", "FL": "12", "GA": "13", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26",
    "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
    "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48",
    "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56",
}

TRACT_CACHE_NAME = "census_tract_strata.csv"
TRACT_SHP_CACHE_DIR = "tract_shp_2020"
NODES_STRATA_CACHE = "nodes_tract_strata_2026.parquet"
FCC_TRACT_CACHE = "fcc_tract_geoid_cache.parquet"


def _project_root() -> Path:
    return resolve_project_root()


def tract_cache_path(project_root: Path | None = None) -> Path:
    root = project_root or _project_root()
    return root / "data" / "processed" / TRACT_CACHE_NAME


def ruca_primary_to_stratum(ruca: float) -> str:
    code = int(round(float(ruca)))
    if code <= 3:
        return "urban"
    if code <= 6:
        return "suburban"
    return "rural"


def urban_pct_to_stratum(urban_pct: float) -> str:
    if not np.isfinite(urban_pct):
        return "suburban"
    if urban_pct >= 0.75:
        return "urban"
    if urban_pct <= 0.10:
        return "rural"
    return "suburban"


def fetch_tract_urban_pct_census_api(
    api_key: str,
    state_fps: list[str] | None = None,
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    """Fetch 2020 Decennial PL urban population share for all tracts."""
    state_fps = state_fps or sorted(set(STATE_ABBR_TO_FIPS.values()))
    rows: list[dict] = []
    for st in state_fps:
        url = (
            f"https://api.census.gov/data/2020/dec/pl"
            f"?get=P1_001N,P2_001N"
            f"&for=tract:*"
            f"&in=state:{st}"
            f"&key={api_key}"
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as resp:
                    data = json.loads(resp.read().decode())
                break
            except urllib.error.HTTPError as e:
                if attempt == 2:
                    raise RuntimeError(f"Census API failed for state {st}: {e}") from e
                time.sleep(1.5)
        header, *body = data
        for rec in body:
            total = float(rec[0])
            urban = float(rec[1])
            state, county, tract = rec[2], rec[3], rec[4]
            geoid = f"{state}{county}{tract}"
            urban_pct = urban / total if total > 0 else 0.0
            rows.append(
                {
                    "tract_geoid": geoid,
                    "population": total,
                    "urban_pop": urban,
                    "urban_pct": urban_pct,
                    "stratum": urban_pct_to_stratum(urban_pct),
                    "source": "census_api_2020_pl",
                }
            )
        time.sleep(sleep_s)
    return pd.DataFrame(rows)


def tract_shp_cache_dir(project_root: Path | None = None) -> Path:
    root = project_root or _project_root()
    d = root / "data" / "processed" / TRACT_SHP_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _census_reachable(timeout: float = 15.0) -> bool:
    """Quick check whether this host can reach Census (API or TIGER)."""
    urls = (
        "https://api.census.gov/data/2020/dec/pl?get=NAME&for=state:01",
        "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/",
    )
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ev-tract-setup/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status < 400:
                    return True
        except Exception:
            continue
    return False


def _read_shapefile_from_zip_url(url: str, timeout: int = 300) -> gpd.GeoDataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "ev-tract-setup/1.0"})
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "tiger.zip"
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            zpath.write_bytes(resp.read())
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(td)
        shps = list(Path(td).glob("*.shp"))
        if not shps:
            raise FileNotFoundError(f"No .shp in zip from {url}")
        return gpd.read_file(shps[0])


def _load_state_tracts_tiger_https(abbr: str) -> gpd.GeoDataFrame:
    fips = STATE_ABBR_TO_FIPS[abbr]
    url = f"https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_{fips}_tract.zip"
    print(f"  TIGER HTTPS: {abbr} ({url})", flush=True)
    tr = _read_shapefile_from_zip_url(url)
    st = fips
    tr = tr[tr["STATEFP"].astype(str).str.zfill(2) == st].copy()
    tr["tract_geoid"] = tr["GEOID"].astype(str).str.zfill(11)
    tr["land_km2"] = pd.to_numeric(tr["ALAND"], errors="coerce") / 1e6
    return tr[["tract_geoid", "land_km2", "geometry"]].to_crs("EPSG:4326")


def _load_cbsa_2020() -> gpd.GeoDataFrame:
    cache_name = "cbsa_2020_national.parquet"
    cache = tract_shp_cache_dir() / cache_name
    if cache.exists():
        return gpd.read_parquet(cache)

    url = "https://www2.census.gov/geo/tiger/TIGER2020/CBSA/tl_2020_us_cbsa.zip"
    errors: list[str] = []
    try:
        print("  TIGER HTTPS: national CBSA...", flush=True)
        cbsa = _read_shapefile_from_zip_url(url)
        cbsa.to_parquet(cache)
        return cbsa
    except Exception as e:
        errors.append(f"tiger_https: {e}")

    import pygris

    for protocol in ("https", "http", "ftp"):
        try:
            cbsa = pygris.core_based_statistical_areas(year=2020, cb=True, protocol=protocol, timeout=180)
            cbsa.to_parquet(cache)
            return cbsa
        except Exception as e:
            errors.append(f"pygris({protocol}): {e}")
    raise RuntimeError("Failed to load CBSA boundaries: " + "; ".join(errors))


def load_state_tracts(abbr: str, project_root: Path | None = None) -> gpd.GeoDataFrame:
    """Load cached state tract polygons or download (HTTPS TIGER, then pygris)."""
    project_root = project_root or _project_root()
    cache = tract_shp_cache_dir(project_root) / f"{abbr}_tract2020.parquet"
    if cache.exists():
        return gpd.read_parquet(cache)

    st = STATE_ABBR_TO_FIPS[abbr]
    errors: list[str] = []
    tr: gpd.GeoDataFrame | None = None

    try:
        tr = _load_state_tracts_tiger_https(abbr)
    except Exception as e:
        errors.append(str(e))

    if tr is None:
        import pygris

        for protocol in ("https", "http", "ftp"):
            for attempt in range(2):
                try:
                    tr = pygris.tracts(state=abbr, year=2020, cb=True, timeout=180, protocol=protocol)
                    tr = tr[tr["STATEFP"] == st].copy()
                    tr["tract_geoid"] = tr["GEOID"].astype(str).str.zfill(11)
                    tr["land_km2"] = pd.to_numeric(tr["ALAND"], errors="coerce") / 1e6
                    tr = tr[["tract_geoid", "land_km2", "geometry"]].to_crs("EPSG:4326")
                    break
                except Exception as e:
                    errors.append(f"pygris/{protocol}: {e}")
                    time.sleep(2.0 * (attempt + 1))
            if tr is not None:
                break

    if tr is None:
        raise RuntimeError(
            f"Failed to download tracts for {abbr}. "
            "GPU nodes often block Census FTP. Options:\n"
            "  1) export CENSUS_API_KEY=... && python build_census_tract_strata.py  (CSV only, no polygons)\n"
            "  2) Run download_tract_shapes.py on your laptop, then scp data/processed/tract_shp_2020/ to server\n"
            f"  Details: {' | '.join(errors[-3:])}"
        ) from None

    tr.to_parquet(cache)
    return tr


def build_tract_strata_cbsa_fallback(project_root: Path | None = None) -> pd.DataFrame:
    """Classify tracts using CBSA (metro/micro) and tract land area."""
    project_root = project_root or _project_root()
    cbsa = _load_cbsa_2020()
    metro = cbsa[cbsa["LSAD"] == "M1"].copy().to_crs("EPSG:4326")
    micro = cbsa[cbsa["LSAD"] == "M2"].copy().to_crs("EPSG:4326")

    tract_rows: list[gpd.GeoDataFrame] = []
    for abbr in STATES_ORDER:
        tract_rows.append(load_state_tracts(abbr, project_root))

    tracts = pd.concat(tract_rows, ignore_index=True)
    tracts = gpd.GeoDataFrame(tracts, geometry="geometry", crs="EPSG:4326")

    tracts["stratum"] = "rural"
    tracts["source"] = "cbsa_fallback_2020"

    metro_hit = gpd.sjoin(
        tracts[["tract_geoid", "land_km2", "geometry"]],
        metro[["geometry", "NAME"]],
        how="inner",
        predicate="intersects",
    )
    metro_hit = metro_hit.drop_duplicates("tract_geoid")
    micro_hit = gpd.sjoin(
        tracts.loc[~tracts["tract_geoid"].isin(metro_hit["tract_geoid"]), ["tract_geoid", "land_km2", "geometry"]],
        micro[["geometry", "NAME"]],
        how="inner",
        predicate="intersects",
    )
    micro_hit = micro_hit.drop_duplicates("tract_geoid")

    metro_land = metro_hit["land_km2"].astype(float)
    small_cut = float(metro_land.quantile(0.35)) if len(metro_land) else 10.0

    strata = tracts.set_index("tract_geoid")
    for geoid in metro_hit["tract_geoid"]:
        lk = float(strata.loc[geoid, "land_km2"])
        strata.loc[geoid, "stratum"] = "urban" if lk <= small_cut else "suburban"
    for geoid in micro_hit["tract_geoid"]:
        strata.loc[geoid, "stratum"] = "suburban"

    out = strata.reset_index()[["tract_geoid", "land_km2", "stratum", "source"]]
    out["urban_pct"] = np.nan
    out["population"] = np.nan
    return out


STRATUM_RADIUS_KM = {"urban": 10.0, "suburban": 30.0, "rural": 80.0}


def stratum_at_lat_lon(
    lat: float,
    lon: float,
    project_root: Path | None = None,
) -> dict[str, object]:
    """
    Same source as node labels in Step-3: FCC lat/lon → 2020 tract GEOID → census_tract_strata.csv.
    """
    project_root = project_root or _project_root()
    geoid = lookup_tract_geoid_fcc(float(lat), float(lon))
    if not geoid:
        return {}
    try:
        tract_strata = load_or_build_tract_strata(project_root)
    except Exception:
        return {}
    tract_strata = tract_strata.copy()
    tract_strata["tract_geoid"] = tract_strata["tract_geoid"].astype(str).str.zfill(11)
    row = tract_strata.loc[tract_strata["tract_geoid"] == geoid]
    if row.empty:
        return {}
    r = row.iloc[0]
    st = str(r["stratum"]).strip().lower()
    if st not in STRATUM_RADIUS_KM:
        st = "suburban"
    return {
        "lat": float(lat),
        "lon": float(lon),
        "window_stratum": st,
        "radius_km": float(STRATUM_RADIUS_KM[st]),
        "tract_geoid": geoid,
        "urban_pct": float(r["urban_pct"]) if pd.notna(r.get("urban_pct")) else None,
        "method": "census_tract_fcc",
        "source": str(r.get("source", "census_tract_strata.csv")),
    }


def load_or_build_tract_strata(
    project_root: Path | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    project_root = project_root or _project_root()
    cache = tract_cache_path(project_root)

    if cache.exists() and not force_rebuild:
        return pd.read_csv(cache, dtype={"tract_geoid": str})

    api_key = os.environ.get("CENSUS_API_KEY", "").strip()
    allow_cbsa = os.environ.get("ALLOW_CBSA_FALLBACK", "").strip().lower() in ("1", "true", "yes")
    if api_key:
        print("Building tract strata from Census 2020 PL API...", flush=True)
        df = fetch_tract_urban_pct_census_api(api_key)
        df["land_km2"] = np.nan
    elif allow_cbsa:
        print(
            "CENSUS_API_KEY not set; ALLOW_CBSA_FALLBACK=1 → CBSA+tract download (needs Census network).",
            flush=True,
        )
        df = build_tract_strata_cbsa_fallback(project_root)
    else:
        reachable = _census_reachable()
        raise RuntimeError(
            "Cannot build census_tract_strata.csv without CENSUS_API_KEY.\n"
            "  Free key: https://api.census.gov/data/key_signup.html\n"
            "  export CENSUS_API_KEY='your_key'\n"
            "  python build_census_tract_strata.py\n"
            "This uses the Census API only (no TIGER/pygris download).\n"
            + (
                "This host cannot reach Census (typical on GPU nodes). "
                "Build the CSV on your laptop with the API key, then upload:\n"
                f"  {cache}\n"
                "For map polygons, run download_tract_shapes.py locally and upload:\n"
                f"  {tract_shp_cache_dir(project_root)}/\n"
                if not reachable
                else "Census appears reachable; to force CBSA+polygon fallback instead:\n"
                "  export ALLOW_CBSA_FALLBACK=1\n"
            )
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"Saved tract strata: {cache} ({len(df):,} tracts)")
    return df


def nodes_strata_cache_path(project_root: Path | None = None, geocode_method: str = "fcc") -> Path:
    root = project_root or _project_root()
    suffix = "" if geocode_method == "fcc" else f"_{geocode_method}"
    return root / "data" / "processed" / NODES_STRATA_CACHE.replace(".parquet", f"{suffix}.parquet")


def fcc_cache_path(project_root: Path | None = None) -> Path:
    return (project_root or _project_root()) / "data" / "processed" / FCC_TRACT_CACHE


def lookup_tract_geoid_fcc(lat: float, lon: float, retries: int = 4) -> str | None:
    """FCC Census 2020 geocoder: lat/lon -> 11-digit tract GEOID."""
    url = (
        "https://geo.fcc.gov/api/census/block/find"
        f"?latitude={lat:.6f}&longitude={lon:.6f}"
        "&format=json"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=45) as resp:
                data = json.loads(resp.read().decode())
            block = data.get("Block") or {}
            fips = block.get("FIPS")
            if not fips or len(str(fips)) < 11:
                return None
            return str(fips)[:11]
        except Exception:
            time.sleep(0.5 * (2**attempt))
    return None


def geocode_unique_coords_fcc(
    coords: pd.DataFrame,
    cache: pd.DataFrame | None,
    coord_round: int = 4,
    max_workers: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """coords columns: lat, lon (unique rows). Returns lat, lon, tract_geoid."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache = cache if cache is not None else pd.DataFrame(columns=["lat", "lon", "tract_geoid"])
    if not cache.empty:
        cache = cache.copy()
        cache["lat"] = cache["lat"].round(coord_round)
        cache["lon"] = cache["lon"].round(coord_round)

    work = coords.copy()
    work["lat"] = work["lat"].round(coord_round)
    work["lon"] = work["lon"].round(coord_round)
    merged = work.merge(cache, on=["lat", "lon"], how="left")
    todo = merged[merged["tract_geoid"].isna()][["lat", "lon"]].drop_duplicates()

    new_rows: list[dict] = []
    if len(todo) > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(lookup_tract_geoid_fcc, float(r.lat), float(r.lon)): (r.lat, r.lon)
                for r in todo.itertuples(index=False)
            }
            done = 0
            for fut in as_completed(futures):
                lat, lon = futures[fut]
                geoid = fut.result()
                new_rows.append({"lat": lat, "lon": lon, "tract_geoid": geoid})
                done += 1
                if done % 500 == 0:
                    print(f"  FCC geocoded {done}/{len(todo)} unique coords...")
                    cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(
                        ["lat", "lon"], keep="last"
                    )
                    new_rows.clear()

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(
            ["lat", "lon"], keep="last"
        )

    out = work.merge(cache, on=["lat", "lon"], how="left")
    return out, cache


def assign_nodes_county_from_tract_strata(
    nodes: pd.DataFrame,
    tract_strata: pd.DataFrame,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """
    Fast fallback: map nodes to county (shapefile), stratum = modal tract stratum in county.
    Tract GEOID is not assigned per node; use FCC geocoder for tract-level IDs.
    """
    project_root = project_root or _project_root()
    shp = project_root.parent / "tl_2021_us_county" / "tl_2021_us_county.shp"
    if not shp.exists():
        raise FileNotFoundError(f"County shapefile not found: {shp}")

    ts = tract_strata.copy()
    ts["county_fips"] = ts["tract_geoid"].astype(str).str.zfill(11).str[:5]
    county_stratum = (
        ts.groupby("county_fips")["stratum"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "suburban")
        .reset_index()
    )

    allowed = set(STATE_ABBR_TO_FIPS.values())
    gdf = gpd.read_file(shp)
    gdf = gdf[gdf["STATEFP"].astype(str).isin(allowed)].copy()
    gdf["county_fips"] = gdf["GEOID"].astype(str).str.zfill(5)
    gdf = gdf.merge(county_stratum, on="county_fips", how="left")

    pts = gpd.GeoDataFrame(
        nodes.copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(nodes["lon"], nodes["lat"])],
        crs="EPSG:4326",
    )
    gdf = gdf.to_crs("EPSG:4326")
    joined = gpd.sjoin(
        pts,
        gdf[["county_fips", "stratum", "geometry"]],
        how="left",
        predicate="within",
    )
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    out["tract_geoid"] = ""
    out["urban_pct"] = np.nan
    out["source"] = "county_modal_tract_stratum"
    out["stratum"] = out["stratum"].fillna("suburban")
    return out


def assign_nodes_to_tract_strata(
    nodes: pd.DataFrame,
    tract_strata: pd.DataFrame,
    project_root: Path | None = None,
    use_nodes_cache: bool = True,
    geocode_method: str = "fcc",
) -> pd.DataFrame:
    """Map nodes to census tracts and attach urban/suburban/rural labels."""
    project_root = project_root or _project_root()
    nodes_cache = nodes_strata_cache_path(project_root, geocode_method)
    if use_nodes_cache and nodes_cache.exists():
        cached = pd.read_parquet(nodes_cache)
        if len(cached) == len(nodes):
            return cached

    if geocode_method == "county":
        out = assign_nodes_county_from_tract_strata(nodes, tract_strata, project_root)
        out.to_parquet(nodes_cache, index=False)
        return out

    strata = tract_strata.set_index("tract_geoid")
    tract_attrs = strata.reset_index()[["tract_geoid", "stratum", "urban_pct", "source"]]

    unique_coords = nodes[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    fcc_cache = pd.read_parquet(fcc_cache_path(project_root)) if fcc_cache_path(project_root).exists() else None
    print(f"Geocoding {len(unique_coords):,} unique coordinates via FCC Census API...")
    geocoded, fcc_cache = geocode_unique_coords_fcc(unique_coords, fcc_cache)
    fcc_cache_path(project_root).parent.mkdir(parents=True, exist_ok=True)
    fcc_cache.to_parquet(fcc_cache_path(project_root), index=False)

    out = nodes.merge(geocoded, on=["lat", "lon"], how="left")
    out = out.merge(tract_attrs, on="tract_geoid", how="left")
    out["stratum"] = out["stratum"].fillna("suburban")
    out["source"] = out["source"].fillna("unmapped_tract")

    nodes_cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(nodes_cache, index=False)
    print(f"Saved node strata: {nodes_cache}")
    return out
