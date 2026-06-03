#!/usr/bin/env python3
"""
Overlay EV charging stations on county-level outage risk map.

Reads:
- advi_results_2018_2023/county_estimates_advi.csv
- tl_2021_us_county.shp (Census county boundaries)
- EV station CSV with latitude/longitude (2026 data)

Outputs:
- advi_results_2018_2023/map_lambda_with_ev_stations.png
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError as e:
    raise ImportError(
        "geopandas is required for plotting maps. Install with:\n"
        '  pip install "geopandas[plot]" mapclassify'
    ) from e


def main() -> None:
    # Resolve project root (Pro_directory)
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[4]  # .../Pro_directory

    # Paths
    results_dir = project_root / "advi_results_2018_2023"
    county_csv = results_dir / "county_estimates_advi.csv"

    # County shapefile
    ev_project_root = project_root.parent
    shapefile = ev_project_root / "tl_2021_us_county" / "tl_2021_us_county.shp"

    # EV station CSV：使用你当前的 2026 数据
    # 需要至少包含经纬度列（下面支持 Latitude/Longitude 或 latitude/longitude）
    ev_csv = project_root / "data" / "raw" / "alt_fuel_stations_historical_day (Jan 1 2026).csv"

    if not county_csv.exists():
        raise FileNotFoundError(f"County ADVI results not found: {county_csv}")
    if not shapefile.exists():
        raise FileNotFoundError(
            f"County shapefile not found: {shapefile}\n"
            "请确认 tl_2021_us_county 放在 EV_Project 根目录下，或修改脚本中的 shapefile 路径。"
        )
    if not ev_csv.exists():
        raise FileNotFoundError(
            f"EV station CSV not found: {ev_csv}\n"
            "请将 2026 年 EV 站点数据 CSV 路径改成你实际文件的位置，"
            "并确保包含 'latitude' 和 'longitude' 列。"
        )

    print(f"Loading county ADVI results from: {county_csv}")
    county = pd.read_csv(county_csv)
    county["fips_str"] = county["fips"].astype(int).astype(str).str.zfill(5)

    print(f"Loading county boundaries from: {shapefile}")
    gdf = gpd.read_file(shapefile)

    # 只保留本土 48 州 + DC
    if "STATEFP" in gdf.columns:
        non_continental = {"02", "15", "72", "78", "60", "66", "69"}
        before = len(gdf)
        gdf = gdf[~gdf["STATEFP"].astype(str).isin(non_continental)].copy()
        after = len(gdf)
        print(f"Filtered to CONUS + DC counties: {after}/{before} rows retained.")

    # FIPS
    if "GEOID" in gdf.columns:
        gdf["fips_str"] = gdf["GEOID"].astype(str).str.zfill(5)
    elif {"STATEFP", "COUNTYFP"}.issubset(gdf.columns):
        gdf["fips_str"] = (
            gdf["STATEFP"].astype(str).str.zfill(2)
            + gdf["COUNTYFP"].astype(str).str.zfill(3)
        )
    else:
        raise ValueError(
            "无法在 shapefile 中找到 FIPS 字段（需要 GEOID 或 STATEFP+COUNTYFP）。"
        )

    # Merge λ
    merged = gdf.merge(county, on="fips_str", how="left")
    merged_valid = merged[merged["lambda_mean"].notna()].copy()

    if merged_valid.empty:
        raise ValueError("Merge 后没有任何县带有 lambda_mean，请检查 FIPS 对齐。")

    # 底图：蓝色系 sequential 色带，适合高风险区域突出
    cmap_lambda = "Blues"  # 也可以改成 'YlGnBu' 试效果

    # 加载 EV station 点
    print(f"Loading EV stations from: {ev_csv}")
    ev = pd.read_csv(ev_csv)
    cols_lower = {c.lower(): c for c in ev.columns}
    if "latitude" in cols_lower and "longitude" in cols_lower:
        lat_col = cols_lower["latitude"]
        lon_col = cols_lower["longitude"]
    else:
        raise ValueError(
            f"EV station CSV 需要包含经纬度列（例如 'Latitude'/'Longitude' 或 'latitude'/'longitude'），当前列为: {list(ev.columns)}"
        )

    ev_gdf = gpd.GeoDataFrame(
        ev,
        geometry=gpd.points_from_xy(ev[lon_col], ev[lat_col]),
        crs="EPSG:4326",
    )

    # 将 EV 点投影到与县界相同坐标系
    if merged_valid.crs is not None:
        ev_gdf = ev_gdf.to_crs(merged_valid.crs)

    # 可选：只保留落在本土多边形里的站点
    try:
        ev_gdf = gpd.sjoin(ev_gdf, merged_valid[["fips_str", "geometry"]], how="inner", predicate="within")
        print(f"EV stations within CONUS counties: {len(ev_gdf)}")
    except Exception:
        # 如果 gpd.sjoin 不可用，就跳过 spatial filter
        print("Warning: spatial join failed; showing all EV stations without CONUS filtering.")

    # 绘图
    print("Plotting λ map with EV stations overlay...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    merged_valid.plot(
        column="lambda_mean",
        cmap=cmap_lambda,
        scheme="quantiles",
        k=7,
        linewidth=0.05,
        edgecolor="none",
        legend=True,
        legend_kwds={"loc": "lower right"},
        ax=ax,
        alpha=0.9,
    )

    # EV 站点：半透明小点
    ev_gdf.plot(
        ax=ax,
        markersize=3,
        color="red",        # 或改成 '#2c2c2c' 以更低调
        edgecolor="none",
        linewidth=0,
        alpha=0.25,         # 更透明，减少对底图的遮挡
    )

    ax.set_axis_off()
    ax.set_title(
        "County-level outage rate λ and 2026 EV charging stations",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    out_path = results_dir / "map_lambda_with_ev_stations.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

