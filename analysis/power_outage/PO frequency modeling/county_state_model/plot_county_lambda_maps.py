#!/usr/bin/env python3
"""
Plot county-level outage rates and uncertainty from ADVI results.

Reads:
- advi_results_2018_2023/county_estimates_advi.csv
- tl_2021_us_county.shp (Census county boundaries)

Outputs:
- advi_results_2018_2023/map_lambda_mean.png
- advi_results_2018_2023/map_lambda_uncertainty.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

try:
    import geopandas as gpd
except ImportError as e:  # pragma: no cover
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

    # Shapefile: Census 2021 counties under EV_Project/tl_2021_us_county
    # Adjust here if你用的是别的 boundary 文件
    ev_project_root = project_root.parent
    shapefile = ev_project_root / "tl_2021_us_county" / "tl_2021_us_county.shp"

    if not county_csv.exists():
        raise FileNotFoundError(f"County ADVI results not found: {county_csv}")
    if not shapefile.exists():
        raise FileNotFoundError(
            f"County shapefile not found: {shapefile}\n"
            "请确认 tl_2021_us_county 放在 EV_Project 根目录下，或修改脚本中的 shapefile 路径。"
        )

    print(f"Loading county ADVI results from: {county_csv}")
    county = pd.read_csv(county_csv)
    county["fips_str"] = county["fips"].astype(int).astype(str).str.zfill(5)

    print(f"Loading county boundaries from: {shapefile}")
    gdf = gpd.read_file(shapefile)

    # 只保留本土 48 州 + DC（去掉 AK, HI, PR 等岛屿）
    if "STATEFP" in gdf.columns:
        non_continental = {"02", "15", "72", "78", "60", "66", "69"}  # AK, HI, PR, VI, 以及部分领地
        before = len(gdf)
        gdf = gdf[~gdf["STATEFP"].astype(str).isin(non_continental)].copy()
        after = len(gdf)
        print(f"Filtered to CONUS + DC counties: {after}/{before} rows retained.")

    # 尝试从 GEOID 或 STATEFP+COUNTYFP 构造 FIPS
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

    # 自定义更浅的 sequential 色带（参考图中的 Teal 渐变）
    teal_seq = ["#b5d1ae", "#80b2a9", "#53b687", "#4286b7", "#4b825e", "#127240"]
    cmap_lambda = mcolors.LinearSegmentedColormap.from_list("lambda_seq", teal_seq)

    # Divergent / 暖色调用于不确定性
    warm_seq = ["#e0ecf4", "#9ebcda", "#fdd0a2", "#fc8d59", "#e34a33"]
    cmap_uncert = mcolors.LinearSegmentedColormap.from_list("uncert_seq", warm_seq)

    # Merge
    merged = gdf.merge(county, on="fips_str", how="left")
    merged_valid = merged[merged["lambda_mean"].notna()].copy()
    merged_missing = merged[merged["lambda_mean"].isna()].copy()

    if merged_valid.empty:
        raise ValueError("Merge 后没有任何县带有 lambda_mean，请检查 FIPS 对齐。")

    # 不确定性：95% CI 相对宽度（更适合作为“相对不确定度”）
    merged_valid["lambda_ci_width"] = (
        merged_valid["lambda_ci_upper"] - merged_valid["lambda_ci_lower"]
    )
    merged_valid["lambda_rel_uncertainty"] = (
        merged_valid["lambda_ci_width"] / merged_valid["lambda_mean"].replace(0, np.nan)
    )

    # 画 lambda_mean 地图（Nature 常用的连续色带，用 viridis）
    print("Plotting county λ mean map...")
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
        alpha=0.85,
    )

    # 无数据县：浅灰色边框 + 斜杠填充
    if not merged_missing.empty:
        merged_missing.plot(
            ax=ax,
            facecolor="none",
            edgecolor="lightgrey",
            linewidth=0.2,
            hatch="///",
            alpha=0.6,
        )
    ax.set_axis_off()
    ax.set_title(
        "County-level outage rate λ (mean, events/year)", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path_mean = results_dir / "map_lambda_mean.png"
    plt.savefig(out_path_mean, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out_path_mean}")
    plt.close(fig)

    # 画不确定性地图（相对不确定度），用暖色系 magma_r
    print("Plotting county λ relative uncertainty map (95% CI width / mean)...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    merged_valid.plot(
        column="lambda_rel_uncertainty",
        cmap=cmap_uncert,
        scheme="quantiles",
        k=7,
        linewidth=0.05,
        edgecolor="none",
        legend=True,
        legend_kwds={"loc": "lower right"},
        ax=ax,
        alpha=0.85,
    )

    if not merged_missing.empty:
        merged_missing.plot(
            ax=ax,
            facecolor="none",
            edgecolor="lightgrey",
            linewidth=0.2,
            hatch="///",
            alpha=0.6,
        )
    ax.set_axis_off()
    ax.set_title(
        "Relative uncertainty of λ (95% CI width / mean)", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path_unc = results_dir / "map_lambda_uncertainty.png"
    plt.savefig(out_path_unc, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out_path_unc}")
    plt.close(fig)

    # 简单汇总信息
    print("\nMap stats (λ mean):")
    print(
        merged_valid["lambda_mean"].describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        )
    )
    print("\nMap stats (λ relative uncertainty = CI width / mean):")
    print(
        merged_valid["lambda_rel_uncertainty"].describe(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        )
    )


if __name__ == "__main__":
    main()

