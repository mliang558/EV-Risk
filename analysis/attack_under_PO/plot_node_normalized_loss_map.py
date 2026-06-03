#!/usr/bin/env python3
"""
Plot state-level node_normalized_loss from MC results on a US map.

Reads all *_attack_mc_node_normalized_posterior_lambda.csv in results/,
joins with state boundaries (from county shapefile dissolved by STATEFP),
and saves a choropleth to results/state_node_normalized_loss_map.png.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Project root = Pro_directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

# STATEFP (2-digit Census FIPS) -> full state name (matches CSV "state" column)
STATEFP_TO_STATE_NAME = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut",
    "10": "Delaware", "11": "District of Columbia", "12": "Florida",
    "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky",
    "22": "Louisiana", "23": "Maine", "24": "Maryland", "25": "Massachusetts",
    "26": "Michigan", "27": "Minnesota", "28": "Mississippi", "29": "Missouri",
    "30": "Montana", "31": "Nebraska", "32": "Nevada", "33": "New Hampshire",
    "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma",
    "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee",
    "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia",
    "53": "Washington", "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}


def main() -> None:
    # 1. Collect all state node_normalized results
    records = []
    for csv_path in RESULTS_DIR.glob("*_attack_mc_node_normalized_posterior_lambda.csv"):
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            records.append(df.iloc[0])

    if not records:
        raise FileNotFoundError(
            f"No *_attack_mc_node_normalized_posterior_lambda.csv found in {RESULTS_DIR}"
        )

    metrics_df = pd.DataFrame(records)

    # 2. Load state boundaries: use county shapefile and dissolve by STATEFP
    ev_root = PROJECT_ROOT.parent
    county_shp = ev_root / "tl_2021_us_county" / "tl_2021_us_county.shp"
    if not county_shp.exists():
        county_shp = PROJECT_ROOT / "notebooks" / "tl_2021_us_county" / "tl_2021_us_county.shp"
    if not county_shp.exists():
        raise FileNotFoundError(
            f"County shapefile not found. Tried {ev_root / 'tl_2021_us_county'} and "
            f"{PROJECT_ROOT / 'notebooks' / 'tl_2021_us_county'}"
        )

    counties = gpd.read_file(county_shp)
    non_continental = {"02", "15", "72", "78", "60", "66", "69"}
    if "STATEFP" in counties.columns:
        counties = counties[~counties["STATEFP"].astype(str).isin(non_continental)].copy()
    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)

    states_gdf = counties.dissolve(by="STATEFP").reset_index()
    states_gdf["state"] = states_gdf["STATEFP"].map(STATEFP_TO_STATE_NAME)
    states_gdf = states_gdf[["STATEFP", "state", "geometry"]].copy()

    # 3. Join with metrics
    plot_gdf = states_gdf.merge(metrics_df, on="state", how="left")

    # 4. Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_gdf.plot(
        column="node_normalized_loss",
        cmap="YlOrRd",
        linewidth=0.3,
        edgecolor="gray",
        legend=True,
        ax=ax,
        missing_kwds={
            "color": "lightgray",
            "edgecolor": "white",
            "hatch": "///",
            "label": "No data",
        },
    )
    ax.set_title("State-level node_normalized_loss (MC posterior outage attacks)", fontsize=12)
    ax.set_axis_off()
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("node_normalized_loss")

    out_path = RESULTS_DIR / "state_node_normalized_loss_map.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved map to: {out_path}")


if __name__ == "__main__":
    main()
