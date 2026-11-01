from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate_batch1() -> None:
    slopes = pd.read_csv(
        ROOT
        / "artifacts"
        / "cf_batch1_2023_pooled"
        / "analysis_dose_response"
        / "slopes_by_unit.csv"
    )
    primary = slopes[slopes["family"].isin(("CF-D", "CF-S"))]
    assert len(primary) == 90
    assert primary["unit"].nunique() == 45
    assert set(primary["n_doses"]) == {5}

    audit = pd.read_csv(
        ROOT
        / "artifacts"
        / "cf_batch1_2023_pooled"
        / "analysis_denominator_audit"
        / "pooled_dose_L_vs_W.csv"
    )
    assert not audit.isna().any().any()
    assert set(audit["family"]) == {"CF-D", "CF-S"}
    assert set(audit["metric"]) == {"L", "W"}


def validate_batch2() -> None:
    base = ROOT / "artifacts" / "batch2_yearly" / "analysis_batch2"
    panel = pd.read_csv(base / "panel_unit_year_means.csv")
    assert len(panel) == 810
    assert not panel.isna().any().any()
    assert not panel.duplicated(["family", "unit", "tag"]).any()
    assert set(panel["family"]) == {"A", "B", "C"}
    for family in ("A", "B", "C"):
        g = panel[panel["family"] == family]
        assert len(g) == 270
        assert g["unit"].nunique() == 45

    a = panel[
        (panel["family"] == "A")
        & (panel["year_network"] == 2023)
        & (panel["year_outage"].astype(str) == "2023")
    ]
    b = panel[
        (panel["family"] == "B")
        & (panel["year_network"] == 2023)
        & (panel["year_outage"].astype(str) == "2023")
    ]
    matched = a.merge(b, on=["unit", "year_network", "year_outage"], suffixes=("_A", "_B"))
    assert len(matched) == 45
    for metric in ("L_cum", "W", "P_hit"):
        assert np.allclose(matched[f"{metric}_A"], matched[f"{metric}_B"], rtol=0, atol=0)

    spearman = pd.read_csv(base / "spearman_by_family_yearpair.csv")
    odi = pd.read_csv(base / "odi_ndi_by_yearpair.csv")
    assert len(spearman) == 36 and not spearman.isna().any().any()
    assert len(odi) == 12 and not odi.isna().any().any()


if __name__ == "__main__":
    validate_batch1()
    validate_batch2()
    print("PASS: bundled Batch 1 and Batch 2 review artifacts are internally consistent")
