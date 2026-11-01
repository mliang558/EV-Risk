from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "attack_under_PO"))

from counterfactual_batch1.analyze_dose_response import (  # noqa: E402
    DOSE_PCTS,
    dose_response_table,
    fit_slopes,
)


def test_explicit_zero_dose_is_not_duplicated() -> None:
    rows = [{"unit": "TX", "family": "baseline", "dose_pct": 0, "L_tilde_mean": 10.0}]
    for family in ("CF-D", "CF-S"):
        for dose in DOSE_PCTS:
            rows.append(
                {
                    "unit": "TX",
                    "family": family,
                    "dose_pct": dose,
                    "L_tilde_mean": 10.0 - dose / 10.0,
                }
            )

    dose = dose_response_table(pd.DataFrame(rows))
    slopes = fit_slopes(dose)

    assert not dose.duplicated(["unit", "family", "dose_pct"]).any()
    assert set(slopes["n_doses"]) == {len(DOSE_PCTS)}


def test_control_without_explicit_zero_inherits_one_baseline_row() -> None:
    rows = [
        {"unit": "TX", "family": "baseline", "dose_pct": 0, "L_tilde_mean": 10.0},
        {"unit": "TX", "family": "CF-D-pop", "dose_pct": 10, "L_tilde_mean": 9.0},
        {"unit": "TX", "family": "CF-D-pop", "dose_pct": 20, "L_tilde_mean": 8.0},
    ]

    dose = dose_response_table(pd.DataFrame(rows))
    control = dose[dose["family"] == "CF-D-pop"]

    assert control["dose_pct"].tolist() == [0.0, 10.0, 20.0]
    assert control.iloc[0]["L_tilde_mean"] == 10.0
    assert control.iloc[0]["delta_L_tilde_pct"] == 0.0


if __name__ == "__main__":
    test_explicit_zero_dose_is_not_duplicated()
    test_control_without_explicit_zero_inherits_one_baseline_row()
    print("PASS: dose-response uses unique dose values")
