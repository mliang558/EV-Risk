from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "analysis" / "attack_under_PO" / "canonical"
if str(CANONICAL) not in sys.path:
    sys.path.insert(0, str(CANONICAL))

from analyze_rank_uncertainty import (  # noqa: E402
    bootstrap_summary,
    descending_ranks,
    load_metric_panel,
)


def test_descending_ranks() -> None:
    values = np.array([2.0, 5.0, 1.0])
    assert descending_ranks(values).tolist() == [2, 1, 3]


def test_load_and_bootstrap(tmp_path: Path) -> None:
    for unit, values in {"A": [3.0, 4.0, 5.0], "B": [1.0, 2.0, 3.0]}.items():
        unit_dir = tmp_path / unit
        unit_dir.mkdir()
        pd.DataFrame(
            {"sim_id": [0, 1, 2], "L_tilde": values}
        ).to_csv(unit_dir / "metrics_baseline.csv", index=False)

    units, sim_ids, panel = load_metric_panel(
        tmp_path, "L_tilde", expected_units=2, expected_sims=3
    )
    assert units == ["A", "B"]
    assert sim_ids.tolist() == [0, 1, 2]

    table = bootstrap_summary(
        units,
        panel,
        "L_cum",
        n_bootstrap=500,
        rng=np.random.default_rng(1),
    ).set_index("unit")
    assert table.loc["A", "rank"] == 1
    assert table.loc["A", "prob_top5"] == 1.0
    assert table.loc["B", "rank"] == 2
