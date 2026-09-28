#!/usr/bin/env python3
"""
Batch 2 parallel launcher — families A / B / C (baseline only).

A: net t × outage t
B: net 2023 × outage t   (t=2018..2022 by default; t=2023 shared with A)
C: net t × pooled outages (Batch-1-like λ, no r scaling)

Usage:
  python analysis/attack_under_PO/batch2_yearly/run_parallel.py --families A B C
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ATTACK = Path(__file__).resolve().parents[1]
for p in (ROOT, ATTACK):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from region_merge_config import ABBR_TO_STATE_NAME  # noqa: E402
from batch2_yearly import YEARS  # noqa: E402
from batch2_yearly.rates import build_r_table, ensure_outage_year, save_r_table  # noqa: E402

DEFAULT_NETWORK_ROOT = ROOT / "outputs" / "network_graph_10km_2018_2026"
DEFAULT_OUT = ROOT / "results_batch2_yearly"


def members_abbr_to_full_names(members: str) -> list[str]:
    names: list[str] = []
    for abbr in members.split(","):
        abbr = abbr.strip()
        if abbr:
            names.append(ABBR_TO_STATE_NAME.get(abbr, abbr))
    return names


def _units_for_year(network_root: Path, year: int, only: set[str] | None) -> list[dict]:
    year_dir = network_root / str(year)
    summary = year_dir / "network_summary.csv"
    net_dir = year_dir / "network_structures"
    if not summary.is_file():
        return []
    df = pd.read_csv(summary)
    jobs = []
    for _, row in df.iterrows():
        unit = str(row["unit"])
        display = str(row.get("display_name", unit))
        if only and unit not in only and display not in only:
            continue
        pkl = net_dir / f"network_{unit}.pkl"
        if not pkl.is_file():
            alt = net_dir / f"network_region_{unit}.pkl"
            pkl = alt if alt.is_file() else pkl
        if not pkl.is_file():
            continue
        members = str(row.get("members", unit))
        jobs.append(
            {
                "unit": unit,
                "label": display,
                "member_states": members_abbr_to_full_names(members),
                "pkl_path": str(pkl.resolve()),
                "n_nodes": int(row.get("n_nodes", 0) or 0),
            }
        )
    return jobs


def build_jobs(
    *,
    families: list[str],
    years: list[int],
    network_root: Path,
    only: set[str] | None,
) -> list[dict]:
    """
    Each job: family, year_network, year_outage (None=pooled), unit meta.
    A_2023 and B_2023 share one physical run (family marker 'A_share_B').
    """
    jobs: list[dict] = []
    fams = [f.upper() for f in families]

    if "A" in fams or "B" in fams:
        for t in years:
            # A: always
            if "A" in fams:
                for u in _units_for_year(network_root, t, only):
                    jobs.append(
                        {
                            **u,
                            "family": "A",
                            "year_network": t,
                            "year_outage": t,
                            "share_as_B2023": bool(t == 2023 and "B" in fams),
                        }
                    )
            # B: skip 2023 if A also runs (shared); else include
            if "B" in fams:
                if t == 2023 and "A" in fams:
                    continue
                for u in _units_for_year(network_root, 2023, only):
                    jobs.append(
                        {
                            **u,
                            "family": "B",
                            "year_network": 2023,
                            "year_outage": t,
                            "share_as_B2023": False,
                        }
                    )

    if "C" in fams:
        for t in years:
            for u in _units_for_year(network_root, t, only):
                jobs.append(
                    {
                        **u,
                        "family": "C",
                        "year_network": t,
                        "year_outage": None,
                        "share_as_B2023": False,
                    }
                )

    # largest |V| first
    jobs.sort(key=lambda j: -int(j.get("n_nodes") or 0))
    return jobs


def _out_path(out_root: Path, family: str, year_network: int, year_outage, unit: str) -> Path:
    if year_outage is None:
        return out_root / family / f"net{year_network}_pooled" / unit
    return out_root / family / f"net{year_network}_out{year_outage}" / unit


def _worker(args: tuple) -> str:
    (
        family,
        year_network,
        year_outage,
        label,
        unit,
        member_states,
        pkl_path,
        out_root,
        n_sims,
        force,
        write_events,
        share_as_B2023,
    ) = args

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    root = Path(__file__).resolve().parents[3]
    attack = Path(__file__).resolve().parents[1]
    for p in (str(root), str(attack)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from batch2_yearly.engine import run_unit_batch2

    out_dir = _out_path(
        Path(out_root), family, int(year_network), year_outage, unit
    )
    msg = run_unit_batch2(
        family=family,
        label=label,
        unit=unit,
        member_states=member_states,
        pkl_path=Path(pkl_path),
        out_dir=out_dir,
        year_network=int(year_network),
        year_outage=None if year_outage is None else int(year_outage),
        n_sims=int(n_sims),
        project_root=root,
        force=bool(force),
        write_events=bool(write_events),
    )

    if share_as_B2023:
        # A_2023 ≡ B_2023: copy outputs into B tree and assert metrics match
        b_dir = _out_path(Path(out_root), "B", 2023, 2023, unit)
        b_dir.parent.mkdir(parents=True, exist_ok=True)
        if b_dir.exists():
            shutil.rmtree(b_dir)
        shutil.copytree(out_dir, b_dir)
        # rewrite family tag in copied metrics/summary
        for name in ("metrics_baseline.csv", "summary.csv"):
            p = b_dir / name
            if not p.is_file():
                continue
            df = pd.read_csv(p)
            if "family" in df.columns:
                df["family"] = "B"
            df.to_csv(p, index=False)
        a_m = pd.read_csv(out_dir / "metrics_baseline.csv")
        b_m = pd.read_csv(b_dir / "metrics_baseline.csv")
        # compare loss columns (ignore family label)
        cols = [
            c
            for c in (
                "sim_id",
                "L_tilde",
                "L_cum",
                "total_rel_loss_x_duration",
                "P_hit",
                "n_events",
            )
            if c in a_m.columns and c in b_m.columns
        ]
        merged = a_m[cols].merge(b_m[cols], on="sim_id", suffixes=("_a", "_b"))
        for c in cols:
            if c == "sim_id":
                continue
            if not (merged[f"{c}_a"] - merged[f"{c}_b"]).abs().fillna(0).le(1e-12).all():
                raise AssertionError(f"A_2023 != B_2023 metrics for {unit} col={c}")
        msg = msg + "+shared_B2023"
    return msg


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch 2 yearly A/B/C parallel runner")
    ap.add_argument("--network-root", type=str, default=str(DEFAULT_NETWORK_ROOT))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--families", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--write-events",
        action="store_true",
        help="Write per-sim event dumps (default off)",
    )
    args = ap.parse_args()

    network_root = Path(args.network_root)
    if not network_root.is_absolute():
        network_root = ROOT / network_root
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    jobs = build_jobs(
        families=args.families,
        years=args.years,
        network_root=network_root,
        only=only,
    )

    # write r table once
    outages = ensure_outage_year(
        pd.read_csv(
            ROOT / "notebooks" / "PO_data_cleaning" / "cleaned_outages_2018_2023.csv"
        )
    )
    r_table = build_r_table(outages)
    save_r_table(r_table, out_root / "r_st_table.csv")

    nproc = os.cpu_count() or 8
    workers = args.workers if args.workers is not None else nproc
    workers = max(1, min(int(workers), max(len(jobs), 1)))

    write_events = bool(args.write_events)
    (out_root / "batch_manifest.json").write_text(
        json.dumps(
            {
                "families": args.families,
                "years": args.years,
                "n_sims": args.n_sims,
                "n_jobs": len(jobs),
                "note": "A_2023 written once and copied to B_2023 with assert",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Batch2] jobs={len(jobs)} workers={workers} out={out_root}", flush=True)

    task_args = [
        (
            j["family"],
            j["year_network"],
            j["year_outage"],
            j["label"],
            j["unit"],
            j["member_states"],
            j["pkl_path"],
            str(out_root),
            args.n_sims,
            args.force,
            write_events,
            j.get("share_as_B2023", False),
        )
        for j in jobs
    ]

    t0 = time.time()
    ok = skip = other = 0
    if workers <= 1:
        for a in task_args:
            msg = _worker(a)
            print(msg, flush=True)
            if str(msg).startswith("ok"):
                ok += 1
            elif str(msg).startswith("skip"):
                skip += 1
            else:
                other += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker, a) for a in task_args]
            for fut in as_completed(futs):
                msg = fut.result()
                print(msg, flush=True)
                if str(msg).startswith("ok"):
                    ok += 1
                elif str(msg).startswith("skip"):
                    skip += 1
                else:
                    other += 1
    print(
        f"[Batch2] done ok={ok} skip={skip} other={other} "
        f"in {(time.time()-t0)/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
