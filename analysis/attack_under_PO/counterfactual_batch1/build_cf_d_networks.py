#!/usr/bin/env python3
"""
Pre-build CF-D densified networks from EXISTING 2023 10 km pickles.

Does not rebuild the base hypernode graphs — only appends coverage-expansion
nodes under outputs/network_graph_cf_densify_2023/.

Usage:
  python analysis/attack_under_PO/counterfactual_batch1/build_cf_d_networks.py
  python analysis/attack_under_PO/counterfactual_batch1/build_cf_d_networks.py --only TX --modes nevi,pop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ATTACK = Path(__file__).resolve().parents[1]
for p in (ROOT, ATTACK, ATTACK.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from monte_carlo_outage_attack import load_network  # noqa: E402
from counterfactual_batch1.densify_network import (  # noqa: E402
    NULL_REPLICATES,
    load_or_build_densified,
)
from counterfactual_batch1.scenarios import DOSE_PCTS  # noqa: E402
from compute_impact_radius import STATE_NAME_TO_ABBR  # noqa: E402

DEFAULT_NETWORK_ROOT = ROOT / "outputs" / "network_graph_10km_2018_2026"
DEFAULT_CACHE = ROOT / "outputs" / "network_graph_cf_densify_2023"
NETWORK_YEAR = 2023


def members_to_abbrs(members: str) -> list[str]:
    out = []
    for tok in members.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if len(tok) == 2:
            out.append(tok.upper())
        else:
            out.append(STATE_NAME_TO_ABBR.get(tok, tok))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prebuild CF-D densified nets")
    parser.add_argument("--network-root", type=str, default=str(DEFAULT_NETWORK_ROOT))
    parser.add_argument("--cache", type=str, default=str(DEFAULT_CACHE))
    parser.add_argument("--only", type=str, default="", help="Comma units, e.g. TX,CA")
    parser.add_argument(
        "--modes",
        type=str,
        default="nevi,pop,uniform",
        help="Comma: nevi,pop,uniform",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    network_root = Path(args.network_root)
    cache = Path(args.cache)
    year_dir = network_root / str(NETWORK_YEAR)
    summary = __import__("pandas").read_csv(year_dir / "network_summary.csv")
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]

    n_built = 0
    for _, row in summary.iterrows():
        unit = str(row["unit"])
        display = str(row.get("display_name", unit))
        if only and unit not in only and display not in only:
            continue
        pkl = year_dir / "network_structures" / f"network_{unit}.pkl"
        if not pkl.is_file():
            alt = year_dir / "network_structures" / f"network_region_{unit}.pkl"
            pkl = alt if alt.is_file() else pkl
        if not pkl.is_file():
            print(f"[skip] missing {unit}")
            continue
        G = load_network(pkl)
        abbrs = members_to_abbrs(str(row.get("members", unit)))
        for mode in modes:
            for pct in DOSE_PCTS:
                frac = pct / 100.0
                if mode == "uniform":
                    for r in range(NULL_REPLICATES):
                        _, meta = load_or_build_densified(
                            G,
                            unit,
                            frac,
                            mode,
                            cache,
                            replicate=r,
                            member_state_abbrs=abbrs,
                            force=args.force,
                        )
                        n_built += 1
                        print(
                            f"[ok] {unit} {mode}_p{pct}_r{r} +{meta.get('n_added', '?')}",
                            flush=True,
                        )
                else:
                    _, meta = load_or_build_densified(
                        G,
                        unit,
                        frac,
                        mode,
                        cache,
                        member_state_abbrs=abbrs,
                        force=args.force,
                    )
                    n_built += 1
                    print(
                        f"[ok] {unit} {mode}_p{pct} +{meta.get('n_added', '?')}",
                        flush=True,
                    )
    print(f"Done: {n_built} densified graphs → {cache}")


if __name__ == "__main__":
    main()
