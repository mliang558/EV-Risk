#!/usr/bin/env python3
"""Build data/processed/census_tract_strata.csv (run from lgb_fastapi_ui on server)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from census_tract_strata import load_or_build_tract_strata
from project_paths import resolve_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build census_tract_strata.csv cache")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    parser.add_argument(
        "--allow-cbsa-fallback",
        action="store_true",
        help="Allow CBSA+TIGER download when CENSUS_API_KEY is unset (often fails on GPU nodes)",
    )
    args = parser.parse_args()
    if args.allow_cbsa_fallback:
        os.environ["ALLOW_CBSA_FALLBACK"] = "1"

    data_dir = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent)).resolve()
    root = resolve_project_root(data_dir)
    print(f"Project root: {root}", flush=True)

    if not os.environ.get("CENSUS_API_KEY", "").strip() and not args.allow_cbsa_fallback:
        print(
            "Tip: set CENSUS_API_KEY (free) so this step needs no Census TIGER download.\n"
            "  https://api.census.gov/data/key_signup.html\n",
            flush=True,
        )

    try:
        load_or_build_tract_strata(root, force_rebuild=args.force)
    except RuntimeError as e:
        print(str(e), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
