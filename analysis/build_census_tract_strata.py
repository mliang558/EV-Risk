#!/usr/bin/env python3
"""Pre-build census tract urban/suburban/rural table (Census API or CBSA fallback)."""

from __future__ import annotations

import argparse

from census_tract_strata import load_or_build_tract_strata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build census_tract_strata.csv cache")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    args = parser.parse_args()
    load_or_build_tract_strata(force_rebuild=args.force)


if __name__ == "__main__":
    main()
