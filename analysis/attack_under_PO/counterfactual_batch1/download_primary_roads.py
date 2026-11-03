#!/usr/bin/env python3
"""Download the pinned 2023 Census TIGER/Line national primary-roads file."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

# Allow this file to be executed directly from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from densify_network import PRIMARY_ROADS_FILENAME, PRIMARY_ROADS_URL, ROOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "raw" / PRIMARY_ROADS_FILENAME,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and not args.force:
        print(f"EXISTS: {out}\nSHA256: {_sha256(out)}")
        return

    request = urllib.request.Request(
        PRIMARY_ROADS_URL,
        headers={"User-Agent": "EV-Risk primary-roads downloader/1.0"},
    )
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".zip", delete=False) as temp:
        temp_path = Path(temp.name)
        with urllib.request.urlopen(request, timeout=300) as response:
            while chunk := response.read(1024 * 1024):
                temp.write(chunk)
    try:
        with zipfile.ZipFile(temp_path) as archive:
            names = archive.namelist()
            if not any(name.lower().endswith(".shp") for name in names):
                raise ValueError("Downloaded archive contains no shapefile")
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt member in downloaded archive: {bad}")
        temp_path.replace(out)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"DOWNLOADED: {PRIMARY_ROADS_URL}\nSAVED: {out}\nSHA256: {_sha256(out)}")


if __name__ == "__main__":
    main()
