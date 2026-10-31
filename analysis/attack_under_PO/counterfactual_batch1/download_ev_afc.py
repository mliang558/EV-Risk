#!/usr/bin/env python3
"""Download and validate the official USDOT/BTS EV Alternative Fuel Corridors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


SERVICE = (
    "https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/"
    "NTAD_Alternative_Fuel_Corridors/FeatureServer/0/query"
)
FILENAME = "ntad_ev_alternative_fuel_corridors_round8_2025.geojson"


def _request(**parameters) -> dict:
    url = f"{SERVICE}?{urlencode(parameters)}"
    with urlopen(url, timeout=120) as response:
        payload = json.load(response)
    if "error" in payload:
        raise RuntimeError(f"AFC service error: {payload['error']}")
    return payload


def download() -> dict:
    count = int(
        _request(where="EV > 0", returnCountOnly="true", f="json")["count"]
    )
    features: list[dict] = []
    offset = 0
    page_size = 500
    while offset < count:
        page = _request(
            where="EV > 0",
            outFields="*",
            returnGeometry="true",
            outSR=4326,
            orderByFields="OBJECTID",
            resultOffset=offset,
            resultRecordCount=page_size,
            f="geojson",
        )
        rows = page.get("features", [])
        if not rows:
            raise RuntimeError(f"AFC service returned no records at offset {offset}")
        features.extend(rows)
        offset += len(rows)
        print(f"downloaded {offset}/{count} corridor segments", flush=True)

    object_ids = [feature.get("properties", {}).get("OBJECTID") for feature in features]
    if len(features) != count or len(set(object_ids)) != count:
        raise RuntimeError(
            f"Incomplete or duplicate AFC download: expected={count}, got={len(features)}"
        )
    for feature in features:
        properties = feature.get("properties", {})
        if float(properties.get("EV", 0) or 0) <= 0 or not properties.get("ELECTRICVE"):
            raise RuntimeError("Downloaded record is not a designated EV corridor")
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw") / FILENAME)
    args = parser.parse_args()

    payload = download()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.out)

    digest = hashlib.sha256(encoded).hexdigest()
    metadata = {
        "source": SERVICE.rsplit("/query", 1)[0],
        "filter": "EV > 0 (FHWA-designated EV corridors, ready and pending)",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(payload["features"]),
        "sha256": digest,
    }
    args.out.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"PASS: downloaded {len(payload['features'])} official EV AFC segments\n"
        f"data: {args.out.absolute()}\nsha256: {digest}"
    )


if __name__ == "__main__":
    main()
