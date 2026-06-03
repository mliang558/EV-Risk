"""Resolve directory containing network_<STATE>.pkl files."""

from __future__ import annotations

from pathlib import Path


def resolve_network_dir(arg: str | None = None, *, cwd: Path | None = None) -> Path:
    """
    Find folder with network_AL.pkl, network_CA.pkl, ...

    Pass e.g. --network-dir network_structures or omit for auto-search under cwd.
    """
    base = (cwd or Path.cwd()).resolve()
    candidates: list[Path] = []
    if arg:
        candidates.append(Path(arg))
    candidates.extend(
        [
            base / "network_structures",
            base.parent / "network_structures",
            base / "outputs" / "network_graph_2026_step2" / "network_structures",
        ]
    )

    tried: list[str] = []
    for p in candidates:
        p = p.expanduser().resolve()
        tried.append(str(p))
        if not p.is_dir():
            continue
        if any(p.glob("network_*.pkl")):
            return p

    raise FileNotFoundError(
        "No network_*.pkl found. Tried:\n  "
        + "\n  ".join(tried)
        + "\n\nUse: --network-dir network_structures\n"
        "(not the placeholder /path/to/network_structures)"
    )
