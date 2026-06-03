"""Pack / extract Step-4 subgraph archives (single file for upload)."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def pack_subgraphs_dir(
    subgraphs_dir: Path,
    out_archive: Path,
    fmt: str = "zip",
) -> Path:
    """Pack all *.npz under subgraphs_dir into one archive."""
    files = sorted(subgraphs_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz in {subgraphs_dir}")

    out_archive.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "zip":
        with zipfile.ZipFile(out_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in files:
                zf.write(p, arcname=f"subgraphs/{p.name}")
    elif fmt in ("tar.gz", "tgz"):
        with tarfile.open(out_archive, "w:gz") as tf:
            for p in files:
                tf.add(p, arcname=f"subgraphs/{p.name}")
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return out_archive


def extract_subgraphs_archive(
    archive: Path,
    out_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Extract archive to out_dir/subgraphs/. Skips if already extracted unless force=True."""
    subgraphs_dir = out_dir / "subgraphs"
    if subgraphs_dir.exists() and any(subgraphs_dir.glob("*.npz")) and not force:
        return subgraphs_dir

    subgraphs_dir.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive: {archive}")

    if not any(subgraphs_dir.glob("*.npz")):
        raise RuntimeError(f"No .npz after extract to {subgraphs_dir}")
    return subgraphs_dir


def resolve_subgraph_path(data_dir: Path, subgraph_rel: str) -> Path:
    return data_dir / subgraph_rel
