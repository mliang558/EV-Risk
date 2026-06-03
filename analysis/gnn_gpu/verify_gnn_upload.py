#!/usr/bin/env python3
"""Run on GPU server after upload: python verify_gnn_upload.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

checks = []

# step8 version
s8 = ROOT / "step8_benchmark_speed.py"
if s8.exists():
    text = s8.read_text(encoding="utf-8")
    ok = "STEP8_VERSION" in text and "inf-only" in text
    checks.append(("step8_benchmark_speed.py (root)", ok))
else:
    checks.append(("step8_benchmark_speed.py (root)", False))

# edge_utils
eu = ROOT / "gnn" / "edge_utils.py"
if eu.exists():
    text = eu.read_text(encoding="utf-8")
    ok = "align_edge_tensors_for_gat" in text and "edge_attr_row_count" in text
    checks.append(("gnn/edge_utils.py", ok))
else:
    checks.append(("gnn/edge_utils.py", False))

# models
mo = ROOT / "gnn" / "models.py"
if mo.exists():
    text = mo.read_text(encoding="utf-8")
    ok = "align_edge_tensors_for_gat" in text and "global_dim" in text and "gf.numel() == b * f" in text
    checks.append(("gnn/models.py (GAT+global fuse)", ok))
else:
    checks.append(("gnn/models.py (GAT+global fuse)", False))

s6 = ROOT / "step6_train_gnn.py"
if s6.exists():
    text = s6.read_text(encoding="utf-8")
    ok = "--dry-run" in text and "--no-sanitize" in text
    checks.append(("step6_train_gnn.py (--dry-run)", ok))
else:
    checks.append(("step6_train_gnn.py (--dry-run)", False))

probe = ROOT / "step6_probe_bus.py"
checks.append(("step6_probe_bus.py", probe.is_file()))

# live import test
try:
    from gnn.edge_utils import align_edge_tensors_for_gat
    import torch

    ei = torch.randint(0, 10, (2, 272))
    ea = torch.randn(138, 1)
    ei2, ea2 = align_edge_tensors_for_gat(ei, ea)
    ok = ei2.size(1) == ea2.size(0)
    checks.append(("align_edge_tensors_for_gat (272 vs 138)", ok))
except Exception as e:
    checks.append((f"import test: {e}", False))

print("=== Upload verification ===")
all_ok = True
for name, ok in checks:
    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {name}")
    all_ok = all_ok and ok

if all_ok:
    print("\nReady. Run:")
    print("  cd", ROOT)
    print("  python -u step8_benchmark_speed.py --dataset pyg/pyg_dataset.pt \\")
    print("    --checkpoint pyg/train_gat/best_model.pt --data-dir . --device cuda \\")
    print("    --n-samples 5 --inf-only")
    print("  PYTHONPATH=. python -u step6_probe_bus.py --dataset pyg/pyg_dataset.pt")
else:
    print("\nRe-upload files from analysis/gnn_gpu/ to this directory.")
    sys.exit(1)
