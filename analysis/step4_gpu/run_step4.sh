#!/usr/bin/env bash
# Step 4 pipeline for Linux GPU server (CPU multiprocessing for NetworkX attacks).
set -euo pipefail
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

echo "=== Step 4.1 Export window subgraphs ==="
python analysis/step4_export_window_subgraphs.py \
  --windows-dir outputs/window_samples_2026_step3 \
  --network-dir outputs/network_graph_2026_step2/network_structures \
  --out data/step4_gpu

echo "=== Step 4.1b Pack subgraphs (optional, for upload) ==="
if [ ! -f data/step4_gpu/subgraphs.zip ]; then
  python analysis/step4_pack_subgraphs.py --data-dir data/step4_gpu
fi

echo "=== Step 4.2 Compute 30-dim attack curve labels (3x10) ==="
ARCHIVE_ARG=""
if [ -f data/step4_gpu/subgraphs.zip ] && [ ! -d data/step4_gpu/subgraphs ]; then
  ARCHIVE_ARG="--archive data/step4_gpu/subgraphs.zip"
elif [ -f data/step4_gpu/subgraphs.zip ]; then
  ARCHIVE_ARG="--archive data/step4_gpu/subgraphs.zip"
fi
python analysis/step4_compute_y_labels.py \
  --data-dir data/step4_gpu \
  $ARCHIVE_ARG \
  --workers "${WORKERS:-16}"

echo "Done. Labels: data/step4_gpu/labels/y_labels.parquet"
