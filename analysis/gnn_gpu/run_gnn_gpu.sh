#!/usr/bin/env bash
# Step 5-7 on GPU server (run from step4_gpu directory with scripts copied in)
set -euo pipefail
DATA_DIR="${DATA_DIR:-.}"
NETWORK_DIR="${NETWORK_DIR:?Set NETWORK_DIR to network_structures}"
STATIONS_CSV="${STATIONS_CSV:?Set STATIONS_CSV to stations_2026_48states.csv}"
WINDOWS_NODES="${WINDOWS_NODES:-windows_nodes.parquet}"
DEVICE="${DEVICE:-cuda}"

python step5_build_pyg_dataset.py --data-dir "$DATA_DIR" \
  --windows-nodes "$WINDOWS_NODES" \
  --network-dir "$NETWORK_DIR" \
  --stations-csv "$STATIONS_CSV"

python step6_train_gnn.py --dataset "$DATA_DIR/pyg/pyg_dataset.pt" --model gat --device "$DEVICE"
python step6_train_gnn.py --dataset "$DATA_DIR/pyg/pyg_dataset.pt" --model gcn --device "$DEVICE" --out "$DATA_DIR/pyg/train_gcn"

python step7_evaluate_gnn.py \
  --dataset "$DATA_DIR/pyg/pyg_dataset.pt" \
  --checkpoint "$DATA_DIR/pyg/train_gat/best_model.pt" \
  --out "$DATA_DIR/results_gnn/gat" \
  --device "$DEVICE" --loo

echo "Done."
