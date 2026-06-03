# Step 5–7: PyG dataset + GAT/GCN training (GPU)

## Data layout (on server)

```
step4_gpu/
  manifest.csv
  windows_meta.csv
  labels/y_labels.parquet
  subgraphs/*.npz
  windows_nodes.parquet          # upload from Step 3 (for node lat/lon)
```

Optional (for `dcfc_ratio`):

```
stations_2026_48states.csv
outputs/network_graph_2026_step2/network_structures/network_*.pkl
```

Or run `step5_enrich_subgraphs_npz.py` once so each `.npz` contains `lat`, `lon`, `dcfc_ratio`.

## Install (GPU)

```bash
pip install -r requirements-gnn.txt
# PyG wheels (CUDA 11.8 example):
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

## Step 5 — PyG `Data`

Each graph:

| Field | Shape | Meaning |
|-------|-------|---------|
| `x` | `[N, 4]` | lon, lat, capacity, DCFC ratio (globally standardized) |
| `edge_index` | `[2, E]` | Voronoi (undirected, both directions) |
| `edge_attr` | `[E, 1]` | normalized distance |
| `y` | `[30]` | betweenness(10) + capacity(10) + random(10) |

Upload folder `analysis/gnn_gpu/` into `step4_gpu/` (contains `gnn/`, step5–7 scripts, `build_charging_network_step2.py`).

```bash
cd /opt/data_repo/mliang_work/step4_gpu
# after upload: step4_gpu/gnn/, step5_build_pyg_dataset.py, ...

python step5_build_pyg_dataset.py --data-dir . \
  --windows-nodes windows_nodes.parquet \
  --network-dir /path/to/network_structures \
  --stations-csv /path/to/stations_2026_48states.csv
```

Output: `pyg/pyg_dataset.pt` (graphs + east/west masks).

Split: **train** = window `center_lon > -100` (East), **test** = West.

## Step 6 — Train

```bash
# Main model: GAT
python step6_train_gnn.py --dataset pyg/pyg_dataset.pt --model gat --device cuda --epochs 150

# Baseline: GCN
python step6_train_gnn.py --dataset pyg/pyg_dataset.pt --model gcn --device cuda --out pyg/train_gcn
```

Loss: **MSE** on full 30-dim `y` (three heads, 10 points each).

Checkpoints: `pyg/train_gat/best_model.pt`

## MLflow (experiment tracking, default ON)

```bash
pip install mlflow
# local UI (on server or after downloading mlruns/):
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

Each Step 6/7/8 run logs params, metrics, artifacts under experiment `ev-charging-resilience-gnn`.
Runs are stored in `./mlruns/` (or set `--mlflow-tracking-uri`).

Disable: add `--no-mlflow` to any script.

## Step 7 — Evaluate

```bash
python step7_evaluate_gnn.py \
  --dataset pyg/pyg_dataset.pt \
  --checkpoint pyg/train_gat/best_model.pt \
  --out results_gnn/gat \
  --device cuda

# + leave-one-state-out (retrains per state, slower):
python step7_evaluate_gnn.py ... --loo --loo-epochs 40
```

Outputs:

- `test_metrics.json` — West test R² / MAE (overall + per attack)
- `pred_vs_true_west.png`
- `attention_maps/` — top GAT edges (GAT only)
- `leave_one_state_out.csv` (with `--loo`)

## Note on Y

Labels are **10-point attack curves** (not scalar AUC): 5%–90% relative efficiency loss.

GNN heads match Step 4:

```python
self.head_betweenness = Linear(32, 10)
self.head_capacity = Linear(32, 10)
self.head_random = Linear(32, 10)
```
