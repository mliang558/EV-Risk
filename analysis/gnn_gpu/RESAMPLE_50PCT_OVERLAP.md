# Resample windows: max 50% overlap + cross-stratum

## Problem

Old pipeline used `min_sep_frac=0.2` (center distance ≥ 0.2×R), allowing **~80% disk overlap** between windows. Urban 10 km disks on the East Coast looked almost solid green on the map.

**If dedupe keeps only ~1.4k–2.4k with `placement_order=radius_asc`:** strict **cross-stratum** 50% area overlap on real charging-station geography often caps near ~2–3k nationwide. Use **`--dedupe-strategy per_stratum`** (default in `run_resample_50pct_overlap.sh`): 50% overlap **within** urban / suburban / rural, fill 5:3:2 quotas (~10k). Cross-stratum disks may overlap on the map.

## Fix

- `gnn/window_overlap.py`: geometric disk overlap fraction
- `step3_dedupe_window_centers.py`: greedy dedupe with **`--max-overlap-frac 0.5`** (default)
- **Cross-stratum by default**: urban / suburban / rural windows cannot overlap each other (`--within-stratum-only` to opt out)

Equal-radius guide (approx. min center separation / R for 50% cap):

| Radius | ~min center sep / R |
|--------|---------------------|
| 10 km  | ~0.86 |
| 30 km  | ~0.86 |
| 80 km  | ~0.86 |

## One-shot on server

```bash
cd /opt/data_repo/mliang_work/step4_gpu
mkdir -p logs
export PYTHONPATH=.

# Upload: gnn/window_overlap.py, step3_dedupe_window_centers.py,
#         step3_random_seed_sample.py, run_resample_50pct_overlap.sh

nohup bash run_resample_50pct_overlap.sh > logs/resample_50pct.log 2>&1 &
tail -f logs/resample_50pct.log
```

Then retrain (do **not** reuse old `pyg_dataset.pt` or checkpoints):

```bash
MINIMAL=0 SPLITS=stratified_stratum,stratified_state \
  nohup bash run_stratified_splits_train.sh > logs/stratified_retrain.log 2>&1 &
```

## Manual commands

```bash
PYTHONPATH=. .venv_gnn/bin/python -u step3_random_seed_sample.py \
  --network-dir ./network_structures \
  --out . --target-total 10000 \
  --dedupe-centers --max-overlap-frac 0.5 \
  --oversample-factor 5.0 --dedupe-fill-rounds 10 --seed 42

PYTHONPATH=. .venv_gnn/bin/python -u step3_dedupe_window_centers.py \
  --meta windows_meta.csv --nodes windows_nodes.parquet \
  --max-overlap-frac 0.5 --verify --target 10000
```

## After resample

1. `step4_export_window_subgraphs.py` + `step4_compute_y_labels.py`
2. `step5_build_pyg_dataset.py --splits east_west,stratified_stratum,stratified_state`
3. Retrain LGB + GNN (`run_stratified_splits_train.sh` or `run_all_experiments.sh`)

Backup old data: script copies to `backup_windows_<timestamp>/`.
