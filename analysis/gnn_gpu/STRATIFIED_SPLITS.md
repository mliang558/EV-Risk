# Stratified train/test splits (urban / suburban / rural)

Three protocols are stored in `pyg/pyg_dataset.pt` → `split_masks`:

| Key | Description |
|-----|-------------|
| `east_west_*` | East (`center_lon > -100`) train, West test — **geographic** generalization |
| `stratified_stratum_*` | Within each stratum, ~30% windows → test; **matched urban/suburban/rural %** in train & test |
| `stratified_state_*` | Whole states held out into test until ~30% windows; greedy match of stratum mix |

## Rebuild dataset masks (once)

```bash
PYTHONPATH=. python -u step5_build_pyg_dataset.py --data-dir . \\
  --splits east_west,stratified_stratum,stratified_state \\
  --test-frac 0.3 --split-seed 42
```

Prints stratum % for each split (global | train | test).

## Train on stratified split (recommended for fair comparison)

**Trains** on train_mask and evaluates on test_mask under the same protocol.  
Does **not** reuse East/West checkpoints.

```bash
sed -i 's/\r$//' run_stratified_splits_train.sh
chmod +x run_stratified_splits_train.sh

# Start with one split + main models (LGB + GAT L5):
MINIMAL=1 SPLITS=stratified_stratum nohup bash run_stratified_splits_train.sh > logs/stratified_train.log 2>&1 &

# Full 9-model sweep (both splits, many hours):
nohup bash run_stratified_splits_train.sh > logs/stratified_train.log 2>&1 &
```

Outputs:

- `pyg/train_gat_global_L5_ext_br_{split}/best_model.pt`
- `lgbm/train_lgbm_ext_br_{split}/lgb_report.json`
- `results_gnn/stratified_eval/{split}/eval_*/test_metrics.json`

Optional: bake masks into `.pt` once (otherwise Step 6/7 compute masks on the fly — OK):

```bash
PYTHONPATH=. python -u step5_build_pyg_dataset.py --data-dir . \
  --splits east_west,stratified_stratum,stratified_state --test-frac 0.3 --split-seed 42
```

**Upload these** (server copies are often older than git):

- `gnn/splits.py`
- `step6_train_gnn.py`, `step7_evaluate_gnn.py`, `step7_window_lgb_baseline.py` (must accept `--split`)
- `run_stratified_splits_train.sh`

### Two GPUs (faster)

Pipeline uses **one GPU per process** (no multi-GPU inside one model). Run **different jobs** on GPU 0 and GPU 1.

**A. Two splits in parallel** (good with `MINIMAL=1`: LGB+GAT on each split):

```bash
PARALLEL_SPLITS=1 MINIMAL=1 \
  nohup bash run_stratified_splits_train_2gpu.sh > logs/stratified_2gpu.log 2>&1 &
```

**B. One split, models split across GPUs** (full 8 GNN sweep ~2× faster):

```bash
SPLITS=stratified_stratum MINIMAL=0 \
  nohup bash run_stratified_splits_train_2gpu.sh > logs/stratified_2gpu.log 2>&1 &
```

**C. Manual** (any two commands):

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda PYTHONPATH=. .venv_gnn/bin/python -u step6_train_gnn.py ... --device cuda &
CUDA_VISIBLE_DEVICES=1 DEVICE=cuda PYTHONPATH=. .venv_gnn/bin/python -u step6_train_gnn.py ... --device cuda &
wait
```

Check GPUs: `nvidia-smi`. LGB is mostly CPU — it can run while GNN trains on both cards.

**D. Single job on GPU 1 only:** `CUDA_VISIBLE_DEVICES=1 DEVICE=cuda bash run_stratified_splits_train.sh`

---

**FAQ**

| Question | Answer |
|----------|--------|
| Progress bar? | GNN: `tqdm` epoch bar in Step6 (`pip install tqdm`). LGB: feature build may show tqdm; fit is quick. Shell: `tail -f logs/stratified_train.log` |
| Resume after kernel restart? | Default `SKIP_EXISTING=1`: re-run the same `nohup bash run_stratified_splits_train.sh ...` — skips models that already have `best_model.pt` / `lgb_report.json` |
| Overwrite old East/West models? | **No.** New dirs: `pyg/train_*_stratified_stratum/`, `lgbm/train_lgbm_ext_br_stratified_stratum/`. Old `pyg/train_gat_global_L5_ext_br/` unchanged |
| `unrecognized arguments: --split`? | Upload `step7_window_lgb_baseline.py` (and `gnn/splits.py`) from `analysis/gnn_gpu/` |

## Eval only (reuse East/West checkpoints — old behavior)

Uses checkpoints from `experiments/enhancement_manifest.json`.  
**Does not overwrite** `results_gnn/eval_train_*` (east_west).

```bash
sed -i 's/\r$//' run_stratified_splits.sh
chmod +x run_stratified_splits.sh
nohup bash run_stratified_splits.sh > logs/stratified_splits.log 2>&1 &
```

### MLflow tracking (visualization)

Each model × split gets one run in experiment **`ev-charging-stratified-splits`**:

- Metrics: `test_overall_r2`, `test_betweenness_r2`, `test_random_r2`, stratum % (train/test)
- Artifacts: `test_metrics.json`, per-split `comparison_table.md`

After the job finishes:

```bash
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

JupyterHub: use port-forward or your cluster’s proxy for port **5000**.  
Disable logging: `NO_MLFLOW=1 bash run_stratified_splits.sh`

Outputs:

- `results_gnn/stratified_eval/stratified_stratum/{model}/test_metrics.json`
- `results_gnn/stratified_eval/stratified_state/{model}/test_metrics.json`
- `lgbm/train_lgbm_ext_br_{split}/lgb_report.json`
- `results_gnn/stratified_eval/all_splits_comparison.md` (east_west from old + new splits)

## LightGBM-only quick comparison

```bash
PYTHONPATH=. python -u step7_stratified_split_experiment.py \\
  --data-dir . --out results_gnn/stratified_split_lgb
```

## Per-script `--split`

- `step6_train_gnn.py --split stratified_stratum`
- `step7_evaluate_gnn.py --split stratified_stratum`
- `step7_window_lgb_baseline.py --split stratified_stratum`

If masks missing in `.pt`, splits are **recomputed on the fly** from `windows_meta.csv` + `manifest.csv`.

## Paper wording

> We report East–West geographic holdout and two stratum-balanced splits: (i) per-window stratified sampling by window radius class (10/30/80 km ≈ urban/suburban/rural), and (ii) state-level holdout with matched stratum proportions, to separate spatial generalization from urban–rural distribution shift.
