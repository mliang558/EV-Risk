# Enhancement experiments — cleanup, upload, run, report

**Keep data layout unchanged:** `manifest.csv`, `windows_meta.csv`, `labels/`, `subgraphs/`, `pyg/pyg_dataset.pt`, `.venv_gnn/`.

---

## Step 1 — Delete on server (safe)

```bash
cd /opt/data_repo/mliang_work/step4_gpu

# Outputs / checkpoints (re-train)
rm -rf results_gnn mlruns pyg/train_* __pycache__
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Messy scripts (re-upload clean copy)
rm -rf gnn/gnn
rm -f step*.py verify_gnn_upload.py run_gnn_gpu.sh
rm -rf gnn
rm -f constants.py models.py metrics.py dataset.py 2>/dev/null

# DO NOT DELETE:
# subgraphs/ labels/ pyg/pyg_dataset.pt manifest.csv windows_meta.csv
# windows_nodes.parquet network_structures/ .venv_gnn/
```

Optional backup:

```bash
mkdir -p ~/ckpt_backup
cp -r pyg/train_gat_global_br results_gnn/lgb_global_br/lgb_report.json ~/ckpt_backup/ 2>/dev/null
```

---

## Step 2 — Upload from `analysis/gnn_gpu/`

### `gnn/` (flat, no nested `gnn/gnn/`)

- `constants.py`, `attack_targets.py`, `train_utils.py`, `models.py`, `metrics.py`
- `edge_utils.py`, `global_graph_features.py`, `local_window_features.py`
- `mlflow_utils.py`, `splits.py`, `__init__.py`

### Scripts

| File | Role |
|------|------|
| `step5_build_window_global_features.py` | **NEW** — build 15 or 18-dim global CSV |
| `step6_train_gnn.py` | `--num-layers`, `--model transformer`, `--global-feature-set` |
| `step7_evaluate_gnn.py` | load deep / extended ckpt |
| `step7_window_lgb_baseline.py` | `--global-feature-set extended` |
| `step7_build_comparison_table.py` | original paper table |
| `step7_build_enhancement_report.py` | **NEW** — multi-run summary |
| `step7_compare_single_vs_multitask.py` | single-task vs multi |
| `experiments/enhancement_manifest.json` | **NEW** — list of runs for report |

---

## About “larger window”

Current 10k windows use **fixed radii** in `windows_meta.csv` (urban 10 km / suburban 30 km / rural 80 km).

- **Cannot** enlarge windows without re-running **Step 3 → 4 → 5** (new `subgraphs/`, `labels/`, `pyg_dataset.pt`).
- This plan improves **model + global features** on **existing** windows.
- For larger-window data later: change radii in `step3_random_seed_sample.py`, rebuild into e.g. `step4_gpu_large/`.

---

## Extended global features (18-dim)

| Feature | In set | Note |
|---------|--------|------|
| `diameter` | base (15) | already included |
| `algebraic_connectivity` | **extended** | Fiedler value |
| `avg_shortest_path` | **extended** | largest component, weighted |
| `degree_heterogeneity` | **extended** | std(degree)/mean(degree) |

---

## Step 3 — Run pipeline

```bash
cd /opt/data_repo/mliang_work/step4_gpu
export PYTHONPATH=.
export PY=.venv_gnn/bin/python

# A) Build extended global CSV (~30–90 min without --fast)
$PY -u step5_build_window_global_features.py --data-dir . \
  --feature-set extended \
  --out results_gnn/global_extended/window_global_features.csv

# B) LGB extended baseline
$PY -u step7_window_lgb_baseline.py --data-dir . \
  --attacks no-capacity --feature-set global \
  --global-feature-set extended \
  --reuse-features \
  --out results_gnn/lgb_global_ext_br

# C) GAT 5-layer + extended global (multi-task B+R)
$PY -u step6_train_gnn.py --dataset pyg/pyg_dataset.pt --model gat \
  --num-layers 5 --hidden 64 --global-features \
  --global-feature-set extended \
  --global-features-csv results_gnn/global_extended/window_global_features.csv \
  --attacks no-capacity \
  --out pyg/train_gat_global_L5_ext_br

# D) Graph Transformer 4-layer + extended
$PY -u step6_train_gnn.py --dataset pyg/pyg_dataset.pt --model transformer \
  --num-layers 4 --hidden 64 --global-features \
  --global-feature-set extended \
  --global-features-csv results_gnn/global_extended/window_global_features.csv \
  --attacks no-capacity \
  --out pyg/train_gt_global_L4_ext_br

# E) Single-task betweenness (optional, for B head)
$PY -u step6_train_gnn.py --dataset pyg/pyg_dataset.pt --model gat \
  --num-layers 5 --hidden 64 --global-features \
  --global-feature-set extended \
  --global-features-csv results_gnn/global_extended/window_global_features.csv \
  --attack betweenness \
  --out pyg/train_gat_global_L5_ext_betweenness

# F) Evaluate each ckpt
$PY -u step7_evaluate_gnn.py --dataset pyg/pyg_dataset.pt \
  --checkpoint pyg/train_gat_global_L5_ext_br/best_model.pt \
  --global-features-csv results_gnn/global_extended/window_global_features.csv \
  --out results_gnn/eval_gat_L5_ext --no-plots

# G) Combined enhancement table
$PY -u step7_build_enhancement_report.py --data-dir . \
  --manifest experiments/enhancement_manifest.json \
  --out results_gnn/enhancement_report
```

Output: `results_gnn/enhancement_report/enhancement_table.md`

## Generalization (fast path — default in `run_all_experiments.sh`)

**Main text — region holdout** (one train, ~hours): train West+Midwest+South, test **Northeast**:

```bash
$PY -u step7_leave_one_state_out.py --mode region --holdout-region northeast \
  --checkpoint pyg/train_gat_global_L5_ext_br/best_model.pt \
  --global-features-csv results_gnn/global_extended/window_global_features.csv \
  --epochs 40 --out results_gnn/geo_holdout_northeast
```

**Appendix — 5-state LOO** (DE, CA, TX, ME, IA):

```bash
$PY -u step7_leave_one_state_out.py --mode loo --representative \
  --checkpoint pyg/train_gat_global_L5_ext_br/best_model.pt \
  --global-features-csv results_gnn/global_extended/window_global_features.csv \
  --epochs 40 --out results_gnn/loo5_representative
```

**Optional** east/west retrain: `--mode east_west` (same lon>-100 as main split).

**Full 48-state LOO** (days): `SKIP_FULL_LOO=0 SKIP_LOO=0` in `run_all_experiments.sh`.

**Radius sensitivity** (10 / 30 / 80 km West): `step7_r2_by_radius.py` → `r2_by_radius_L5_ext/`.

Env flags: `SKIP_GEO=1` skip region+5-state; `SKIP_FULL_LOO=0` enable full LOO.

Edit `experiments/enhancement_manifest.json` after each eval so paths match your runs.

---

## Step 4 — Original comparison table (optional)

```bash
$PY -u step7_build_comparison_table.py --from-artifacts --data-dir . \
  --gat-global pyg/train_gat_global_L5_ext_br/best_model.pt \
  --out results_gnn/comparison_table
```
