# Step 4: Simulation labels (Y1, Y2, Y3)

## Targets (primary: 10-point curves)

Fixed removal percentages for **all** windows (GNN-friendly fixed dim):

`5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%` (no 100% — fully removed network is trivial)

At each point: **relative efficiency loss** \((E_0 - E_k) / E_0\) after removing that
fraction of nodes in attack order.

| Attack | Columns | Runs |
|--------|---------|------|
| Betweenness | `y_betweenness_p5` … `p90` | 1 (deterministic) |
| Capacity | `y_capacity_p5` … `p90` | 1 (deterministic) |
| Random | `y_random_p5` … `p90` | **10** random orders, mean per point (default) |

**30 labels per window** (10 per attack head). Optional `--also-auc` adds scalar AUC.

For each Step-3 **window subgraph**:
1. Sort nodes by attack order (or random permutation)
2. At each target %, remove \(\mathrm{round}(p \cdot n)\) nodes (capped at \(n-1\))
3. Record loss curve; random curve = average over 10 simulations (default)

**Pilot** (optional, before full batch): compare 10 vs 20 vs 30 reps on 5 windows:

```bash
python step4_compute_y_labels.py --data-dir . --pilot-random
```

Uses the same 30 permutations per window; reports max relative diff vs 30-rep mean.
If 10-rep max diff &lt; 1%, `n_random=10` is sufficient.

Efficiency = weighted global efficiency (inverse shortest-path distance, edge weight = Step-2 normalized distance).

## Upload to GPU server (recommended: one zip)

5093 个 `.npz` 小文件不便上传，请先打包：

```bash
python analysis/step4_pack_subgraphs.py
# -> data/step4_gpu/subgraphs.zip (~15 MB)
```

**最小上传包：**

```
data/step4_gpu/
  manifest.csv
  windows_meta.csv
  subgraphs.zip            # 单文件，替代 subgraphs/ 目录
analysis/
  step4_compute_y_labels.py
  step4_archive_utils.py
  step4_pack_subgraphs.py
requirements-step4.txt
```

服务器上会自动解压再计算（见下）。

## Prerequisite: fix Step-3 node lists

If `windows_nodes.parquet` has duplicate `node_id` per window (BallTree bug), run once:

```bash
python analysis/fix_step3_window_nodes.py
```

## Run on server

```bash
pip install numpy pandas networkx scipy pyarrow tqdm

# 只上传了 zip 时（自动解压到 data/step4_gpu/subgraphs/）:
python analysis/step4_compute_y_labels.py --data-dir data/step4_gpu --archive data/step4_gpu/subgraphs.zip --workers 16

# 或已手动解压 subgraphs/ 目录:
python analysis/step4_compute_y_labels.py --data-dir data/step4_gpu --workers 16

# Full pipeline (needs network pickles + Step-3 outputs):
bash analysis/step4_gpu/run_step4.sh
```

Environment: `WORKERS=32` for more CPU parallelism (NetworkX; GPU not required).

## Outputs

```
data/step4_gpu/labels/y_labels.csv
data/step4_gpu/labels/y_labels.parquet
```

Columns: `window_id`, `y_betweenness_auc`, `y_capacity_auc`, `y_random_auc`, `y_random_auc_std`, `n_nodes`, merged with `manifest` strata fields.

Optional: `--save-curves` writes per-window loss curves under `labels/curves/`.

## Merge with Step-3 features (ML)

Join on `window_id`:

```python
import pandas as pd
y = pd.read_parquet("data/step4_gpu/labels/y_labels.parquet")
meta = pd.read_csv("data/step4_gpu/windows_meta.csv")
df = y.merge(meta, on="window_id")
# X from node features aggregated per window (your Step 5+)
```

## Note on GPU

Attack simulation uses **NetworkX + Dijkstra** (exact efficiency). This step is **CPU-bound**; typical workflow is many-core CPU on the GPU machine. Pure GPU would need cuGraph/custom APSP—not included here.
