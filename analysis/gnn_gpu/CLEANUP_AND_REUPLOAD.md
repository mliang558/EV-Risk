# step4_gpu 清空重传指南

目标：删掉乱掉的脚本/缓存/重复目录，**保留数据和 PyG 数据集**；再从本机 `analysis/gnn_gpu/` 干净上传一份。

---

## 一、必须保留（删了要重跑 Step3–5，很慢）

```
step4_gpu/
  manifest.csv
  windows_meta.csv
  windows_nodes.parquet          # 若有（节点 lat/lon）
  labels/
    y_labels.parquet             # 或 y_labels.csv
  subgraphs/
    *.npz                        # ~1 万个窗口子图
  pyg/
    pyg_dataset.pt               # Step5 产物，训练/评估都靠它
```

可选保留（不是训练必需，但能省时间）：

```
  results_gnn/lgb_baseline/window_global_features.csv
  # 或 results_gnn/xgb_baseline/window_global_features.csv
  # → LGB / GAT+global 可 --reuse-features，不必重算 NetworkX

  stations_2026_48states.csv     # 若做过 enrich npz
  outputs/.../network_*.pkl      # 同上

  .venv_gnn/                     # 虚拟环境，别删（重装 torch 很麻烦）
```

---

## 二、可以整目录删掉（可重新训练/生成）

```
  results_gnn/                   # 所有 LGB 报告、对比表、eval JSON
  pyg/train_*/                   # 所有 checkpoint（best_model.pt、history.json）
  mlruns/                        # MLflow 日志
  __pycache__/  **/__pycache__/

  # 乱掉的重复脚本（重传后会覆盖）
  step*.py
  gnn/                           # 整个删掉后，只上传扁平 gnn/*.py（见下）
  gnn/gnn/                       # 嵌套重复，必删
  verify_gnn_upload.py
  run_gnn_gpu.sh
  README*.md
  UPLOAD*.txt

  # 根目录若误传了重复模块（不应在 step4_gpu 根下）
  constants.py  models.py  metrics.py  dataset.py  node_features.py
```

**建议**：想保留已训好的 GAT，删前先备份：

```bash
mkdir -p ~/backup_ckpts
cp -r pyg/train_gat_global_br pyg/train_gat_br pyg/train_mlp ~/backup_ckpts/ 2>/dev/null
cp results_gnn/lgb_global_br/lgb_report.json ~/backup_ckpts/ 2>/dev/null
cp results_gnn/lgb_baseline/window_global_features.csv ~/backup_ckpts/ 2>/dev/null
```

---

## 三、服务器一键清理（先 cd，再核对 ls）

```bash
cd /opt/data_repo/mliang_work/step4_gpu

# 看数据还在
ls -lh pyg/pyg_dataset.pt manifest.csv windows_meta.csv labels/ | head
ls subgraphs | wc -l

# 删输出与脚本（不碰 data 列出的那些）
rm -rf results_gnn mlruns __pycache__ gnn/gnn
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

rm -f step*.py verify_gnn_upload.py run_gnn_gpu.sh
rm -rf gnn
rm -rf pyg/train_gat* pyg/train_gcn* pyg/train_mlp* 2>/dev/null
# 若 pyg/ 里只剩 pyg_dataset.pt 就对了
```

**不要执行**：`rm -rf subgraphs labels pyg/pyg_dataset.pt .venv_gnn`

---

## 四、本机只上传这些（不要整包 gnn_gpu 拖上去）

### 目录结构（上传后应是）

```
step4_gpu/
  gnn/
    __init__.py
    constants.py
    attack_targets.py
    train_utils.py
    models.py
    metrics.py
    edge_utils.py
    global_graph_features.py
    local_window_features.py
    mlflow_utils.py          # step6/7 用 MLflow 时需要
    splits.py                # step7 LOO 时需要
  step5_build_pyg_dataset.py    # 仅当要重建 .pt 时
  step6_train_gnn.py
  step6_probe_bus.py              # 可选
  step7_evaluate_gnn.py
  step7_window_lgb_baseline.py
  step7_build_comparison_table.py
  step7_compare_single_vs_multitask.py
  step7_window_lgb_optuna.py      # 可选
  UPLOAD_AND_RUN.txt
  requirements-gnn.txt            # 可选
```

### 禁止上传（本地 gnn_gpu 里的垃圾）

- `gnn/gnn/` 整个子目录
- `gnn/step6_train_gnn.py` 等（step 脚本应在 **根目录**，不在 gnn/ 里）
- 根目录重复的 `models.py`, `metrics.py`, `constants.py`（与 gnn/ 重复）

---

## 五、上传后自检

```bash
export PYTHONPATH=.
export PY=.venv_gnn/bin/python

grep -n "def parse_attacks" gnn/constants.py
# 必须有输出

$PY -c "from gnn.constants import parse_attacks; print(parse_attacks('betweenness'))"
# ('betweenness',)

ls gnn/gnn 2>/dev/null && echo "BAD: nested gnn/gnn still exists" || echo "OK: flat gnn/"
```

---

## 六、重传后最小跑通顺序

```bash
# 1) 确认数据集
$PY -c "import torch; b=torch.load('pyg/pyg_dataset.pt',map_location='cpu',weights_only=False); print(len(b['graphs']))"

# 2) LGB global（快）
$PY -u step7_window_lgb_baseline.py --data-dir . --reuse-features \
  --attacks no-capacity --feature-set global --out results_gnn/lgb_global_br

# 3) GAT+Global multi-task（或单任务 --attack betweenness）
$PY -u step6_train_gnn.py --dataset pyg/pyg_dataset.pt --model gat --global-features \
  --attacks no-capacity --out pyg/train_gat_global_br

# 4) 对比表
$PY -u step7_build_comparison_table.py --from-artifacts --data-dir . \
  --gat-global pyg/train_gat_global_br/best_model.pt \
  --out results_gnn/comparison_table
```
