# 本机最有用代码 → GitHub

核心目录（与服务器 `step4_gpu` 对应）：

| 路径 | 内容 |
|------|------|
| `analysis/gnn_gpu/gnn/` | 特征、模型、指标、训练工具 |
| `analysis/gnn_gpu/step3_*.py` … `step8_*.py` | Step3–8 流水线 |
| `analysis/gnn_gpu/lgb_fastapi_ui/` | LightGBM + 地图 FastAPI UI |
| `analysis/gnn_gpu/*.sh`, `requirements-gnn.txt` | 运行脚本与依赖 |
| `analysis/*.py`（根目录） | 与 gnn_gpu 同步的副本脚本（可选） |

**不上传**：`*.parquet/csv/pkl/pt/npz`、`pyg/`、`subgraphs/`、`artifacts/`、`results_*`。

## 推送到已有仓库 EV-Risk

```powershell
cd C:\Users\maple\Desktop\EV_Project\Pro_directory
git add analysis/gnn_gpu/ requirements-gnn.txt requirements-step4.txt
git add analysis/*.py analysis/step4_gpu/ analysis/attack_under_PO/
git add "analysis/power_outage/"
git status
git commit -m "Add GNN/LGB step3-8 pipeline, FastAPI map UI, and stratified analysis scripts"
git push origin main
```

## 新建私有仓库（仅 gnn_gpu 子树）

在 GitHub 网页：**New repository → Private → 不要**勾选 README。

```powershell
cd C:\Users\maple\Desktop\EV_Project\Pro_directory\analysis\gnn_gpu
git init
git add gnn/ lgb_fastapi_ui/ step*.py sample*.py build*.py run*.py network_paths.py verify_gnn_upload.py *.md *.txt *.sh
git commit -m "EV charging network: step3-8 LGB/GNN + FastAPI UI"
git branch -M main
git remote add origin https://github.com/YOUR_USER/ev-step4-gpu.git
git push -u origin main
```

推送时密码处填 **Personal Access Token**（Settings → Developer settings → Tokens → `repo`）。
