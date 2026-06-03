#!/usr/bin/env python3
"""
Build paper-style comparison table (Betweenness R², Random R², Overall R²).

Feature regimes
---------------
局部 (local): 节点属性在窗口内的聚合 — mean/max/std capacity, lat/lon, dcfc, degree 等。
              不含图拓扑 message passing；XGB/MLP 用表格向量。

全局 (global): 图级统计 — n_nodes, density, diameter, clustering, betweenness 等
              (window_global_features.csv 的 15 维)。

GAT: 整图结构 (edge_index + GAT) + 可选 concat 全局向量 (GAT+Global)。

树模型: **LightGBM**（非 XGBoost），与 step7_window_lgb_baseline.py 一致。

Metrics: West test (center_lon <= -100), attacks = betweenness + random (no capacity).
          Always use **best** checkpoint (best_model.pt), not last epoch.

Usage
-----
  # 示例表（dummy）
  PYTHONPATH=. python step7_build_comparison_table.py --dummy

  # 从已有 checkpoint / report 填表（不重新训练）
  PYTHONPATH=. python step7_build_comparison_table.py --from-artifacts

  # 训练缺失的 LightGBM(local) 并评估所有提供的 checkpoint
  PYTHONPATH=. python step7_build_comparison_table.py --run-missing --data-dir .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from step7_window_lgb_baseline import make_model

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnn.constants import ATTACK_ORDER, parse_attacks, y_cols_for_attacks
from gnn.global_graph_features import GLOBAL_FEATURE_NAMES, subgraph_path_for_window
from gnn.local_window_features import LOCAL_FEATURE_NAMES, build_local_feature_table

ATTACKS = parse_attacks("no-capacity")  # betweenness + random

DUMMY_ROWS = [
    {
        "model": "MLP",
        "feature_regime": "局部特征（节点聚合，无图结构）",
        "betweenness_r2": None,
        "random_r2": None,
        "overall_r2": None,
    },
    {
        "model": "LightGBM alone",
        "feature_regime": "局部特征（节点 mean/max/std 聚合）",
        "betweenness_r2": None,
        "random_r2": None,
        "overall_r2": None,
    },
    {
        "model": "LightGBM + Global",
        "feature_regime": "局部聚合 + 全局图统计（15 维）",
        "betweenness_r2": None,
        "random_r2": None,
        "overall_r2": None,
    },
    {
        "model": "GAT alone",
        "feature_regime": "图结构（message passing，无全局 concat）",
        "betweenness_r2": 0.478,
        "random_r2": 0.748,
        "overall_r2": 0.52,
    },
    {
        "model": "GAT + Global",
        "feature_regime": "图结构 + 全局特征 concat",
        "betweenness_r2": 0.593,
        "random_r2": 0.880,
        "overall_r2": 0.611,
    },
]


def load_labels_and_meta(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(data_dir / "windows_meta.csv")
    labels_path = data_dir / "labels" / "y_labels.parquet"
    if not labels_path.exists():
        labels_path = data_dir / "labels" / "y_labels.csv"
    labels = pd.read_parquet(labels_path) if labels_path.suffix == ".parquet" else pd.read_csv(labels_path)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    if "subgraph_path" not in labels.columns and "subgraph_path" in manifest.columns:
        labels = labels.merge(manifest[["window_id", "subgraph_path"]], on="window_id", how="left")
    meta_cols = [c for c in ("window_id", "center_lon", "center_lat") if c in meta.columns]
    if len(meta_cols) > 1:
        labels = labels.merge(meta[meta_cols], on="window_id", how="left")
    return labels, meta


def west_masks(merged: pd.DataFrame, lon_threshold: float = -100.0) -> tuple[np.ndarray, np.ndarray]:
    lon = merged["center_lon"].astype(float).values
    train = lon > lon_threshold
    test = ~train
    return train, test


def metrics_from_vectors(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from gnn.attack_targets import select_attacks_numpy
    from gnn.metrics import metrics_by_attack

    yt = select_attacks_numpy(y_true, ATTACKS)
    yp = select_attacks_numpy(y_pred, ATTACKS)
    k = yt.shape[1] // len(ATTACKS)
    by = metrics_by_attack(yt, yp, attacks=ATTACKS, n_points=k)
    return {
        "betweenness_r2": float(by["betweenness"]["r2"]),
        "random_r2": float(by["random"]["r2"]),
        "overall_r2": float(r2_score(yt.ravel(), yp.ravel())),
        "overall_mae": float(mean_absolute_error(yt.ravel(), yp.ravel())),
    }


def train_lgbm_table(
    merged: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    n_estimators: int = 200,
    max_depth: int = 8,
    num_leaves: int = 31,
) -> dict[str, float]:
    X = merged[x_cols].astype(float).values
    Y = merged[y_cols].astype(float).values
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_mask])
    X_te = scaler.transform(X[test_mask])
    X_tr_df = pd.DataFrame(X_tr, columns=x_cols)
    X_te_df = pd.DataFrame(X_te, columns=x_cols)
    model = make_model("lightgbm", n_estimators, max_depth, num_leaves, n_jobs=-1)
    model.fit(X_tr_df, Y[train_mask])
    pred = model.predict(X_te_df)
    return metrics_from_vectors(Y[test_mask], pred)


def eval_gnn_checkpoint(
    ckpt_path: Path,
    dataset_path: Path,
    data_dir: Path,
    global_csv: Path | None,
    device: str = "cuda",
) -> dict[str, float]:
    import torch
    from torch_geometric.loader import DataLoader

    from gnn.edge_utils import sanitize_graph_list
    from gnn.global_graph_features import attach_global_features_from_csv
    from gnn.metrics import metrics_by_attack, curve_metrics
    from gnn.models import ResilienceGNN
    from gnn.train_utils import match_target_to_pred

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    attacks = tuple(ckpt.get("attacks", ATTACKS))
    try:
        bundle = torch.load(dataset_path, map_location="cpu", weights_only=False)
    except TypeError:
        bundle = torch.load(dataset_path, map_location="cpu")
    graphs = bundle["graphs"]
    sanitize_graph_list(graphs, show_progress=False)
    split_masks = bundle.get("split_masks", {})
    train_mask = split_masks.get("east_west_train", np.ones(len(graphs), bool))
    test_mask = split_masks.get("east_west_test", np.zeros(len(graphs), bool))

    global_dim = int(ckpt.get("global_dim", 0))
    if global_dim > 0 and not hasattr(graphs[0], "gf") and global_csv and global_csv.is_file():
        attach_global_features_from_csv(graphs, global_csv, train_mask, verbose=False)

    model = ResilienceGNN(
        in_channels=4,
        hidden_channels=int(ckpt.get("hidden", 32)),
        gat_heads=int(ckpt.get("gat_heads", 4)),
        edge_dim=1 if ckpt.get("model_type", "gat") == "gat" else 0,
        model_type=ckpt.get("model_type", "gat"),
        n_out_points=int(ckpt.get("curve_points", 10)),
        global_dim=global_dim,
        attacks=attacks,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    test_graphs = [graphs[i] for i in range(len(graphs)) if test_mask[i]]
    loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dev)
            pred, _, _, _ = model(batch)
            tgt = batch.y
            if tgt.dim() == 1:
                tgt = tgt.view(pred.size(0), -1)
            tgt = match_target_to_pred(tgt, pred, attacks)
            ys.append(tgt.cpu().numpy())
            ps.append(pred.cpu().numpy())
    y_true = np.vstack(ys)
    y_pred = np.vstack(ps)
    k = y_true.shape[1] // len(attacks)
    by = metrics_by_attack(y_true, y_pred, attacks=attacks, n_points=k)
    return {
        "betweenness_r2": float(by["betweenness"]["r2"]),
        "random_r2": float(by["random"]["r2"]),
        "overall_r2": float(curve_metrics(y_true, y_pred, n_points=k)["r2"]),
        "overall_mae": float(curve_metrics(y_true, y_pred, n_points=k)["mae"]),
        "best_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint": str(ckpt_path),
    }


def metrics_from_report_json(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    rep = json.loads(path.read_text(encoding="utf-8"))
    if "west_test_r2" in rep:
        overall = float(rep["west_test_r2"])
    else:
        overall = float(rep.get("test_r2", np.nan))
    out = {
        "overall_r2": overall,
        "overall_mae": float(rep.get("west_test_mae", rep.get("test_mae", np.nan))),
        "betweenness_r2": np.nan,
        "random_r2": np.nan,
    }
    tm = rep.get("test_metrics")
    if isinstance(tm, dict):
        rep = {**rep, **tm}
    for head, key in (("betweenness", "betweenness_r2"), ("random", "random_r2"), ("overall", "overall_r2")):
        block = rep.get(head)
        if isinstance(block, dict) and "r2" in block:
            out[key] = float(block["r2"])
    if isinstance(rep.get("overall"), dict) and "r2" in rep["overall"]:
        out["overall_r2"] = float(rep["overall"]["r2"])
    if "test_betweenness_r2" in rep:
        out["betweenness_r2"] = float(rep["test_betweenness_r2"])
    if "test_random_r2" in rep:
        out["random_r2"] = float(rep["test_random_r2"])
    return out


def apply_report_to_row(rows: list[dict], report_path: Path, model_name: str) -> None:
    met = metrics_from_report_json(report_path)
    if not met:
        return
    try:
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        regime = {
            "local": "局部特征（节点 mean/max/std 聚合）",
            "global": "全局图统计（15 维）",
            "both": "局部聚合 + 全局图统计（15 维）",
        }.get(str(rep.get("feature_set", "")), None)
    except Exception:
        regime = None
    for i, r in enumerate(rows):
        if r["model"] == model_name:
            rows[i] = {
                **r,
                "betweenness_r2": met.get("betweenness_r2"),
                "random_r2": met.get("random_r2"),
                "overall_r2": met.get("overall_r2"),
            }
            if regime:
                rows[i]["feature_regime"] = regime


def format_table(df: pd.DataFrame) -> str:
    cols = ["model", "feature_regime", "betweenness_r2", "random_r2", "overall_r2"]

    def fmt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):.3f}"

    lines = [
        "| Model | 特征 | Betweenness R² | Random R² | Overall R² |",
        "|-------|------|----------------|-----------|------------|",
    ]
    for _, r in df[cols].iterrows():
        lines.append(
            f"| {r['model']} | {r['feature_regime']} | "
            f"{fmt(r['betweenness_r2'])} | {fmt(r['random_r2'])} | {fmt(r['overall_r2'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model comparison table (BR attacks only)")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--dummy", action="store_true", help="Print placeholder table only")
    parser.add_argument("--from-artifacts", action="store_true", help="Load metrics from JSON/checkpoints")
    parser.add_argument("--run-missing", action="store_true", help="Train LightGBM(local/both) if needed; eval GNN ckpts")
    parser.add_argument("--out", default="results_gnn/comparison_table")
    parser.add_argument("--dataset", default="pyg/pyg_dataset.pt")
    parser.add_argument("--global-csv", default="results_gnn/lgb_baseline/window_global_features.csv")
    parser.add_argument("--gat-plain", default="pyg/train_gat_br/best_model.pt")
    parser.add_argument("--gat-global", default="pyg/train_gat_global_br/best_model.pt")
    parser.add_argument("--mlp", default="pyg/train_mlp/best_model.pt")
    parser.add_argument(
        "--lgb-local-report",
        default="results_gnn/lgb_local_br/lgb_report.json",
        help="LightGBM alone (feature-set local)",
    )
    parser.add_argument(
        "--lgb-global-report",
        default="results_gnn/lgb_global_br/lgb_report.json",
        help="LightGBM 15-dim global features",
    )
    parser.add_argument(
        "--lgb-both-report",
        default="results_gnn/lgb_both_br/lgb_report.json",
        help="LightGBM local+global (preferred for +Global row)",
    )
    parser.add_argument("--lgb-n-estimators", type=int, default=200)
    parser.add_argument("--lgb-max-depth", type=int, default=8)
    parser.add_argument("--lgb-num-leaves", type=int, default=31)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-lgb",
        action="store_true",
        help="Only evaluate GAT/MLP checkpoints; skip LightGBM local feature build/train",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dummy:
        df = pd.DataFrame(DUMMY_ROWS)
        md = format_table(df)
        print(md)
        df.to_csv(out_dir / "comparison_dummy.csv", index=False)
        (out_dir / "comparison_dummy.md").write_text(md + "\n", encoding="utf-8")
        print(f"\nSaved -> {out_dir}/comparison_dummy.csv")
        return

    data_dir = Path(args.data_dir).resolve()
    if str(args.data_dir) in (".", "./"):
        data_dir = Path.cwd().resolve()

    rows = [dict(r) for r in DUMMY_ROWS]

    if args.from_artifacts or args.run_missing:
        labels, wmeta = load_labels_and_meta(data_dir)
        y_cols = [c for c in y_cols_for_attacks(ATTACKS) if c in labels.columns]
        merged = labels.dropna(subset=["center_lon"] + y_cols, how="any")

        local_df = None
        if args.run_missing and not args.skip_lgb:
            local_csv = out_dir / "window_local_features.csv"
            local_df = build_local_feature_table(
                merged["window_id"].astype(str).tolist(),
                data_dir,
                cache_csv=local_csv,
                windows_meta=wmeta,
            )

        global_csv = Path(args.global_csv)
        if not global_csv.is_file():
            for cand in (
                data_dir / "results_gnn/lgb_baseline/window_global_features.csv",
                data_dir / "results_gnn/xgb_baseline/window_global_features.csv",
            ):
                if cand.is_file():
                    global_csv = cand
                    break
        global_df = pd.read_csv(global_csv) if global_csv.is_file() else None

        def _merge_feats(extra: pd.DataFrame) -> pd.DataFrame:
            overlap = [c for c in extra.columns if c in merged.columns and c != "window_id"]
            return merged.merge(extra.drop(columns=overlap, errors="ignore"), on="window_id", how="inner")

        if args.run_missing and not args.skip_lgb:
            m_local = _merge_feats(local_df)
            tr_m, te_m = west_masks(m_local)
            x_local = [c for c in LOCAL_FEATURE_NAMES if c in m_local.columns]
            lgb_kw = dict(
                n_estimators=args.lgb_n_estimators,
                max_depth=args.lgb_max_depth,
                num_leaves=args.lgb_num_leaves,
            )
            met = train_lgbm_table(m_local, x_local, y_cols, tr_m, te_m, **lgb_kw)
            for i, r in enumerate(rows):
                if r["model"] == "LightGBM alone":
                    rows[i] = {**r, **met}

            if global_df is not None:
                m_both = _merge_feats(local_df).merge(
                    global_df, on="window_id", how="inner", suffixes=("", "_g")
                )
                x_both = [c for c in LOCAL_FEATURE_NAMES if c in m_both.columns] + [
                    c for c in GLOBAL_FEATURE_NAMES if c in m_both.columns
                ]
                tr_m2, te_m2 = west_masks(m_both)
                met = train_lgbm_table(m_both, x_both, y_cols, tr_m2, te_m2, **lgb_kw)
                for i, r in enumerate(rows):
                    if r["model"] == "LightGBM + Global":
                        rows[i] = {**r, **met}

        # LightGBM rows from saved reports (no re-train)
        apply_report_to_row(rows, Path(args.lgb_local_report), "LightGBM alone")
        both_p = Path(args.lgb_both_report)
        if both_p.is_file():
            apply_report_to_row(rows, both_p, "LightGBM + Global")
        else:
            apply_report_to_row(rows, Path(args.lgb_global_report), "LightGBM + Global")

        ds = Path(args.dataset).resolve()
        gcsv = global_csv if global_csv.is_file() else None
        eval_path = data_dir / "results_gnn/eval_gat_global_br/test_metrics.json"
        eval_gat = metrics_from_report_json(eval_path)
        if eval_gat:
            for i, r in enumerate(rows):
                if r["model"] == "GAT + Global":
                    rows[i] = {**r, **{k: eval_gat[k] for k in ("overall_r2", "betweenness_r2", "random_r2") if k in eval_gat}}

        ckpt_map = {
            "GAT alone": Path(args.gat_plain),
            "GAT + Global": Path(args.gat_global),
            "MLP": Path(args.mlp),
        }
        for i, r in enumerate(rows):
            name = r["model"]
            if name not in ckpt_map:
                continue
            if name == "GAT + Global" and eval_gat and not np.isnan(
                float(rows[i].get("overall_r2") or eval_gat.get("overall_r2", np.nan))
            ):
                print(f"[ok] {name} from eval_gat_global_br/test_metrics.json (skip re-eval)", flush=True)
                continue
            ck = ckpt_map[name]
            if not ck.is_file():
                print(f"[skip] {name}: no checkpoint {ck}", flush=True)
                continue
            try:
                met = eval_gnn_checkpoint(ck, ds, data_dir, gcsv, device=args.device)
                rows[i] = {**r, **met}
                print(f"[ok] {name} epoch={met.get('best_epoch')} overall R²={met['overall_r2']:.4f}", flush=True)
            except Exception as e:
                print(f"[fail] {name}: {e}", flush=True)

        # step7 eval json overrides if present
        for name, sub in (
            ("GAT alone", "eval_gat_plain"),
            ("GAT + Global", "eval_gat_global"),
            ("MLP", "eval_mlp"),
        ):
            p = out_dir.parent / sub / "test_metrics.json"
            if not p.is_file():
                p = data_dir / "results_gnn" / sub / "test_metrics.json"
            met = metrics_from_report_json(p)
            if met:
                for i, r in enumerate(rows):
                    if r["model"] == name:
                        rows[i] = {**r, **{k: met.get(k, r.get(k)) for k in met}}

    df = pd.DataFrame(rows)
    md = format_table(df)
    print("\n" + md)
    df.to_csv(out_dir / "comparison_table.csv", index=False)
    (out_dir / "comparison_table.md").write_text(md + "\n", encoding="utf-8")
    with open(out_dir / "comparison_table.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved -> {out_dir}/comparison_table.csv")


if __name__ == "__main__":
    main()
