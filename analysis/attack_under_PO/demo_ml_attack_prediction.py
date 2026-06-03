#!/usr/bin/env python3
"""
Minimal ML demo for hypernode vulnerability prediction.

What this demo does
-------------------
1. Load one state network pickle.
2. Extract per-hypernode features.
3. Build a ground-truth label for each node:
      single-node performance loss
   defined as the relative drop in weighted global efficiency after removing
   that node alone.
4. Train a simple regression model (RandomForestRegressor).
5. Evaluate ML on a train/test split.
6. Compare actual vs ML-predicted attack curves under:
   - betweenness-centrality attack order
   - capacity attack order

Important
---------
This is a minimal demonstration, not the final production pipeline.
The attack-curve prediction here uses:
  predicted cumulative loss = cumulative sum of predicted single-node losses
along a chosen attack order.
This is a simple proxy to visualize whether ML captures vulnerable nodes.

Fast demo options
-----------------
- --max-nodes: sample a smaller induced subgraph for a quicker demo
- --max-removal-frac: only evaluate the first part of the attack curve
- --curve-step: evaluate every N-th removal point instead of every point
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from extract_hypernode_features import extract_features, parse_state_abbr_from_path, parse_year_from_path
from compute_impact_radius import STATE_NAME_TO_ABBR


ABBR_TO_STATE_NAME = {abbr: name for name, abbr in STATE_NAME_TO_ABBR.items()}


def load_network(data_path: Path) -> nx.Graph:
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data["network"]


def global_efficiency_weighted(G: nx.Graph) -> float:
    """Weighted global efficiency using shortest path lengths with edge weight."""
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    try:
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        total = 0.0
        count = 0
        nodes = list(G.nodes())
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                d = lengths.get(i, {}).get(j, float("inf"))
                if np.isfinite(d) and d > 1e-12:
                    total += 1.0 / d
                count += 1
        return total / count if count else 0.0
    except Exception:
        return 0.0


def single_node_loss_labels(G: nx.Graph) -> pd.DataFrame:
    """
    For each node, compute relative performance loss after removing that node alone.
    Label is:
        y = (E0 - E_without_node) / E0
    """
    E0 = global_efficiency_weighted(G)
    rows = []
    for node in G.nodes():
        G2 = G.copy()
        G2.remove_node(node)
        E = global_efficiency_weighted(G2)
        loss = (E0 - E) / E0 if E0 > 0 else 0.0
        if loss < 0:
            loss = 0.0
        rows.append({"node_id": node, "y_eff_drop": float(loss)})
    return pd.DataFrame(rows)


def actual_attack_curve(
    G: nx.Graph,
    order: list[int],
    *,
    max_removal_frac: float = 1.0,
    curve_step: int = 1,
) -> pd.DataFrame:
    """
    Exact cumulative attack curve for a given removal order.
    """
    E0 = global_efficiency_weighted(G)
    records = []
    max_k = min(len(order) - 1, max(1, int(np.floor(len(order) * max_removal_frac))))
    for k in range(1, max_k + 1, max(1, curve_step)):
        remove = order[:k]
        G2 = G.copy()
        G2.remove_nodes_from(remove)
        E = global_efficiency_weighted(G2)
        loss = (E0 - E) / E0 if E0 > 0 else 0.0
        if loss < 0:
            loss = 0.0
        records.append(
            {
                "k_removed": k,
                "removal_frac": k / G.number_of_nodes(),
                "actual_cum_loss": float(loss),
            }
        )
    return pd.DataFrame(records)


def predicted_attack_curve(
    df_nodes: pd.DataFrame,
    order: list[int],
    *,
    max_removal_frac: float = 1.0,
    curve_step: int = 1,
) -> pd.DataFrame:
    """
    Proxy cumulative attack curve using predicted single-node losses summed
    along the chosen attack order.
    """
    pred_map = dict(zip(df_nodes["node_id"], df_nodes["y_pred"]))
    cum = 0.0
    records = []
    max_k = min(len(order) - 1, max(1, int(np.floor(len(order) * max_removal_frac))))
    for k in range(1, max_k + 1):
        node = order[k - 1]
        cum += max(float(pred_map.get(node, 0.0)), 0.0)
        if k % max(1, curve_step) != 0 and k != 1 and k != max_k:
            continue
        records.append(
            {
                "k_removed": k,
                "pred_cum_loss": cum,
            }
        )
    return pd.DataFrame(records)


def build_attack_order(G: nx.Graph, strategy: str) -> list[int]:
    if strategy == "betweenness":
        b = nx.betweenness_centrality(G, weight="weight")
        return sorted(G.nodes(), key=lambda u: -b.get(u, 0.0))
    if strategy == "capacity":
        cap = nx.get_node_attributes(G, "capacity")
        return sorted(G.nodes(), key=lambda u: -cap.get(u, 0.0))
    raise ValueError(f"Unsupported strategy: {strategy}")


def make_plot(curves: dict[str, pd.DataFrame], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    strategies = ["betweenness", "capacity"]

    for ax, strategy in zip(axes, strategies):
        df = curves[strategy]
        ax.plot(df["removal_frac"], df["actual_cum_loss"], label="Actual", linewidth=2)
        ax.plot(df["removal_frac"], df["pred_cum_loss"], label="Predicted (ML proxy)", linewidth=2, linestyle="--")
        ax.set_title(strategy)
        ax.set_xlabel("Removal fraction")
        ax.set_ylabel("Cumulative performance loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_prediction_scatter(y_true: pd.Series, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.7)
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.5, color="black")
    ax.set_xlabel("Actual loss")
    ax.set_ylabel("Predicted loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_shap_outputs(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    output_dir: Path,
    state_abbr: str,
) -> tuple[Path | None, Path | None]:
    """
    Save SHAP summary plot and SHAP mean(|value|) importance CSV.
    Returns (plot_path, csv_path). If shap is unavailable, returns (None, None).
    """
    try:
        import shap
    except Exception:
        return None, None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_path = output_dir / f"{state_abbr}_shap_importance_demo.csv"
    importance_df.to_csv(importance_path, index=False)

    plot_path = output_dir / f"{state_abbr}_shap_summary_demo.png"
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plot_path, importance_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal ML demo for attack performance loss prediction.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to state network pickle, e.g. outputs/.../network_NY.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_ml_demo",
        help="Directory to save demo outputs.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test fraction for simple train/test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Optional: sample at most this many nodes for a faster demo.",
    )
    parser.add_argument(
        "--max-removal-frac",
        type=float,
        default=0.3,
        help="Only evaluate the first part of the attack curve (default: 0.3).",
    )
    parser.add_argument(
        "--curve-step",
        type=int,
        default=5,
        help="Evaluate every N-th removal point on the attack curve (default: 5).",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Network pickle not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_abbr = parse_state_abbr_from_path(data_path)
    state_name = ABBR_TO_STATE_NAME.get(state_abbr) if state_abbr else None
    year = parse_year_from_path(data_path)

    G = load_network(data_path)
    if args.max_nodes is not None and G.number_of_nodes() > args.max_nodes:
        rng = np.random.default_rng(args.random_state)
        sampled_nodes = rng.choice(list(G.nodes()), size=args.max_nodes, replace=False)
        G = G.subgraph(sampled_nodes).copy()

    feat_df = extract_features(G, state_abbr=state_abbr, state_name=state_name, year=year)
    label_df = single_node_loss_labels(G)
    df = feat_df.merge(label_df, on="node_id", how="inner")

    feature_cols = [
        "capacity",
        "degree",
        "betweenness_centrality",
        "clustering_coefficient",
        "nearest_neighbor_distance_km",
        "neighbor_mean_capacity",
        "neighbor_mean_degree",
        "neighbor_capacity_gini",
        "state_n_nodes",
        "state_mean_capacity",
        "state_capacity_gini",
        "year",
    ]

    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df["y_eff_drop"].copy()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=args.test_size, random_state=args.random_state
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=args.random_state,
        n_jobs=-1,
        min_samples_leaf=2,
        criterion="squared_error",
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Refit on full data for a cleaner demo attack-curve prediction
    model.fit(X, y)
    df["y_pred"] = model.predict(X)
    df["y_pred"] = df["y_pred"].clip(lower=0.0)

    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    curves = {}
    for strategy in ("betweenness", "capacity"):
        order = build_attack_order(G, strategy)
        df_actual = actual_attack_curve(
            G,
            order,
            max_removal_frac=args.max_removal_frac,
            curve_step=args.curve_step,
        )
        df_pred = predicted_attack_curve(
            df,
            order,
            max_removal_frac=args.max_removal_frac,
            curve_step=args.curve_step,
        )
        df_curve = df_actual.merge(df_pred, on="k_removed", how="left")
        curves[strategy] = df_curve
        df_curve.to_csv(output_dir / f"{state_abbr}_{strategy}_attack_curve_demo.csv", index=False)

    nodes_out = output_dir / f"{state_abbr}_node_ml_demo.csv"
    df.to_csv(nodes_out, index=False)
    feature_importance.to_csv(output_dir / f"{state_abbr}_feature_importance_demo.csv", index=False)
    model_path = output_dir / f"{state_abbr}_random_forest_model.joblib"
    joblib.dump(model, model_path)

    plot_path = output_dir / f"{state_abbr}_ml_demo_attack_curves.png"
    make_plot(
        curves,
        plot_path,
        title=f"{state_name or state_abbr} ML demo: actual vs predicted attack curves",
    )
    scatter_path = output_dir / f"{state_abbr}_actual_vs_pred_demo.png"
    make_prediction_scatter(
        y_test,
        y_pred_test,
        scatter_path,
        title=f"{state_name or state_abbr} test set: actual vs predicted",
    )
    shap_plot_path, shap_csv_path = save_shap_outputs(model, X, output_dir, state_abbr)

    metrics = pd.DataFrame(
        [
            {
                "state_abbr": state_abbr,
                "state_name": state_name,
                "year": year,
                "n_nodes": G.number_of_nodes(),
                "max_nodes_used": args.max_nodes if args.max_nodes is not None else G.number_of_nodes(),
                "max_removal_frac": args.max_removal_frac,
                "curve_step": args.curve_step,
                "model_type": "RandomForestRegressor",
                "training_objective": "squared_error (MSE split criterion; no epoch-wise loss curve)",
                "train_MAE": train_mae,
                "test_MAE": test_mae,
                "train_RMSE": train_rmse,
                "test_RMSE": test_rmse,
                "train_R2": train_r2,
                "test_R2": test_r2,
            }
        ]
    )
    metrics.to_csv(output_dir / f"{state_abbr}_ml_demo_metrics.csv", index=False)

    print("Saved demo outputs:")
    print(f"  Node table: {nodes_out}")
    print(f"  Metrics: {output_dir / f'{state_abbr}_ml_demo_metrics.csv'}")
    print(f"  Feature importance: {output_dir / f'{state_abbr}_feature_importance_demo.csv'}")
    print(f"  Model: {model_path}")
    print(f"  Plot: {plot_path}")
    print(f"  Prediction scatter: {scatter_path}")
    if shap_plot_path is not None:
        print(f"  SHAP plot: {shap_plot_path}")
    if shap_csv_path is not None:
        print(f"  SHAP importance: {shap_csv_path}")
    print("\nTest metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
