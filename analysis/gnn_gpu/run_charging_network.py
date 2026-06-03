#!/usr/bin/env python3
"""
充电网络批量分析（对齐参考 charging_network_enhanced）：
  - 读 data/raw 下 AFDC 原始 CSV
  - 按州 0.2km DBSCAN 聚类，每簇 capacity 加和（capacity = 5*Level1 + 25*Level2 + 300*Level3）
  - 转 EPSG:3857，Voronoi 邻接，边权 w_ij = (d_ij/dist_max + ε) / sqrt(c_i_norm * c_j_norm)
  - 攻击模拟，random 10 次
  - 写出 network_characteristics_ALL、attack_results_ALL

用法:
  python analysis/run_charging_network.py "data/raw/alt_fuel_stations_historical_day (Jan 1 2025).csv"
  python analysis/run_charging_network.py --batch-from-raw   # 遍历 data/raw 下所有 Jan 1 YYYY.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN

# 与 charging_network_resume 一致：49 州（48 contiguous + DC），顺序与历史输出一致
STATES_ORDER = (
    "AL AZ AR CO CT DE FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND "
    "OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC CA"
).split()

# 攻击策略与 removal 比例（与现有输出一致）
STRATEGIES = ["random", "degree", "betweenness", "capacity", "closeness"]
REMOVAL_PCTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
N_RANDOM_SIMS = 10   # random 策略 10 次（与参考一致）
EPS_KM = 0.2  # DBSCAN 0.2 km 聚类
MIN_SAMPLES = 2     # 至少 2 点才成簇；未聚上的（噪声）各自算单独节点
EPSILON = 1e-6   # 边权公式防除零


def _to_epsg3857_meters(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(lat, lon) 转 EPSG:3857 坐标 (x, y) 米。无 geopandas 时用近似。"""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(
            geometry=[Point(lon[i], lat[i]) for i in range(len(lat))],
            crs="EPSG:4326",
        )
        gdf = gdf.to_crs("EPSG:3857")
        return np.array([[gdf.geometry.iloc[i].x, gdf.geometry.iloc[i].y] for i in range(len(lat))])
    except Exception:
        # 近似：1 deg lat ≈ 111320 m, 1 deg lon ≈ 111320*cos(lat) m
        lat_rad = np.radians(lat)
        x = np.deg2rad(lon) * 6371000 * np.cos(lat_rad)
        y = np.deg2rad(lat) * 6371000
        return np.column_stack([x, y])


def build_state_network(df: pd.DataFrame, state: str, eps_km: float = EPS_KM, min_samples: int = MIN_SAMPLES):
    """
    单州：使用 200 米聚合算法（避免 DBSCAN 连片）：
    - 在投影坐标下，任选尚未聚合的站点作为簇中心，
    - 将 200 米内的所有站点聚为一个簇，簇中心 = 这些点在经纬度下的均值，
    - 簇内 capacity 相加。

    然后：
    - 将簇中心转 EPSG:3857，
    - 用 Voronoi 邻接（fallback: Delaunay）构建边，
    - 边权按参考公式。

    返回 (G, stats) 或 (None, None)。
    """
    sub = df[df["State"] == state].copy()
    if sub.shape[0] < 3:
        return None, None
    n_raw = len(sub)
    coords_ll = sub[["lat", "lon"]].values.astype(np.float64)  # (n, 2)

    # 在 EPSG:3857（米）下做 200m 聚合，避免 DBSCAN 连片
    coords_m = _to_epsg3857_meters(coords_ll[:, 0], coords_ll[:, 1])
    radius_m = 200.0

    used = np.zeros(len(coords_m), dtype=bool)
    centers = []
    capacities = []

    for i in range(len(coords_m)):
        if used[i]:
            continue
        center_m = coords_m[i]
        dists = np.linalg.norm(coords_m - center_m, axis=1)
        idx = (~used) & (dists <= radius_m)
        idx_indices = np.where(idx)[0]
        pts_ll = coords_ll[idx_indices]

        # 簇中心：经纬度均值
        centers.append(pts_ll.mean(axis=0))

        if "capacity" in sub.columns:
            cap = sub.iloc[idx_indices]["capacity"].sum()
        else:
            cap = float(len(idx_indices)) * 10.0
        capacities.append(cap)

        used[idx_indices] = True

    centers = np.array(centers)
    capacities = np.asarray(capacities, dtype=float)
    n_clusters = len(centers)
    n_noise = 0  # 该聚合算法没有噪声点的概念
    n_nodes = len(centers)
    if n_nodes < 3:
        return None, None
    # 转 EPSG:3857（米）用于邻接和距离
    coords_3857 = _to_epsg3857_meters(centers[:, 0], centers[:, 1])
    # 邻接：Voronoi ridge_points（与参考一致）
    try:
        from scipy.spatial import Voronoi
        vor = Voronoi(coords_3857)
        adj = set()
        for (p1, p2) in vor.ridge_points:
            if p1 < p2:
                adj.add((p1, p2))
            else:
                adj.add((p2, p1))
    except Exception:
        # fallback: Delaunay
        tri = Delaunay(coords_3857)
        adj = set()
        for s in tri.simplices:
            for i in range(3):
                a, b = s[i], s[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                adj.add((a, b))
    # 距离矩阵（米）、dist_max、capacity 归一化
    dist_matrix = np.linalg.norm(coords_3857[:, None] - coords_3857[None, :], axis=2)
    dist_max = float(np.max(dist_matrix))
    if dist_max <= 0:
        dist_max = 1.0
    c_max = float(np.max(capacities)) if np.any(capacities > 0) else 1.0
    cap_norm = np.where(capacities > 0, capacities / c_max, 1.0)
    G = nx.Graph()
    for i in range(len(centers)):
        G.add_node(
            i,
            lat=centers[i, 0],
            lon=centers[i, 1],
            location=(centers[i, 0], centers[i, 1]),
            capacity=capacities[i],
        )
    for (a, b) in adj:
        d_ij = dist_matrix[a, b]
        cap_geo = np.sqrt(cap_norm[a] * cap_norm[b])
        if cap_geo <= 0:
            cap_geo = EPSILON
        w_ij = (d_ij / dist_max + EPSILON) / cap_geo
        G.add_edge(a, b, weight=w_ij)
    stats = {"n_raw": n_raw, "n_clusters": n_clusters, "n_noise": n_noise, "n_nodes": n_nodes}
    return G, stats


def global_efficiency_weighted(G: nx.Graph) -> float:
    """Weighted global efficiency: (1/(n(n-1))) * sum_{i!=j} 1/d(i,j), d = shortest path length (sum of edge weights)."""
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


def attack_and_loss(G: nx.Graph, strategy: str, removal_pct: float, n_sims: int = N_RANDOM_SIMS) -> tuple[float, float | None, float | None]:
    """返回 (efficiency_loss_pct, std 或 None, n_simulations 或 None)。random 策略固定 n_sims=10 与历史一致。"""
    n = G.number_of_nodes()
    if n == 0:
        return 0.0, None, None
    k = max(1, int(round(n * removal_pct)))
    if k >= n:
        return 100.0, None, None
    E0 = global_efficiency_weighted(G)
    if E0 <= 0:
        return 0.0, None, None

    if strategy == "random":
        losses = []
        rng = np.random.default_rng()
        for _ in range(n_sims):
            remove = rng.choice(n, size=k, replace=False)
            G2 = G.copy()
            G2.remove_nodes_from(remove)
            E = global_efficiency_weighted(G2)
            loss = (E0 - E) / E0 * 100.0 if E0 > 0 else 0.0
            losses.append(loss)
        return float(np.mean(losses)), float(np.std(losses)) if len(losses) > 1 else 0.0, float(n_sims)
    else:
        if strategy == "degree":
            deg = dict(G.degree())
            order = sorted(G.nodes(), key=lambda u: -deg.get(u, 0))
        elif strategy == "betweenness":
            b = nx.betweenness_centrality(G, weight="weight")
            order = sorted(G.nodes(), key=lambda u: -b.get(u, 0))
        elif strategy == "capacity":
            cap = nx.get_node_attributes(G, "capacity") or {}
            order = sorted(G.nodes(), key=lambda u: -(cap.get(u, 0)))
        elif strategy == "closeness":
            c = nx.closeness_centrality(G, distance="weight")
            order = sorted(G.nodes(), key=lambda u: -c.get(u, 0))
        else:
            order = list(G.nodes())[:k]
        remove = order[:k]
        G2 = G.copy()
        G2.remove_nodes_from(remove)
        E = global_efficiency_weighted(G2)
        loss = (E0 - E) / E0 * 100.0 if E0 > 0 else 0.0
        return loss, None, None


def network_characteristics_row(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {"num_nodes": 0, "num_edges": 0, "avg_degree": 0, "max_degree": 0, "min_degree": 0,
                "is_connected": False, "num_components": 0, "avg_clustering": 0.0, "global_efficiency": 0.0, "density": 0.0}
    degs = [d for _, d in G.degree()]
    n_comp = nx.number_connected_components(G)
    try:
        clust = nx.average_clustering(G, weight="weight") if G.number_of_edges() > 0 else 0.0
    except Exception:
        clust = 0.0
    eff = global_efficiency_weighted(G)
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0
    return {
        "num_nodes": n,
        "num_edges": m,
        "avg_degree": np.mean(degs),
        "max_degree": int(max(degs)),
        "min_degree": int(min(degs)),
        "is_connected": n_comp == 1,
        "num_components": n_comp,
        "avg_clustering": clust,
        "global_efficiency": eff,
        "density": density,
    }


def _run_complete(run_dir: Path, expected_states: int = 49) -> bool:
    """是否已有完整结果（用于 --batch-from-raw 跳过）。"""
    if not run_dir.is_dir():
        return False
    cand = list(run_dir.glob("attack_results_ALL_*.csv")) or list((run_dir / "attacks").glob("attack_results_ALL_*.csv"))
    if not cand:
        return False
    try:
        with open(cand[0], encoding="utf-8") as f:
            r = csv.DictReader(f)
            states = {row["state"] for row in r}
        return len(states) >= expected_states
    except Exception:
        return False


def _load_existing_partial(run_dir: Path):
    """读取已有部分结果，用于断点续跑。返回 (done_states, nc_df 或 None, att_df 或 None)。"""
    if not run_dir.is_dir():
        return set(), None, None
    att_cand = list(run_dir.glob("attack_results_ALL_*.csv")) or list((run_dir / "attacks").glob("attack_results_ALL_*.csv"))
    nc_cand = list(run_dir.glob("network_characteristics_ALL_*.csv")) or list((run_dir / "networks").glob("network_characteristics_ALL_*.csv"))
    # 选包含州数最多的那份（断点续跑时可能有多份）
    best_att = None
    best_nc = None
    done_states = set()
    if att_cand:
        for p in att_cand:
            try:
                df = pd.read_csv(p, encoding="utf-8")
                if "state" not in df.columns:
                    continue
                s = set(df["state"].unique())
                if len(s) > len(done_states):
                    done_states = s
                    best_att = p
            except Exception:
                continue
    if nc_cand:
        for p in nc_cand:
            try:
                df = pd.read_csv(p, encoding="utf-8")
                if "state" not in df.columns:
                    continue
                if len(df) >= len(done_states):
                    best_nc = p
                    break
            except Exception:
                continue
    nc_df = pd.read_csv(best_nc, encoding="utf-8") if best_nc else None
    att_df = pd.read_csv(best_att, encoding="utf-8") if best_att else None
    return done_states, nc_df, att_df


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # 列名可能带空格
    df.columns = [c.strip() for c in df.columns]
    for col in ["Latitude", "Longitude", "State"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    df = df.rename(columns={"Latitude": "lat", "Longitude": "lon", "State": "State"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "State"])
    # 只保留 EV 充电站
    if "Fuel Type Code" in df.columns:
        df = df[df["Fuel Type Code"].astype(str).str.upper().str.strip() == "ELEC"]
    # capacity: Level1*5 + Level2*25 + DC Fast*300
    l1 = df.get("EV Level1 EVSE Num", pd.Series(0, index=df.index))
    l2 = df.get("EV Level2 EVSE Num", pd.Series(0, index=df.index))
    dc = df.get("EV DC Fast Count", pd.Series(0, index=df.index))
    for x in [l1, l2, dc]:
        if x.dtype == object:
            x = pd.to_numeric(x, errors="coerce").fillna(0)
    df["capacity"] = pd.to_numeric(l1, errors="coerce").fillna(0) * 5 + pd.to_numeric(l2, errors="coerce").fillna(0) * 25 + pd.to_numeric(dc, errors="coerce").fillna(0) * 300
    return df


def run_one_csv(csv_path: Path, out_base: Path, verbose: bool = True, save_network: bool = False) -> bool:
    """对单个原始 CSV 跑全部分析；若该年已有部分结果则只跑缺的州并合并（断点续跑）。"""
    m = re.search(r"Jan 1 (\d{4})\)", csv_path.name)
    if not m:
        if verbose:
            print(f"Skip (no year in name): {csv_path.name}")
        return False
    year = m.group(1)
    run_name = f"run__alt_fuel_stations_historical_day_(Jan_1_{year})"
    run_dir = out_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if save_network:
        (run_dir / "network_structures").mkdir(parents=True, exist_ok=True)

    # 断点续跑：读取已有部分结果，只跑缺的州
    done_states, existing_nc_df, existing_att_df = _load_existing_partial(run_dir)
    states_to_run = [s for s in STATES_ORDER if s not in done_states]
    if not states_to_run:
        if verbose:
            print("{} 已全部完成，跳过。".format(csv_path.name))
        return True
    if verbose:
        print("=" * 80)
        print(f"Run: {csv_path.name} -> {run_dir}")
        if done_states:
            print("  断点续跑：已完成 {} 州，剩余 {} 州".format(len(done_states), len(states_to_run)))
        print("=" * 80)
    df = load_raw_csv(csv_path)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows_nc = []
    rows_att = []
    n_states = len(states_to_run)
    nc_cols = ["state", "num_nodes", "num_edges", "avg_degree", "max_degree", "min_degree", "is_connected", "num_components", "avg_clustering", "global_efficiency", "density"]
    for idx, state in enumerate(states_to_run):
        if verbose:
            print()
            print("[{}/{}]".format(idx + 1, n_states))
            print()
            print("=" * 80)
            print("分析 {}".format(state))
            print("=" * 80)
        G, stats = build_state_network(df, state)
        if G is None:
            if verbose:
                print("  跳过（站点过少或建图失败）")
            continue
        if verbose and stats:
            print("原始站点数: {}".format(stats["n_raw"]))
            print("  Cluster数量: {}".format(stats["n_clusters"]))
            print("  噪声点: {}".format(stats["n_noise"]))
            print("  总节点数: {}".format(stats["n_nodes"]))
            print()
        nc = network_characteristics_row(G)
        nc["state"] = state
        rows_nc.append(nc)
        if verbose:
            n_edges = G.number_of_edges()
            conn = nx.is_connected(G)
            n_comp = nx.number_connected_components(G)
            eff = nc["global_efficiency"]
            print("  💾 保存网络结构: {}".format(state))
            if save_network:
                import pickle
                net_path = run_dir / "network_structures" / f"network_{state}.pkl"
                with open(net_path, "wb") as f:
                    pickle.dump(G, f)
            print("  节点: {}, 边: {}".format(stats["n_nodes"], n_edges))
            print("  连通性: {}".format(conn))
            print("  连通分量: {}".format(n_comp))
            print("  全局效率: {:.4f}".format(eff))
            print()
        for strat in STRATEGIES:
            for rp in REMOVAL_PCTS:
                loss, std, n_sim = attack_and_loss(G, strat, rp)
                rows_att.append({
                    "state": state,
                    "strategy": strat,
                    "removal_pct": rp,
                    "efficiency_loss_pct": loss,
                    "efficiency_loss_std": std if std is not None else "",
                    "n_simulations": n_sim if n_sim is not None else "",
                })
        if verbose:
            print("  ✓ 攻击模拟完成")
    if not rows_nc and not (existing_nc_df is not None and len(existing_nc_df) > 0):
        if verbose:
            print("No state networks built.")
        return False
    # 合并已有 + 本次新跑
    new_nc_df = pd.DataFrame(rows_nc)
    new_att_df = pd.DataFrame(rows_att)
    if existing_nc_df is not None and len(existing_nc_df) > 0:
        existing_nc_df = existing_nc_df[existing_nc_df["state"].isin(done_states)]
        nc_df = pd.concat([existing_nc_df, new_nc_df], ignore_index=True)
    else:
        nc_df = new_nc_df
    if existing_att_df is not None and len(existing_att_df) > 0:
        existing_att_df = existing_att_df[existing_att_df["state"].isin(done_states)]
        att_df = pd.concat([existing_att_df, new_att_df], ignore_index=True)
    else:
        att_df = new_att_df
    # 按州顺序排好（与 STATES_ORDER 一致）
    nc_df["_ord"] = nc_df["state"].apply(lambda s: STATES_ORDER.index(s) if s in STATES_ORDER else 999)
    nc_df = nc_df.sort_values("_ord").drop(columns=["_ord"])
    att_df["_ord"] = att_df["state"].apply(lambda s: STATES_ORDER.index(s) if s in STATES_ORDER else 999)
    att_df = att_df.sort_values(["_ord", "strategy", "removal_pct"]).drop(columns=["_ord"])
    nc_df = nc_df[[c for c in nc_cols if c in nc_df.columns]]
    nc_path = run_dir / f"network_characteristics_ALL_{ts}.csv"
    nc_df.to_csv(nc_path, index=False)
    if verbose:
        print()
        print("  Wrote {}".format(nc_path))
    att_path = run_dir / f"attack_results_ALL_{ts}.csv"
    att_df.to_csv(att_path, index=False)
    if verbose:
        print("  Wrote {}".format(att_path))
    return True


def main():
    parser = argparse.ArgumentParser(description="Charging network batch: raw CSV -> network + attack results")
    parser.add_argument("csv", nargs="?", help="Single CSV path (e.g. data/raw/alt_fuel_stations_historical_day (Jan 1 2025).csv)")
    parser.add_argument("--batch-from-raw", action="store_true", help="Loop over all Jan 1 YYYY CSVs in data/raw, skip completed years")
    parser.add_argument("--only-csvs", type=str, default=None, help="Path to file listing CSV paths to run (one per line); used with --batch-from-raw")
    parser.add_argument("--out", type=str, default="outputs/charging_network", help="Output base dir")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw CSV directory for --batch-from-raw")
    parser.add_argument("--save-network", action="store_true", help="Save per-state network to run_dir/network_structures/")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_base = project_root / args.out
    raw_dir = project_root / args.raw_dir

    if args.csv:
        csv_path = (project_root / args.csv) if not Path(args.csv).is_absolute() else Path(args.csv)
        if not csv_path.exists():
            print(f"File not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
        ok = run_one_csv(csv_path, out_base, save_network=args.save_network)
        sys.exit(0 if ok else 1)

    if args.batch_from_raw:
        if args.only_csvs:
            list_path = (project_root / args.only_csvs) if not Path(args.only_csvs).is_absolute() else Path(args.only_csvs)
            if not list_path.exists():
                print(f"List file not found: {list_path}", file=sys.stderr)
                sys.exit(1)
            with open(list_path) as f:
                csv_paths = [Path(p.strip()) for p in f if p.strip()]
        else:
            csv_paths = []
            for f in raw_dir.iterdir():
                if f.is_file() and f.suffix.lower() == ".csv" and re.search(r"Jan 1 \d{4}\)", f.name):
                    csv_paths.append(f)
            csv_paths.sort(key=lambda p: (re.search(r"Jan 1 (\d{4})\)", p.name) or [None, "0"])[1])
        for csv_path in csv_paths:
            m = re.search(r"Jan 1 (\d{4})\)", csv_path.name)
            if m and _run_complete(out_base / f"run__alt_fuel_stations_historical_day_(Jan_1_{m.group(1)})"):
                print(f"Skip (already complete): {csv_path.name}")
                continue
            run_one_csv(csv_path, out_base, save_network=args.save_network)
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
