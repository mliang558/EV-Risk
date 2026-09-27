#!/usr/bin/env python3
"""
Utilities for sampling outage epicenters using existing charging stations.

Idea / 依据
----------
- Outage 发生的位置，物理上更可能出现在“有电网设施的地方”（例如变电站、
  配电线路附近），而不是县面上任意一点。
- 对于 EV network，我们至少知道所有 station 的坐标，因此可以直接
  “在该县的 station 里随机选一个作为 epicenter”，保证每次攻击都打到站。

用法示意
--------
假设 G 的每个节点有属性:
    - 'location' = (lat, lon)  或
    - 'lat', 'lon'
且每个节点还有 'county_fips'（五位字符串），则:

from select_epicenter_by_stations import choose_station_epicenter_in_county

epic = choose_station_epicenter_in_county(G, "06037")
if epic is not None:
    epic_lat, epic_lon = epic
    # 用这个 epicenter + radius_km 去做攻击
"""

from __future__ import annotations

from typing import Optional, Tuple, Iterable

import numpy as np
import networkx as nx


def _get_node_location(node_data) -> Optional[Tuple[float, float]]:
    """
    从节点属性中提取 (lat, lon)。
    支持:
    - 'location' = (lat, lon)
    - 'lat' and 'lon' 两个字段
    """
    loc = node_data.get("location")
    if loc is not None and isinstance(loc, (tuple, list)) and len(loc) == 2:
        return float(loc[0]), float(loc[1])
    if "lat" in node_data and "lon" in node_data:
        return float(node_data["lat"]), float(node_data["lon"])
    return None


def _filter_nodes_in_county(
    G: nx.Graph,
    county_fips: str,
    county_attr: str = "county_fips",
) -> Iterable[int]:
    """
    在图中筛选属于指定县的节点。

    参数
    ----
    G : nx.Graph
        EV charging network.
    county_fips : str
        五位 FIPS，例如 "06037"。会自动 zero-pad。
    county_attr : str, default "county_fips"
        节点属性里保存 county FIPS 的字段名。
    """
    target = str(county_fips).zfill(5)
    for node in G.nodes:
        data = G.nodes[node]
        fips = data.get(county_attr)
        if fips is None:
            continue
        if str(fips).zfill(5) == target:
            yield node


def choose_station_epicenter_in_county(
    G: nx.Graph,
    county_fips: str,
    county_attr: str = "county_fips",
    weight_attr: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
    *,
    exclude_cf_added: bool = True,
) -> Optional[Tuple[float, float]]:
    """
    在指定县内的 station 节点中随机选取一个作为 outage epicenter。

    参数
    ----
    G : nx.Graph
        EV charging network，节点上至少要有坐标信息:
        - 'location' = (lat, lon)  或
        - 'lat', 'lon'
        同时建议有 county FIPS 属性（默认 'county_fips'）。
    county_fips : str
        五位 FIPS，例如 "06037"。
    county_attr : str, default "county_fips"
        节点属性里保存 county FIPS 的字段名。
    weight_attr : str, optional
        若指定，则按该字段的值作为权重随机抽样 station，
        例如 'power_kw', 'capacity', 'degree' 等；
        否则在该县所有 station 中均匀随机选一个。
    rng : np.random.Generator, optional
        随机数生成器；若为 None，则使用 default_rng()。
    exclude_cf_added : bool, default True
        Skip counterfactual coverage-expansion nodes (``is_cf_added`` /
        ``coverage_expansion``). CRN epicenters must come from the baseline
        network; new nodes may only be covered by R_c, never chosen as epicenter.

    返回
    ----
    epicenter : (lat, lon) 或 None
        若该县中没有任何带坐标的 station，则返回 None。
    """
    if rng is None:
        rng = np.random.default_rng()

    # 筛选目标县的节点
    candidate_nodes = list(_filter_nodes_in_county(G, county_fips, county_attr))
    if not candidate_nodes:
        return None

    # 过滤掉没有坐标的节点（以及 CF 加站节点，避免破坏 CRN）
    nodes_with_loc = []
    weights = []
    for node in candidate_nodes:
        data = G.nodes[node]
        if exclude_cf_added and (
            data.get("is_cf_added") or data.get("coverage_expansion")
        ):
            continue
        loc = _get_node_location(data)
        if loc is None:
            continue
        nodes_with_loc.append((node, loc))
        if weight_attr is not None:
            w = float(data.get(weight_attr, 0.0))
        else:
            w = 1.0
        weights.append(max(w, 0.0))

    if not nodes_with_loc:
        return None

    # 构造概率分布（若全为 0 则退回均匀）
    weights_arr = np.asarray(weights, dtype=float)
    total_w = weights_arr.sum()
    if total_w > 0:
        probs = weights_arr / total_w
    else:
        probs = None  # 默认均匀

    idx_choices = np.arange(len(nodes_with_loc))
    chosen_idx = rng.choice(idx_choices, p=probs)
    _, loc = nodes_with_loc[chosen_idx]
    return loc[0], loc[1]


__all__ = [
    "choose_station_epicenter_in_county",
]

