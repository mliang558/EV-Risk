#!/usr/bin/env python3
"""
Utilities for sampling outage epicenters using population density.

Motivation / 依据
----------------
- 真实世界里，大规模停电事件的“中心”更可能出现在人口和负荷集中的区域
  （例如变电站附近、城市中心），而不是县内任意一点都等概率。
- 如果我们已经有 finer-scale 的人口空间数据（census block / tract 等），
  可以用人口作为权重来选 epicenter，更贴近“人在哪儿、电网风险在哪儿”。

核心思想
--------
1. 把县内划分成若干人口单元（例如 census block，多边形），每个单元有：
   - geometry : 多边形
   - population : 该单元人口（或加权负荷）
   - county_fips : 所在县 FIPS（与 outage 事件的 county fips 对应）
2. 对于一次 outage 事件，在对应县的所有单元中，按 population 作为权重做加权抽样：
   - 先随机选中一个人口单元（概率 ∝ population）
   - 然后在该多边形内部均匀抽样一个点，作为本次 outage 的 epicenter

这样做后：
- 县内人口高的区域更容易成为 outage epicenter；
- 仍然保持 Monte Carlo 的随机性，只是有了“人越多越容易被打”的物理依据。

注意：本文件不直接修改主仿真脚本，只提供通用函数，方便在
`monte_carlo_outage_attack.py` 或其他脚本中按需调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import geopandas as gpd
from shapely.geometry import Point


@dataclass
class PopulationUnitSchema:
    """
    列名约定（你的人口数据只要能映射成这个 schema 就可以用）：

    - county_fips_col : 县级 FIPS，字符串形式，长度 5，例如 "06037"
    - population_col  : 该空间单元人口数（或其他负荷 proxy）
    - geometry_col    : 多边形 geometry
    """

    county_fips_col: str = "county_fips"
    population_col: str = "population"
    geometry_col: str = "geometry"


def sample_point_in_polygon(poly, rng: np.random.Generator) -> Point:
    """
    Uniform-ish random point inside polygon via rejection sampling.
    和项目里其他地方用的一致，只是单独放在这里复用。
    """
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(100):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if poly.contains(p):
            return p
    # Fallback to representative point if rejection fails
    return poly.representative_point()


def load_population_units(path: str) -> gpd.GeoDataFrame:
    """
    读取人口空间数据（census block / tract 等）。

    参数
    ----
    path : str
        支持任意 GeoPandas 能读的格式（shapefile, GeoJSON, GeoPackage 等）。

    返回
    ----
    gdf : GeoDataFrame
        包含至少 geometry 列。其他列名由调用者与 PopulationUnitSchema 协调。
    """
    gdf = gpd.read_file(path)
    if "geometry" not in gdf.columns:
        raise ValueError("Population GeoDataFrame must contain a 'geometry' column.")
    return gdf


def choose_population_weighted_epicenter(
    county_fips: str,
    pop_gdf: gpd.GeoDataFrame,
    schema: Optional[PopulationUnitSchema] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Point]:
    """
    在某个县内按人口密度（人口权重）选择 outage epicenter。

    参数
    ----
    county_fips : str
        五位县级 FIPS，比如 "06037"。会自动 zero-pad 到 5 位。
    pop_gdf : GeoDataFrame
        人口空间单元数据，至少包含：
        - county_fips_col（schema 里定义）
        - population_col
        - geometry_col（多边形）
    schema : PopulationUnitSchema, optional
        定义列名的 schema。如果 None，则使用默认列名：
        - county_fips
        - population
        - geometry
    rng : np.random.Generator, optional
        随机数生成器；如果 None，则使用 default_rng()。

    返回
    ----
    epicenter : shapely.geometry.Point 或 None
        若在该县找不到任何人口单元，则返回 None。
    """
    if schema is None:
        schema = PopulationUnitSchema()

    if rng is None:
        rng = np.random.default_rng()

    fips_str = str(county_fips).zfill(5)

    # 过滤出目标县的人口单元
    sub = pop_gdf[
        pop_gdf[schema.county_fips_col].astype(str).str.zfill(5) == fips_str
    ].copy()
    if sub.empty:
        return None

    if schema.population_col not in sub.columns:
        raise ValueError(
            f"Population column '{schema.population_col}' not found in population GeoDataFrame."
        )

    # 权重向量（人口），需要保证非负且总和 > 0
    weights = np.asarray(sub[schema.population_col], dtype=float).clip(min=0.0)
    total_w = weights.sum()
    if total_w <= 0:
        # 该县人口列异常，退回到简单均匀抽样单元
        probs = None
    else:
        probs = weights / total_w

    # 在县内所有人口单元中，加权（或均匀）选一个多边形
    idx_choices = np.arange(len(sub))
    chosen_idx = rng.choice(idx_choices, p=probs)
    poly = sub.iloc[chosen_idx][schema.geometry_col]

    # 在所选多边形内部均匀抽样 epicenter
    return sample_point_in_polygon(poly, rng)


__all__ = [
    "PopulationUnitSchema",
    "load_population_units",
    "choose_population_weighted_epicenter",
]

