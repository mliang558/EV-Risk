# Global features: base 15 vs extended 18

## base (15-dim) — `GLOBAL_FEATURE_NAMES_BASE`

| # | Name | Meaning |
|---|------|---------|
| 1–7 | n_nodes, n_edges, density, total/mean/std/max capacity | Size & charging capacity |
| 8–9 | avg_degree, max_degree | Degree summary |
| 10 | avg_clustering | Clustering coefficient |
| 11–12 | max_betweenness, mean_betweenness | **Graph-theoretic** betweenness on the subgraph |
| 13 | global_efficiency | Global efficiency |
| 14 | **diameter** | Network diameter (largest component if disconnected) |
| 15 | n_components | # connected components |

## extended (+3) → **18-dim total**

**Not** diameter again — diameter is already in base.

| # | Name | Meaning | Why for betweenness attack |
|---|------|---------|---------------------------|
| 16 | **algebraic_connectivity** | Fiedler value (2nd smallest Laplacian eigenvalue) | Connectivity / bottlenecks |
| 17 | **avg_shortest_path** | Mean shortest path length (weighted; largest component) | Path structure related to flow |
| 18 | **degree_heterogeneity** | std(degree) / mean(degree) | Hub heterogeneity vs targeted removal |

Compute with full NetworkX in `step5_build_window_global_features.py` (**no `--fast`**).

## CLI

- GNN / LGB: `--global-feature-set base` (15) or `extended` (18)
- Precompute CSV: `step5_build_window_global_features.py --feature-set extended`
