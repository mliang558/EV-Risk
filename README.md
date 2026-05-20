# EV Charging Network Resilience Under Power Outages

Analysis of electric vehicle (EV) charging infrastructure vulnerability to power outages and coordinated infrastructure attacks across the United States.

## Overview

This project investigates resilience of EV charging networks through:
- **Network analysis** on AFDC charging station data (2018-2026, 49 states)
- **Bayesian power outage modeling** using 10 years of historical EAGLEI data (2014-2023)
- **Attack simulations** combining targeted node removal with realistic outage scenarios

## Project Structure

```
analysis/
├── run_charging_network.py              # Main network construction & attack simulation
├── plot_charging_network_results.py     # Results visualization
├── build_network_pickle_only.py         # Network serialization
│
├── attack_under_PO/                     # Integrated attack + outage analysis
│   ├── monte_carlo_outage_attack.py     # Monte Carlo simulations
│   ├── realistic_outage_simulator_improved.py
│   ├── compute_impact_radius.py
│   └── module_3_03/                     # Advanced simulation module
│
└── power_outage/
    ├── PO frequency modeling/           # Bayesian hierarchical model for outage frequency
    └── PO severity modeling/            # Bootstrap resampling for outage severity

notebooks/
└── PO_data_cleaning/                    # Data preprocessing & EDA
```

## Quick Start

### 1. Build charging networks
```bash
# Single year
python analysis/run_charging_network.py "data/raw/alt_fuel_stations_historical_day (Jan 1 2025).csv"

# Batch process all years (2018-2026)
python analysis/run_charging_network.py --batch-from-raw
```

**Method**: 200m spatial clustering → Voronoi adjacency → weighted network graph

### 2. Simulate attacks under power outages
```bash
cd analysis/attack_under_PO/
python monte_carlo_outage_attack.py --state CA --n-sims 1000
```

### 3. Visualize results
```bash
python analysis/plot_charging_network_results.py --run-dir outputs/charging_network/run__*
```

## Requirements

```bash
pip install numpy scipy pandas networkx scikit-learn pyproj matplotlib seaborn arviz pymc3
```

- Python 3.7+
- **Core**: numpy, scipy, pandas, networkx
- **Network**: scikit-learn, pyproj
- **Bayesian**: arviz, pymc3
- **Visualization**: matplotlib, seaborn

## Key Findings

**Network Characteristics (2025)**:
- Analyzed **49 US states** with **10,000+ charging stations**
- Built weighted graphs with Voronoi adjacency
- Computed network efficiency, clustering, degree distribution

**Attack Resilience**:
- Targeted attacks (degree/betweenness) cause 3-5× more damage than random failures
- Small networks (MT, WY, VT) show <3 alternative paths per node
- Large networks (CA, TX, NY) demonstrate higher redundancy

**Outputs**:
- `network_characteristics_ALL.csv`: Per-state metrics (nodes, edges, efficiency, clustering, density)
- `attack_results_ALL.csv`: Efficiency loss by strategy, removal %, and node removal strategy
- State-level vulnerability rankings and geospatial impact maps

## Algorithm Details

### Network Construction
1. **Spatial Clustering**: 200m greedy aggregation of charging station locations
2. **Graph Construction**: Voronoi diagram adjacency (Delaunay fallback)
3. **Edge Weighting**: 
   ```
   w_ij = (norm_distance + ε) / √(capacity_i × capacity_j)
   ```

### Attack Simulation
- **Strategies**: Random, Degree, Betweenness, Capacity, Closeness centrality
- **Metrics**: Global efficiency loss at removal percentages [5%, 10%, ..., 85%]
- **Reproducibility**: Random attacks run 10 times with fixed seed

### Power Outage Modeling
- **Frequency**: Bayesian hierarchical GLM (ADVI) for county-level event rates
- **Severity**: Bootstrap resampling from historical EAGLEI database
- **Integration**: Monte Carlo simulation of outage impact on network resilience

## License

Research use - Available for academic and educational purposes

## References

- AFDC (Alternative Fuels Data Center): www.afdc.energy.gov
- EAGLEI Database: Historical power outage records
- NetworkX documentation for graph algorithms
