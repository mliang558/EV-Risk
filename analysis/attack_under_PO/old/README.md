# Realistic Power Outage Impact Simulation

A comprehensive framework for simulating and analyzing the impact of power outages on electric vehicle (EV) charging station networks.

## Overview

This simulation framework incorporates:
- **Bayesian-estimated county-level outage frequencies** (λ)
- **Bootstrap-sampled outage severity** from historical data
- **Spatial propagation** based on population density
- **Temporal integration** of network efficiency loss
- **Multi-state comparison** with normalization for comparability

## Project Structure

```
.
├── realistic_outage_simulator.py   # Core simulation engine
├── data_preparation.py             # Data conversion utilities
├── visualization.py                # Plotting and visualization tools
├── run_simulation.py              # Main execution script
├── README.md                      # This file
└── results/                       # Output directory (auto-created)
```

## Installation

### Requirements

```bash
pip install numpy pandas networkx matplotlib seaborn scipy --break-system-packages
```

### Python Version
- Python 3.7 or higher

## Data Format

The simulator requires a pickle file containing:

```python
state_data = {
    'network': NetworkX_Graph,           # Charging station network
    'county_lambdas': dict,              # County outage frequencies
    'historical_outages': dict,          # Bootstrap sampling data
    'county_demographics': dict          # Population, area, coordinates
}
```

### Network Format
```python
# Each node (charging station) must have:
G.nodes[node_id]['location'] = (latitude, longitude)
```

### County Lambda Format
```python
county_lambdas = {
    'County1': 2.5,  # events per year
    'County2': 3.1,
    ...
}
```

### Historical Outages Format
```python
historical_outages = {
    'County1': [
        {'duration': 3.5, 'affected_customers': 5000},
        {'duration': 2.1, 'affected_customers': 2000},
        ...
    ],
    ...
}
```

### County Demographics Format
```python
county_demographics = {
    'County1': {
        'population': 873965,
        'area_km2': 121.4,
        'centroid': (37.7749, -122.4194)  # (lat, lon)
    },
    ...
}
```

## Quick Start

### 1. Prepare Your Data

```python
from data_preparation import convert_data_to_required_format

# Convert your existing data
convert_data_to_required_format(
    network_pickle_path='your_network.pkl',
    bayesian_lambda_csv='county_lambdas.csv',
    historical_outages_csv='outage_history.csv',
    county_demographics_csv='county_demographics.csv',
    output_path='prepared_data.pkl'
)
```

### 2. Run Single State Analysis

```python
from realistic_outage_simulator import RealisticOutageSimulator

# Initialize simulator
simulator = RealisticOutageSimulator('California', 'prepared_data.pkl')

# Run Monte Carlo simulation
results = simulator.monte_carlo_simulation(n_simulations=1000)

# Generate report
report = simulator.generate_report(results)
print(report.T)
```

### 3. Compare Multiple States

```python
from realistic_outage_simulator import run_multi_state_comparison

state_configs = {
    'California': 'data/california.pkl',
    'Texas': 'data/texas.pkl',
    'Florida': 'data/florida.pkl'
}

comparison = run_multi_state_comparison(state_configs, n_simulations=1000)
print(comparison)
```

### 4. Create Visualizations

```python
from visualization import plot_loss_distribution, plot_state_comparison

# Single state visualization
plot_loss_distribution(results, 'California', save_path='ca_loss_dist.png')

# Multi-state comparison
plot_state_comparison(comparison, save_path='state_comparison.png')
```

## Using the Interactive Runner

The easiest way to run analyses is using the interactive script:

```bash
python run_simulation.py
```

This will guide you through:
1. Single state analysis
2. Multi-state comparison
3. Sensitivity analysis
4. Example with sample data

## Key Features

### Spatial Impact Modeling

The simulator calculates impact radius based on outage severity and population density:

```
r = sqrt(affected_customers / (population_density × π))
```

All charging stations within radius `r` from the county center are affected.

### Temporal Integration

Network efficiency loss is integrated over the outage duration:

**Constant Recovery Model:**
```
Total_Loss = (E_baseline - E_damaged) / E_baseline × duration
```

**Exponential Recovery Model:**
```
Total_Loss = L₀ × τ × (1 - exp(-T/τ))
```
where τ = duration / 2

### Normalization for Cross-State Comparison

Multiple normalization methods ensure fair comparison:

1. **Per Station**: `Loss / N_stations`
2. **Per Capita**: `Loss × 100,000 / Population`
3. **Per Area**: `Loss × 1,000 / Area_km²`
4. **Composite**: Combines population and area normalization

## Output Metrics

### Key Statistics
- **Mean Annual Loss**: Average efficiency loss per year
- **VaR (Value at Risk)**: 95th percentile loss
- **CVaR (Conditional VaR)**: Expected loss beyond VaR
- **Loss per Station**: Normalized by network size
- **Composite Normalized Loss**: Cross-state comparable metric

### Visualizations
- Loss distribution histogram
- Cumulative distribution function (CDF)
- State comparison charts
- Vulnerability rankings
- Risk metrics comparison
- Comprehensive dashboards

## Example Usage

### Complete Analysis Pipeline

```python
import os
from realistic_outage_simulator import RealisticOutageSimulator
from visualization import create_summary_dashboard

# Create output directory
os.makedirs('results', exist_ok=True)

# Run analysis
simulator = RealisticOutageSimulator('California', 'ca_data.pkl')
results = simulator.monte_carlo_simulation(n_simulations=1000)
report = simulator.generate_report(results)

# Save results
report.to_csv('results/california_report.csv', index=False)

# Create visualizations
create_summary_dashboard(
    results, 
    'California',
    save_path='results/california_dashboard.png'
)

print("Analysis complete!")
print(f"Mean Annual Loss: {results['mean_annual_loss']:.4f}")
print(f"95% VaR: {results['VaR_95']:.4f}")
```

## Simulation Parameters

### RealisticOutageSimulator.__init__()
- `state_name`: Name of the state (string)
- `data_path`: Path to prepared data pickle file

### monte_carlo_simulation()
- `n_simulations`: Number of simulation runs (default: 1000)
- `time_horizon`: Simulation duration in days (default: 365)
- `recovery_model`: 'constant' or 'exponential' (default: 'constant')
- `verbose`: Print progress (default: True)

## Advanced Features

### Custom Recovery Models

You can implement custom recovery models by modifying the `compute_efficiency_loss()` method.

### County-Level Analysis

Extract county-level vulnerability:

```python
county_losses = {}
for county in simulator.lambda_county.keys():
    # Simulate outages for specific county
    event = simulator.simulate_single_outage_event(county)
    county_losses[county] = event['efficiency_loss']

# Visualize
from visualization import plot_county_vulnerability
plot_county_vulnerability(simulator, county_losses)
```

### Sensitivity Analysis

Test different scenarios:

```python
scenarios = {
    'Baseline': 'constant',
    'Fast Recovery': 'exponential',
}

results = {}
for name, model in scenarios.items():
    results[name] = simulator.monte_carlo_simulation(
        n_simulations=500,
        recovery_model=model
    )

from visualization import plot_sensitivity_analysis
plot_sensitivity_analysis(results)
```

## Troubleshooting

### Common Issues

**Issue**: "KeyError: 'location'" when loading network
- **Solution**: Ensure all nodes have the 'location' attribute as (lat, lon) tuples

**Issue**: "ValueError: mismatch between counties"
- **Solution**: Ensure all data files cover the same set of counties

**Issue**: Network efficiency always zero
- **Solution**: Check that your network is connected (use `nx.is_connected(G)`)

**Issue**: Very large loss values
- **Solution**: Check your historical outage data for outliers; consider capping affected_customers

## Performance Tips

- For quick testing, use `n_simulations=100`
- For publication-quality results, use `n_simulations=1000+`
- Large networks (>1000 nodes) may require more memory
- Consider using the 'exponential' recovery model for more realistic results

## Citation

If you use this simulation framework in your research, please cite:

```
[Your citation information here]
```

## License

[Your license information]

## Contact

For questions or issues, please contact:
- [Your contact information]

## Acknowledgments

This framework implements the methodology described in:
- Module 3: Realistic Outage Simulation (Research Framework v2)
- Incorporates Bayesian outage frequency estimation
- Utilizes bootstrap sampling for severity modeling

---

**Version**: 1.0  
**Last Updated**: 2024  
**Compatibility**: Python 3.7+
