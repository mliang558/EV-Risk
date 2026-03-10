# State-Level Frequency Modeling for Power Outages

## Overview
This package performs hierarchical Bayesian modeling to estimate power outage frequencies (λ) at the state level, accounting for spatial correlation and data sparsity.

## Files
- `state_frequency_modeling.py` - Main modeling pipeline
- `config.yaml` - Configuration file for data paths and model parameters
- `run_real_data.py` - Simple script to run with your data
- `test_state_modeling.py` - Test script with synthetic data
- `README.md` - This file

## Requirements
```bash
pip install pandas numpy pymc arviz matplotlib seaborn pyyaml
```

## Quick Start

### Method 1: Edit config.yaml (Recommended)

1. **Edit `config.yaml`** to set your data path:
```yaml
data:
  input_path: "cleaned_data_1117.csv"  # Change this to your file path
  columns:
    fips: "fips"
    state: "state"
    start_time: "start_time"
    duration_min: "duration_min"
    customers_affected: "mean_customers"  # Map to your column name
```

2. **Run the pipeline:**
```bash
python state_frequency_modeling.py
```

Or use the simple run script:
```bash
python run_real_data.py
```

### Method 2: Command Line (Override config)
```bash
python state_frequency_modeling.py cleaned_data_1117.csv --output-dir ./my_results
```

### Method 3: Test First
```bash
python test_state_modeling.py  # Uses synthetic data
```

## Configuration File (config.yaml)

The config file lets you customize everything without editing code:

### Data Configuration
```yaml
data:
  input_path: "cleaned_data_1117.csv"
  
  columns:  # Map YOUR column names to expected names
    fips: "fips"
    state: "state"
    start_time: "start_time"
    duration_min: "duration_min"
    customers_affected: "mean_customers"
  
  filters:  # Optional data filtering
    min_duration_min: 0
    states_to_exclude: []  # e.g., ['AK', 'HI']
```

### Model Configuration
```yaml
model:
  priors:
    mu_national_mean: 1.6094  # log(5) events/year
    sigma_state: 1.0          # Between-state variation
  
  sampling:
    n_samples: 2000
    n_tune: 1000
    cores: null  # Auto-detect (1 for Windows)
```

### Output Configuration
```yaml
output:
  output_dir: "./results"
  
  plots:
    create_plots: true
    dpi: 300
    figure_format: "png"  # or "pdf", "svg"
  
  export:
    csv: true
    json: false
    summary_text: true
```

## Your Data Format

Your CSV should have these columns (column names are configurable in `config.yaml`):
- `fips` - County FIPS code (5 digits)
- `state` - State abbreviation
- `start_time` - Event start timestamp
- `duration_min` - Duration in minutes
- `mean_customers` (or similar) - Customers affected

Example with your actual columns:
```csv
fips,state,county,start_time,end_time,min_customers,max_customers,mean_customers,duration,duration_min
06001,CA,Alameda,2020-01-15 14:30:00,2020-01-15 16:30:00,1000,5000,3000,120 min,120
```

Just map your column names in `config.yaml`:
```yaml
columns:
  customers_affected: "mean_customers"  # Your column name
  duration_min: "duration_min"          # Already matches
```

## Command-Line Options
```bash
python state_frequency_modeling.py --help

positional arguments:
  data_path            Path to CSV (overrides config)

optional arguments:
  --config PATH        Config YAML file (default: config.yaml)
  --output-dir DIR     Output directory (overrides config)
  --samples N          MCMC samples (overrides config)
  --tune N             Tuning samples (overrides config)
```

## Output Files

All outputs go to the directory specified in config (default: `./results/`):

### CSV Files
- `state_frequency_estimates.csv` - Lambda estimates with credible intervals
- `summary_statistics.txt` - Text summary

### Plots
- `state_lambda_estimates.png` - State frequencies with error bars
- `shrinkage_comparison.png` - Raw vs hierarchical estimates
- `trace_diagnostics.png` - MCMC convergence checks
- `lambda_distribution.png` - Distribution of frequencies

## Example Python Usage

```python
from state_frequency_modeling import StateFrequencyModeler

# Method 1: Use config file
modeler = StateFrequencyModeler(config_path='config.yaml')
results = modeler.run_full_pipeline()

# Method 2: Override data path
modeler = StateFrequencyModeler(data_path='my_data.csv')
results = modeler.run_full_pipeline(output_dir='./my_analysis')

# Method 3: Step by step with custom parameters
modeler = StateFrequencyModeler('cleaned_data_1117.csv')
modeler.load_and_prepare_data()
modeler.aggregate_to_state_level()
modeler.build_hierarchical_model()
modeler.sample_posterior(n_samples=3000, cores=1)  # More samples
results = modeler.extract_state_estimates()
modeler.plot_results(results)
```

## Troubleshooting

### "Missing required columns" error
Edit the `columns` section in `config.yaml` to match your CSV column names.

### Windows multiprocessing issues
The code auto-detects Windows and uses 1 core. You can also set in config:
```yaml
model:
  sampling:
    cores: 1
```

### Convergence warnings (R-hat > 1.01)
Increase samples in `config.yaml`:
```yaml
model:
  sampling:
    n_samples: 5000
    n_tune: 2000
```

## Next Steps

After getting state-level λ estimates:
1. Use these in Step 2B (severity modeling)
2. Calculate risk scores: f(λ, P_severe, Network_vuln)
3. Feed into Module 3 (attack simulation)

## Contact
Questions? Check the research framework documentation.
