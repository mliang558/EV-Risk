#!/usr/bin/env python3
"""
State-Level Frequency Modeling for Power Outages
=================================================
This script aggregates county-level outage data to state level and performs
hierarchical Bayesian modeling to estimate outage frequencies (lambda).

Input: CSV file with columns: fips, state, start_time, duration_min, customers_affected
Output: State-level frequency estimates with credible intervals
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
import yaml
warnings.filterwarnings('ignore')


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class StateFrequencyModeler:
    """
    Model power outage frequencies at the state level using hierarchical Bayesian approach.
    """
    
    def __init__(self, data_path=None, config_path=None):
        """
        Initialize the modeler with data path or config file.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the cleaned data CSV file (overrides config)
        config_path : str, optional
            Path to YAML config file (default: config.yaml)
        """
        # Load configuration
        if config_path is None:
            config_path = 'config.yaml'
        
        if Path(config_path).exists():
            print(f"Loading configuration from: {config_path}")
            self.config = load_config(config_path)
        else:
            print("No config file found, using defaults")
            self.config = self._default_config()
        
        # Override data path if provided
        if data_path is not None:
            self.config['data']['input_path'] = data_path
        
        self.data_path = self.config['data']['input_path']
        self.df = None
        self.state_agg = None
        self.trace = None
        self.model = None
    
    def _default_config(self):
        """Return default configuration."""
        return {
            'data': {
                'input_path': 'cleaned_data.csv',
                'columns': {
                    'fips': 'fips',
                    'state': 'state',
                    'start_time': 'start_time',
                    'duration_min': 'duration_min',
                    'customers_affected': 'customers_affected'
                },
                'filters': {
                    'min_duration_min': 0,
                    'max_duration_min': None,
                    'min_customers': 0,
                    'states_to_exclude': []
                }
            },
            'model': {
                'priors': {
                    'mu_national_mean': np.log(5),
                    'mu_national_sigma': 2.0,
                    'sigma_state': 1.0
                },
                'sampling': {
                    'n_samples': 2000,
                    'n_tune': 1000,
                    'target_accept': 0.95,
                    'cores': None,
                    'random_seed': 42
                }
            },
            'output': {
                'output_dir': './results',
                'plots': {
                    'create_plots': True,
                    'dpi': 300,
                    'figure_format': 'png'
                },
                'export': {
                    'csv': True,
                    'summary_text': True
                }
            },
            'advanced': {
                'duration_aggregation': 'weighted_mean',
                'duration_weight_column': 'mean_customers',
                'credible_interval': 0.95,
                'verbose': True
            }
        }
        
    def load_and_prepare_data(self):
        """Load data and prepare it for state-level analysis."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Get column mappings from config
        col_map = self.config['data']['columns']
        
        # Rename columns to standard names
        rename_dict = {}
        for standard_name, csv_column in col_map.items():
            if csv_column in self.df.columns:
                rename_dict[csv_column] = standard_name
        
        self.df = self.df.rename(columns=rename_dict)
        
        # Verify required columns exist
        required = ['fips', 'state', 'start_time', 'duration_min']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert start_time to datetime
        self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        self.df['year'] = self.df['start_time'].dt.year
        
        # Calculate duration in hours
        self.df['duration_hr'] = self.df['duration_min'] / 60
        
        # Apply filters if specified
        filters = self.config['data']['filters']
        
        if filters.get('min_duration_min', 0) > 0:
            before = len(self.df)
            self.df = self.df[self.df['duration_min'] >= filters['min_duration_min']]
            print(f"Filtered by min duration: {before} -> {len(self.df)} events")
        
        if filters.get('max_duration_min') is not None:
            before = len(self.df)
            self.df = self.df[self.df['duration_min'] <= filters['max_duration_min']]
            print(f"Filtered by max duration: {before} -> {len(self.df)} events")
        
        if 'customers_affected' in self.df.columns and filters.get('min_customers', 0) > 0:
            before = len(self.df)
            self.df = self.df[self.df['customers_affected'] >= filters['min_customers']]
            print(f"Filtered by min customers: {before} -> {len(self.df)} events")
        
        if filters.get('states_to_exclude'):
            before = len(self.df)
            self.df = self.df[~self.df['state'].isin(filters['states_to_exclude'])]
            print(f"Excluded states {filters['states_to_exclude']}: {before} -> {len(self.df)} events")
        
        print(f"Loaded {len(self.df)} outage events")
        print(f"Date range: {self.df['start_time'].min()} to {self.df['start_time'].max()}")
        print(f"States covered: {self.df['state'].nunique()}")
        
        return self
    
    def aggregate_to_state_level(self):
        """
        Aggregate county-level data to state level.
        
        For each state, we calculate:
        - Total number of events
        - Number of years observed (for exposure calculation)
        - Mean duration (weighted by customers affected or simple mean)
        - Total customers affected (absolute numbers)
        """
        print("\nAggregating to state level...")
        
        # State-level event counts
        state_events = self.df.groupby('state').agg(
            n_events=('state', 'count'),
            min_year=('year', 'min'),
            max_year=('year', 'max')
        ).reset_index()
        
        # Calculate years observed (exposure period)
        state_events['n_years_observed'] = state_events['max_year'] - state_events['min_year'] + 1
        
        # Calculate mean duration based on config
        agg_method = self.config['advanced']['duration_aggregation']
        weight_col = self.config['advanced'].get('duration_weight_column')
        
        if agg_method == 'weighted_mean' and 'customers_affected' in self.df.columns:
            weighted_duration = self.df.groupby('state').apply(
                lambda x: np.average(x['duration_hr'], weights=x['customers_affected'])
            ).reset_index(name='mean_duration_hr')
            print(f"Using weighted mean duration (weighted by {weight_col})")
        elif agg_method == 'median':
            weighted_duration = self.df.groupby('state')['duration_hr'].median().reset_index(
                name='mean_duration_hr'
            )
            print("Using median duration")
        else:
            weighted_duration = self.df.groupby('state')['duration_hr'].mean().reset_index(
                name='mean_duration_hr'
            )
            print("Using simple mean duration")
        
        # Total customers affected per state
        if 'customers_affected' in self.df.columns:
            total_customers = self.df.groupby('state')['customers_affected'].sum().reset_index(
                name='total_customers_affected'
            )
        else:
            print("Warning: 'customers_affected' column not found. Setting to NaN.")
            total_customers = pd.DataFrame({
                'state': state_events['state'],
                'total_customers_affected': np.nan
            })
        
        # Merge all aggregations
        self.state_agg = state_events.merge(weighted_duration, on='state')
        self.state_agg = self.state_agg.merge(total_customers, on='state')
        
        # Calculate raw lambda (events per year)
        self.state_agg['lambda_raw'] = (
            self.state_agg['n_events'] / self.state_agg['n_years_observed']
        )
        
        print(f"Aggregated to {len(self.state_agg)} states")
        print("\nState-level summary statistics:")
        print(self.state_agg[['n_events', 'n_years_observed', 'lambda_raw']].describe())
        
        return self
    
    def build_hierarchical_model(self):
        """
        Build hierarchical Bayesian model for state-level frequency estimation.
        
        Model structure:
        - National-level hyperprior for mean log(lambda)
        - State-level log(lambda) drawn from national distribution
        - Observed events ~ Poisson(lambda * years_observed)
        """
        print("\nBuilding hierarchical Bayesian model...")
        
        n_states = len(self.state_agg)
        Y_obs = self.state_agg['n_events'].values.astype(int)
        T_obs = self.state_agg['n_years_observed'].values.astype(float)
        
        # Get priors from config
        priors = self.config['model']['priors']
        
        with pm.Model() as self.model:
            # National-level hyperpriors
            mu_national = pm.Normal(
                "mu_national",
                mu=priors['mu_national_mean'],
                sigma=priors['mu_national_sigma']
            )
            sigma_state = pm.HalfNormal(
                "sigma_state",
                sigma=priors['sigma_state']
            )
            
            # State-level log(lambda)
            log_lambda = pm.Normal(
                "log_lambda",
                mu=mu_national,
                sigma=sigma_state,
                shape=n_states
            )
            
            # Transform to lambda (events per year)
            lambda_state = pm.Deterministic("lambda_state", pm.math.exp(log_lambda))
            
            # Expected counts = lambda * years_observed
            mu_expected = lambda_state * T_obs
            
            # Likelihood: observed events ~ Poisson(expected)
            pm.Poisson("Y_obs", mu=mu_expected, observed=Y_obs)
        
        print("Model built successfully")
        print(f"Model has {n_states} states")
        print(f"Priors: μ_national ~ N({priors['mu_national_mean']:.2f}, {priors['mu_national_sigma']:.2f})")
        print(f"        σ_state ~ HalfNormal({priors['sigma_state']:.2f})")
        
        return self
    
    def sample_posterior(self, n_samples=None, n_tune=None, target_accept=None, cores=None):
        """
        Sample from the posterior distribution using MCMC.
        
        Parameters:
        -----------
        n_samples : int or None
            Number of samples to draw (overrides config)
        n_tune : int or None
            Number of tuning samples (overrides config)
        target_accept : float or None
            Target acceptance rate for NUTS sampler (overrides config)
        cores : int or None
            Number of CPU cores to use (overrides config)
        """
        # Get sampling params from config if not provided
        sampling_config = self.config['model']['sampling']
        
        if n_samples is None:
            n_samples = sampling_config['n_samples']
        if n_tune is None:
            n_tune = sampling_config['n_tune']
        if target_accept is None:
            target_accept = sampling_config['target_accept']
        if cores is None:
            cores = sampling_config['cores']
        
        print(f"\nSampling posterior (draws={n_samples}, tune={n_tune})...")
        
        # Auto-detect cores if not specified
        if cores is None:
            import platform
            if platform.system() == 'Windows':
                print("Windows detected - using single core for stability")
                cores = 1
            else:
                cores = 4  # Use 4 cores on Unix systems
        
        print(f"Using {cores} core(s) for sampling...")
        
        with self.model:
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                target_accept=target_accept,
                cores=cores,
                return_inferencedata=True,
                random_seed=sampling_config['random_seed']
            )
        
        print("Sampling complete!")
        
        # Print diagnostics
        print("\nConvergence diagnostics:")
        print(f"Effective sample size (ESS):")
        ess = az.ess(self.trace, var_names=['lambda_state'])
        print(f"  Min ESS: {ess.lambda_state.min().values:.0f}")
        print(f"  Mean ESS: {ess.lambda_state.mean().values:.0f}")
        
        rhat = az.rhat(self.trace, var_names=['lambda_state'])
        print(f"R-hat statistics:")
        print(f"  Max R-hat: {rhat.lambda_state.max().values:.4f}")
        
        if rhat.lambda_state.max().values > 1.01:
            print("  WARNING: Some R-hat values > 1.01, consider more samples")
        
        return self
    
    def extract_state_estimates(self):
        """
        Extract posterior estimates for each state.
        
        Returns:
        --------
        pd.DataFrame with columns:
        - state: state name
        - lambda_mean: posterior mean of lambda
        - lambda_median: posterior median
        - lambda_sd: posterior standard deviation
        - lambda_ci_lower: credible interval lower bound
        - lambda_ci_upper: credible interval upper bound
        """
        print("\nExtracting state-level estimates...")
        
        # Get credible interval level from config
        ci_level = self.config['advanced']['credible_interval']
        lower_percentile = (1 - ci_level) / 2 * 100
        upper_percentile = (1 - (1 - ci_level) / 2) * 100
        
        # Get lambda_state posterior samples
        lambda_posterior = self.trace.posterior['lambda_state']
        
        # Calculate summary statistics
        results = []
        for i, state in enumerate(self.state_agg['state']):
            samples = lambda_posterior.sel(lambda_state_dim_0=i).values.flatten()
            
            results.append({
                'state': state,
                'n_events': self.state_agg.iloc[i]['n_events'],
                'n_years_observed': self.state_agg.iloc[i]['n_years_observed'],
                'lambda_raw': self.state_agg.iloc[i]['lambda_raw'],
                'lambda_mean': samples.mean(),
                'lambda_median': np.median(samples),
                'lambda_sd': samples.std(),
                'lambda_ci_lower': np.percentile(samples, lower_percentile),
                'lambda_ci_upper': np.percentile(samples, upper_percentile),
                'mean_duration_hr': self.state_agg.iloc[i]['mean_duration_hr'],
                'total_customers_affected': self.state_agg.iloc[i]['total_customers_affected']
            })
        
        results_df = pd.DataFrame(results)
        
        # Sort by lambda_mean descending
        results_df = results_df.sort_values('lambda_mean', ascending=False)
        
        print(f"\nTop 10 states by estimated frequency (lambda) with {ci_level*100:.0f}% CI:")
        print(results_df[['state', 'lambda_mean', 'lambda_ci_lower', 'lambda_ci_upper']].head(10).to_string(index=False))
        
        return results_df
    
    def plot_results(self, results_df, output_dir=None):
        """
        Create visualization plots for the results.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results dataframe from extract_state_estimates
        output_dir : str or None
            Directory to save plots (overrides config)
        """
        if output_dir is None:
            output_dir = self.config['output']['output_dir']
        
        # Check if plotting is enabled
        if not self.config['output']['plots']['create_plots']:
            print("\nPlotting disabled in config")
            return self
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        dpi = self.config['output']['plots']['dpi']
        fmt = self.config['output']['plots']['figure_format']
        
        print(f"\nCreating visualizations in {output_dir}...")
        
        # Plot 1: State-level lambda estimates with credible intervals
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by mean lambda
        plot_df = results_df.sort_values('lambda_mean', ascending=True)
        
        # Plot error bars
        y_pos = np.arange(len(plot_df))
        ax.errorbar(
            plot_df['lambda_mean'],
            y_pos,
            xerr=[
                plot_df['lambda_mean'] - plot_df['lambda_ci_lower'],
                plot_df['lambda_ci_upper'] - plot_df['lambda_mean']
            ],
            fmt='o',
            markersize=6,
            capsize=3,
            alpha=0.7
        )
        
        # Also plot raw lambda for comparison
        ax.scatter(
            plot_df['lambda_raw'],
            y_pos,
            marker='x',
            s=50,
            alpha=0.5,
            color='red',
            label='Raw λ (observed)'
        )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['state'])
        ax.set_xlabel('Events per Year (λ)', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.set_title('State-Level Outage Frequency Estimates\n(with 95% Credible Intervals)', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path / f'state_lambda_estimates.{fmt}', dpi=dpi, bbox_inches='tight')
        print(f"  Saved: state_lambda_estimates.{fmt}")
        plt.close()
        
        # Plot 2: Shrinkage comparison (raw vs. hierarchical estimates)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(
            results_df['lambda_raw'],
            results_df['lambda_mean'],
            s=100,
            alpha=0.6,
            c=results_df['n_events'],
            cmap='viridis'
        )
        
        # Add diagonal line
        max_val = max(results_df['lambda_raw'].max(), results_df['lambda_mean'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        
        ax.set_xlabel('Raw λ (Observed)', fontsize=12)
        ax.set_ylabel('Hierarchical λ (Posterior Mean)', fontsize=12)
        ax.set_title('Shrinkage Effect: Raw vs. Hierarchical Estimates', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(results_df['n_events'])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Number of Events', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / f'shrinkage_comparison.{fmt}', dpi=dpi, bbox_inches='tight')
        print(f"  Saved: shrinkage_comparison.{fmt}")
        plt.close()
        
        # Plot 3: Trace plots for convergence diagnostics
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # Plot national mean and state sigma
        az.plot_trace(
            self.trace,
            var_names=['mu_national', 'sigma_state'],
            figsize=(14, 8)
        )
        
        plt.tight_layout()
        plt.savefig(output_path / f'trace_diagnostics.{fmt}', dpi=dpi, bbox_inches='tight')
        print(f"  Saved: trace_diagnostics.{fmt}")
        plt.close()
        
        # Plot 4: Distribution of state lambdas
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(results_df['lambda_mean'], bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(
            results_df['lambda_mean'].mean(),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {results_df["lambda_mean"].mean():.2f}'
        )
        ax.set_xlabel('State-Level λ (Events per Year)', fontsize=12)
        ax.set_ylabel('Number of States', fontsize=12)
        ax.set_title('Distribution of State-Level Outage Frequencies', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path / f'lambda_distribution.{fmt}', dpi=dpi, bbox_inches='tight')
        print(f"  Saved: lambda_distribution.{fmt}")
        plt.close()
        
        return self
    
    def save_results(self, results_df, output_dir=None):
        """
        Save results to CSV files.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results dataframe
        output_dir : str or None
            Directory to save outputs (overrides config)
        """
        if output_dir is None:
            output_dir = self.config['output']['output_dir']
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        export_config = self.config['output']['export']
        
        # Save main results CSV
        if export_config.get('csv', True):
            output_file = output_path / 'state_frequency_estimates.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\nSaved results to: {output_file}")
        
        # Save JSON if requested
        if export_config.get('json', False):
            json_file = output_path / 'state_frequency_estimates.json'
            results_df.to_json(json_file, orient='records', indent=2)
            print(f"Saved JSON to: {json_file}")
        
        # Save summary statistics
        if export_config.get('summary_text', True):
            summary_file = output_path / 'summary_statistics.txt'
            ci_level = self.config['advanced']['credible_interval']
            
            with open(summary_file, 'w') as f:
                f.write("STATE-LEVEL FREQUENCY MODELING SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Total states analyzed: {len(results_df)}\n")
                f.write(f"Total events: {results_df['n_events'].sum()}\n")
                f.write(f"Average years observed: {results_df['n_years_observed'].mean():.1f}\n\n")
                
                f.write("Lambda statistics (events per year):\n")
                f.write(f"  Mean: {results_df['lambda_mean'].mean():.2f}\n")
                f.write(f"  Median: {results_df['lambda_mean'].median():.2f}\n")
                f.write(f"  Min: {results_df['lambda_mean'].min():.2f}\n")
                f.write(f"  Max: {results_df['lambda_mean'].max():.2f}\n")
                f.write(f"  Std: {results_df['lambda_mean'].std():.2f}\n\n")
                
                f.write(f"Top 10 highest risk states (with {ci_level*100:.0f}% credible intervals):\n")
                for i, row in results_df.head(10).iterrows():
                    f.write(f"  {row['state']}: {row['lambda_mean']:.2f} "
                           f"[{row['lambda_ci_lower']:.2f}, {row['lambda_ci_upper']:.2f}]\n")
            
            print(f"Saved summary to: {summary_file}")
        
        return self
    
    def run_full_pipeline(self, output_dir=None):
        """
        Run the complete modeling pipeline.
        
        Parameters:
        -----------
        output_dir : str or None
            Directory to save all outputs (overrides config)
        """
        if output_dir is None:
            output_dir = self.config['output']['output_dir']
        
        print("=" * 60)
        print("STATE-LEVEL FREQUENCY MODELING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Aggregate to state level
        self.aggregate_to_state_level()
        
        # Step 3: Build model
        self.build_hierarchical_model()
        
        # Step 4: Sample posterior
        self.sample_posterior()
        
        # Step 5: Extract estimates
        results_df = self.extract_state_estimates()
        
        # Step 6: Create visualizations
        self.plot_results(results_df, output_dir)
        
        # Step 7: Save results
        self.save_results(results_df, output_dir)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        
        return results_df


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='State-level frequency modeling for power outages'
    )
    parser.add_argument(
        'data_path',
        type=str,
        nargs='?',
        default=None,
        help='Path to cleaned data CSV file (overrides config)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (overrides config)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of MCMC samples (overrides config)'
    )
    parser.add_argument(
        '--tune',
        type=int,
        default=None,
        help='Number of tuning samples (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    modeler = StateFrequencyModeler(data_path=args.data_path, config_path=args.config)
    
    # Override sampling parameters if provided
    if args.samples or args.tune:
        modeler.sample_posterior = lambda: modeler.sample_posterior(
            n_samples=args.samples,
            n_tune=args.tune
        )
    
    results = modeler.run_full_pipeline(output_dir=args.output_dir)
    
    print(f"\nResults shape: {results.shape}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    else:
        print(f"Output directory: {modeler.config['output']['output_dir']}")


if __name__ == '__main__':
    main()
