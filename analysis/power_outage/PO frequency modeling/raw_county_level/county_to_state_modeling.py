#!/usr/bin/env python3
"""
County-Level Hierarchical Frequency Model with State Aggregation
================================================================
This script builds a county-level model and aggregates results to state level
for fair inter-state comparison.

Pipeline:
1. County-level Bayesian model: County → State → National hierarchy
2. Get posterior distributions for each county
3. Aggregate to state level (both total and per-county average)
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CountyStateFrequencyModeler:
    """
    County-level hierarchical model with state-level aggregation.
    """
    
    def __init__(self, data_path, config_path=None):
        """Initialize modeler."""
        self.data_path = data_path
        self.df = None
        self.county_agg = None
        self.state_map = None
        self.trace = None
        self.model = None
        
    def load_and_prepare_data(self):
        """Load county-level data."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Convert timestamps
        self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        self.df['year'] = self.df['start_time'].dt.year
        self.df['duration_hr'] = self.df['duration_min'] / 60
        
        print(f"Loaded {len(self.df)} outage events")
        print(f"Counties: {self.df['fips'].nunique()}")
        print(f"States: {self.df['state'].nunique()}")
        
        return self
    
    def aggregate_to_county_level(self):
        """
        Aggregate events to county level.
        Each county gets: n_events, n_years_observed, lambda_raw
        """
        print("\nAggregating to county level...")
        
        # County-level aggregation
        county_agg = self.df.groupby(['fips', 'state']).agg(
            n_events=('fips', 'count'),
            min_year=('year', 'min'),
            max_year=('year', 'max')
        ).reset_index()
        
        # Calculate years observed
        county_agg['n_years_observed'] = (
            county_agg['max_year'] - county_agg['min_year'] + 1
        )
        
        # Raw lambda
        county_agg['lambda_raw'] = (
            county_agg['n_events'] / county_agg['n_years_observed']
        )
        
        # Create state index mapping
        unique_states = sorted(county_agg['state'].unique())
        self.state_map = {s: i for i, s in enumerate(unique_states)}
        county_agg['state_idx'] = county_agg['state'].map(self.state_map)
        
        self.county_agg = county_agg.reset_index(drop=True)
        
        print(f"Aggregated to {len(self.county_agg)} counties")
        print(f"Across {len(unique_states)} states")
        print(f"\nCounty-level lambda statistics:")
        print(self.county_agg['lambda_raw'].describe())
        
        return self
    
    def build_hierarchical_model(self):
        """
        Build 3-level hierarchical model: County → State → National
        
        Structure:
          National: μ_national, σ_state
          State: μ_state[s] ~ Normal(μ_national, σ_state)
          County: log(λ_c) ~ Normal(μ_state[s], σ_county)
          Observation: Y_c ~ Poisson(λ_c × T_c)
        """
        print("\n" + "="*60)
        print("BUILDING COUNTY-LEVEL HIERARCHICAL MODEL")
        print("="*60)
        
        n_counties = len(self.county_agg)
        n_states = len(self.state_map)
        
        # Data
        Y_obs = self.county_agg['n_events'].values.astype(int)
        T_obs = self.county_agg['n_years_observed'].values.astype(float)
        state_idx = self.county_agg['state_idx'].values
        
        # Estimate priors from data
        log_lambda_raw = np.log(self.county_agg['lambda_raw'] + 0.1)
        mu_data = log_lambda_raw.mean()
        sigma_data = log_lambda_raw.std()
        
        print(f"\nData-driven priors:")
        print(f"  Empirical mean log(λ): {mu_data:.3f}")
        print(f"  Empirical SD log(λ): {sigma_data:.3f}")
        
        with pm.Model() as self.model:
            # National-level hyperpriors
            mu_national = pm.Normal("mu_national", mu=mu_data, sigma=sigma_data * 2)
            sigma_state = pm.HalfNormal("sigma_state", sigma=sigma_data)
            sigma_county = pm.HalfNormal("sigma_county", sigma=sigma_data * 0.8)
            
            # State-level parameters (one per state)
            mu_state = pm.Normal(
                "mu_state",
                mu=mu_national,
                sigma=sigma_state,
                shape=n_states
            )
            
            # County-level log(lambda)
            log_lambda = pm.Normal(
                "log_lambda",
                mu=mu_state[state_idx],  # Each county belongs to a state
                sigma=sigma_county,
                shape=n_counties
            )
            
            # Transform to lambda
            lambda_county = pm.Deterministic("lambda_county", pm.math.exp(log_lambda))
            
            # Likelihood
            mu_expected = lambda_county * T_obs
            pm.Poisson("Y_obs", mu=mu_expected, observed=Y_obs)
        
        print(f"\nModel structure:")
        print(f"  Counties: {n_counties}")
        print(f"  States: {n_states}")
        print(f"  Hierarchy: County → State → National")
        
        return self
    
    def sample_posterior(self, n_samples=1000, n_tune=1000, cores=1):
        """Sample from posterior."""
        print(f"\nSampling posterior (draws={n_samples}, tune={n_tune})...")
        print(f"Warning: County-level model with {len(self.county_agg)} counties")
        print(f"This may take 10-30 minutes...")
        
        with self.model:
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                target_accept=0.90,
                cores=cores,
                return_inferencedata=True,
                random_seed=42
            )
        
        print("Sampling complete!")
        
        # Check convergence
        rhat = az.rhat(self.trace, var_names=['lambda_county'])
        print(f"\nConvergence check:")
        print(f"  Max R-hat: {rhat.lambda_county.max().values:.4f}")
        if rhat.lambda_county.max().values > 1.01:
            print("  ⚠️  Warning: Some parameters may not have converged")
        else:
            print("  ✓ All parameters converged")
        
        return self
    
    def aggregate_to_state_level(self):
        """
        Aggregate county-level posteriors to state level.
        
        Returns two types of state-level estimates:
        1. Total lambda: Sum of all counties in the state
        2. Average lambda per county: Mean of counties in the state
        """
        print("\n" + "="*60)
        print("AGGREGATING TO STATE LEVEL")
        print("="*60)
        
        # Get county-level posterior samples
        lambda_posterior = self.trace.posterior['lambda_county']  # (chains, draws, counties)
        
        # Flatten chains and draws
        lambda_samples = lambda_posterior.values.reshape(-1, len(self.county_agg))  # (samples, counties)
        
        print(f"Posterior samples shape: {lambda_samples.shape}")
        print(f"  {lambda_samples.shape[0]} samples × {lambda_samples.shape[1]} counties")
        
        # Aggregate by state
        state_results = []
        
        for state_name, state_idx in self.state_map.items():
            # Get counties in this state
            county_mask = self.county_agg['state_idx'] == state_idx
            county_indices = np.where(county_mask)[0]
            n_counties = len(county_indices)
            
            # Get lambda samples for these counties
            state_county_lambdas = lambda_samples[:, county_indices]  # (samples, n_counties_in_state)
            
            # Method 1: Total lambda (sum across counties)
            lambda_total_samples = state_county_lambdas.sum(axis=1)
            
            # Method 2: Average lambda per county
            lambda_avg_samples = state_county_lambdas.mean(axis=1)
            
            # Summary statistics
            state_results.append({
                'state': state_name,
                'n_counties': n_counties,
                
                # Total frequency
                'lambda_total_mean': lambda_total_samples.mean(),
                'lambda_total_median': np.median(lambda_total_samples),
                'lambda_total_sd': lambda_total_samples.std(),
                'lambda_total_ci_lower': np.percentile(lambda_total_samples, 2.5),
                'lambda_total_ci_upper': np.percentile(lambda_total_samples, 97.5),
                
                # Average per county
                'lambda_per_county_mean': lambda_avg_samples.mean(),
                'lambda_per_county_median': np.median(lambda_avg_samples),
                'lambda_per_county_sd': lambda_avg_samples.std(),
                'lambda_per_county_ci_lower': np.percentile(lambda_avg_samples, 2.5),
                'lambda_per_county_ci_upper': np.percentile(lambda_avg_samples, 97.5),
            })
        
        results_df = pd.DataFrame(state_results)
        
        print("\n✓ State-level aggregation complete")
        print(f"\nTop 10 states by TOTAL lambda:")
        print(results_df.sort_values('lambda_total_mean', ascending=False)
              [['state', 'n_counties', 'lambda_total_mean']].head(10).to_string(index=False))
        
        print(f"\nTop 10 states by AVERAGE lambda per county:")
        print(results_df.sort_values('lambda_per_county_mean', ascending=False)
              [['state', 'n_counties', 'lambda_per_county_mean']].head(10).to_string(index=False))
        
        return results_df
    
    def plot_state_comparison(self, results_df, output_dir='./results'):
        """Create comparison plots for state-level results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\nCreating state comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Total lambda by state
        ax = axes[0, 0]
        plot_df = results_df.sort_values('lambda_total_mean', ascending=True).tail(20)
        y_pos = np.arange(len(plot_df))
        
        ax.errorbar(
            plot_df['lambda_total_mean'],
            y_pos,
            xerr=[
                plot_df['lambda_total_mean'] - plot_df['lambda_total_ci_lower'],
                plot_df['lambda_total_ci_upper'] - plot_df['lambda_total_mean']
            ],
            fmt='o', markersize=6, capsize=3, alpha=0.7
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['state'])
        ax.set_xlabel('Total λ (events/year)', fontsize=11)
        ax.set_title('Top 20 States by TOTAL Frequency', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        
        # Plot 2: Average lambda per county
        ax = axes[0, 1]
        plot_df = results_df.sort_values('lambda_per_county_mean', ascending=True).tail(20)
        y_pos = np.arange(len(plot_df))
        
        ax.errorbar(
            plot_df['lambda_per_county_mean'],
            y_pos,
            xerr=[
                plot_df['lambda_per_county_mean'] - plot_df['lambda_per_county_ci_lower'],
                plot_df['lambda_per_county_ci_upper'] - plot_df['lambda_per_county_mean']
            ],
            fmt='o', markersize=6, capsize=3, alpha=0.7, color='green'
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['state'])
        ax.set_xlabel('Average λ per County (events/county/year)', fontsize=11)
        ax.set_title('Top 20 States by PER-COUNTY Frequency', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        
        # Plot 3: Total vs Per-County (scatter)
        ax = axes[1, 0]
        ax.scatter(
            results_df['lambda_per_county_mean'],
            results_df['lambda_total_mean'],
            s=results_df['n_counties'] * 2,
            alpha=0.6,
            c=results_df['n_counties'],
            cmap='viridis'
        )
        
        # Add state labels for extreme values
        for _, row in results_df.nlargest(5, 'lambda_total_mean').iterrows():
            ax.annotate(
                row['state'],
                (row['lambda_per_county_mean'], row['lambda_total_mean']),
                fontsize=8,
                alpha=0.7
            )
        
        ax.set_xlabel('Average λ per County', fontsize=11)
        ax.set_ylabel('Total State λ', fontsize=11)
        ax.set_title('Total vs Per-County Frequency\n(Size = # Counties)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(results_df['n_counties'])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Number of Counties', fontsize=10)
        
        # Plot 4: Distribution comparison
        ax = axes[1, 1]
        ax.hist(results_df['lambda_total_mean'], bins=20, alpha=0.5, 
                label='Total λ', edgecolor='black')
        ax.hist(results_df['lambda_per_county_mean'], bins=20, alpha=0.5,
                label='Per-County λ', edgecolor='black', color='green')
        ax.set_xlabel('λ (events/year)', fontsize=11)
        ax.set_ylabel('Number of States', fontsize=11)
        ax.set_title('Distribution: Total vs Per-County', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path / 'state_comparison_county_model.png', 
                   dpi=300, bbox_inches='tight')
        print(f"  Saved: state_comparison_county_model.png")
        plt.close()
        
        return self
    
    def save_results(self, results_df, output_dir='./results'):
        """Save state-level aggregated results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save main results
        output_file = output_path / 'state_level_from_county_model.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")
        
        # Save summary
        summary_file = output_path / 'state_comparison_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("STATE-LEVEL RESULTS FROM COUNTY MODEL\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total states: {len(results_df)}\n")
            f.write(f"Total counties: {results_df['n_counties'].sum()}\n\n")
            
            f.write("COMPARISON: Total vs Per-County Lambda\n")
            f.write("-" * 60 + "\n\n")
            
            f.write("Top 10 by TOTAL lambda (reflects state size):\n")
            for _, row in results_df.nlargest(10, 'lambda_total_mean').iterrows():
                f.write(f"  {row['state']:20s}: {row['lambda_total_mean']:8.1f} "
                       f"({row['n_counties']:3d} counties)\n")
            
            f.write("\nTop 10 by PER-COUNTY lambda (comparable across states):\n")
            for _, row in results_df.nlargest(10, 'lambda_per_county_mean').iterrows():
                f.write(f"  {row['state']:20s}: {row['lambda_per_county_mean']:6.2f} "
                       f"events/county/year\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("KEY INSIGHT:\n")
            f.write("  - Total λ: Use for absolute risk assessment\n")
            f.write("  - Per-County λ: Use for fair inter-state comparison\n")
        
        print(f"Saved summary to: {summary_file}")
        
        return self
    
    def run_full_pipeline(self, output_dir='./results', 
                         n_samples=1000, n_tune=1000):
        """Run complete county-to-state pipeline."""
        print("=" * 60)
        print("COUNTY-LEVEL MODEL WITH STATE AGGREGATION")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_and_prepare_data()
        
        # Step 2: County aggregation
        self.aggregate_to_county_level()
        
        # Step 3: Build model
        self.build_hierarchical_model()
        
        # Step 4: Sample (this is slow!)
        self.sample_posterior(n_samples=n_samples, n_tune=n_tune)
        
        # Step 5: Aggregate to state
        results_df = self.aggregate_to_state_level()
        
        # Step 6: Plot
        self.plot_state_comparison(results_df, output_dir)
        
        # Step 7: Save
        self.save_results(results_df, output_dir)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        
        return results_df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='County-level hierarchical model with state aggregation'
    )
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to cleaned data CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./county_model_results',
        help='Output directory'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of MCMC samples (default: 1000, use 500 for quick test)'
    )
    parser.add_argument(
        '--tune',
        type=int,
        default=1000,
        help='Number of tuning samples'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    modeler = CountyStateFrequencyModeler(args.data_path)
    results = modeler.run_full_pipeline(
        output_dir=args.output_dir,
        n_samples=args.samples,
        n_tune=args.tune
    )
    
    print(f"\nResults preview:")
    print(results[['state', 'lambda_total_mean', 'lambda_per_county_mean']].head(10))


if __name__ == '__main__':
    main()
