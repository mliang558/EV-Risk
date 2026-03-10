#!/usr/bin/env python3
"""
Step 2B: County-Level Severity Modeling (修正版)
================================================

层级 Logistic 回归：County → State → Global

Model:
------
logit(P_severe_ij) = α_state[j] + β₁·duration + β₂·customers_pct + ε_county[i]

α_state[j] ~ Normal(μ_global, σ_state)
β₁, β₂ ~ Normal(0, 10)
ε_county ~ Normal(0, σ_county)

Severity 定义（多阈值敏感性分析）:
---------------------------------
1. Duration > [12, 24, 48] hours
2. Customer impact > [2%, 5%, 10%] of county population
3. Logic: Duration AND (Absolute OR Relative)
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


class CountySeverityModeler:
    """
    County-level severity modeling with hierarchical structure.
    """
    
    def __init__(self, data_path, population_path):
        """
        Parameters:
        -----------
        data_path : str
            Path to cleaned outage data
            Required: fips, state, county, start_time, duration_min, customers_affected
        population_path : str
            Path to county population data
            Required: STATE, COUNTY, POPESTIMATE2022
        """
        self.data_path = data_path
        self.population_path = population_path
        self.df = None
        self.county_results = None
        self.trace = None
        
    def load_and_prepare_data(self):
        """Load and merge data."""
        print("="*80)
        print("STEP 2B: COUNTY-LEVEL SEVERITY MODELING (HIERARCHICAL)")
        print("="*80)
        
        # Load outage data
        print("\n[1/4] Loading outage data...")
        self.df = pd.read_csv(self.data_path)
        print(f"  Loaded {len(self.df):,} outage events")
        print(f"  Counties: {self.df['fips'].nunique()}")
        print(f"  States: {self.df['state'].nunique()}")
        
        # Convert and clean
        self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        self.df['duration_hr'] = self.df['duration_min'] / 60
        self.df['fips'] = self.df['fips'].astype(str).str.zfill(5)
        
        # Load population
        print("\n[2/4] Loading population data...")
        pop = pd.read_csv(self.population_path, encoding='latin-1')
        pop['STATE'] = pop['STATE'].astype(str).str.zfill(2)
        pop['COUNTY'] = pop['COUNTY'].astype(str).str.zfill(3)
        pop['fips'] = pop['STATE'] + pop['COUNTY']
        
        print(f"  Population for {pop['fips'].nunique()} counties")
        
        # Merge
        print("\n[3/4] Merging data...")
        self.df = self.df.merge(
            pop[['fips', 'POPESTIMATE2022']],
            on='fips',
            how='left'
        )
        
        # Drop missing
        missing = self.df['POPESTIMATE2022'].isna().sum()
        if missing > 0:
            print(f"  ⚠️  Dropping {missing} events without population data")
            self.df = self.df.dropna(subset=['POPESTIMATE2022'])
        
        # Calculate metrics
        self.df['customers_pct'] = (
            self.df['customers_affected'] / self.df['POPESTIMATE2022']
        ).clip(0, 1)
        
        print(f"\n[4/4] Data prepared:")
        print(f"  Final events: {len(self.df):,}")
        print(f"  Counties: {self.df['fips'].nunique()}")
        print(f"  States: {self.df['state'].nunique()}")
        
        return self
    
    def define_severity_thresholds(self,
                                   duration_thresholds=[12, 24, 48],
                                   customer_abs_thresholds=[1000, 5000, 10000],
                                   customer_pct_thresholds=[0.02, 0.05, 0.10]):
        """
        Define severity using multiple threshold combinations.
        
        Severity logic:
        ---------------
        severe = (duration > T_dur) AND (
            (customers_abs > T_abs) OR (customers_pct > T_pct)
        )
        
        Parameters:
        -----------
        duration_thresholds : list
            Hours [12, 24, 48]
        customer_abs_thresholds : list
            Absolute customers [1000, 5000, 10000]
        customer_pct_thresholds : list
            Percentage of county [0.02, 0.05, 0.10]
        """
        print("\n" + "="*80)
        print("DEFINING SEVERITY (MULTIPLE THRESHOLDS)")
        print("="*80)
        
        self.severity_definitions = []
        
        for dur in duration_thresholds:
            for abs_thresh in customer_abs_thresholds:
                for pct_thresh in customer_pct_thresholds:
                    
                    # Define severity
                    col_name = f'severe_d{dur}_a{abs_thresh}_p{int(pct_thresh*100)}'
                    
                    self.df[col_name] = (
                        (self.df['duration_hr'] > dur) &
                        (
                            (self.df['customers_affected'] > abs_thresh) |
                            (self.df['customers_pct'] > pct_thresh)
                        )
                    ).astype(int)
                    
                    n_severe = self.df[col_name].sum()
                    pct = self.df[col_name].mean() * 100
                    
                    self.severity_definitions.append({
                        'name': col_name,
                        'duration_hr': dur,
                        'customer_abs': abs_thresh,
                        'customer_pct': pct_thresh,
                        'n_severe': n_severe,
                        'pct_severe': pct
                    })
                    
                    print(f"  {col_name:30s}: {n_severe:6,} ({pct:5.2f}%)")
        
        # Save definitions
        self.severity_def_df = pd.DataFrame(self.severity_definitions)
        
        return self
    
    def aggregate_to_county_level(self, severity_col='severe_d24_a5000_p5'):
        """
        Aggregate to county level for modeling.
        
        Parameters:
        -----------
        severity_col : str
            Which severity definition to use
        """
        print(f"\n" + "="*80)
        print(f"AGGREGATING TO COUNTY LEVEL")
        print(f"Using: {severity_col}")
        print("="*80)
        
        self.severity_col = severity_col
        
        # Aggregate
        county_agg = self.df.groupby('fips').agg({
            'state': 'first',
            'county': 'first',
            'fips': 'count',  # n_events
            severity_col: 'sum',  # n_severe
            'duration_hr': 'mean',
            'customers_pct': 'mean',
            'customers_affected': 'mean',
            'POPESTIMATE2022': 'first'
        }).reset_index(drop=True)
        
        county_agg.columns = [
            'fips', 'state', 'county', 'n_events', 'n_severe',
            'mean_duration_hr', 'mean_customers_pct', 'mean_customers_abs',
            'population'
        ]
        
        # Calculate raw P(severe)
        county_agg['p_severe_raw'] = (
            county_agg['n_severe'] / county_agg['n_events']
        )
        
        # Filter: keep counties with at least 5 events
        county_agg = county_agg[county_agg['n_events'] >= 5].copy()
        
        print(f"\n  Counties with ≥5 events: {len(county_agg)}")
        print(f"  Total events: {county_agg['n_events'].sum():,}")
        print(f"  Total severe: {county_agg['n_severe'].sum():,}")
        print(f"  Overall P(severe): {county_agg['n_severe'].sum() / county_agg['n_events'].sum():.3f}")
        
        self.county_agg = county_agg
        
        return self
    
    def build_hierarchical_model(self):
        """
        Build hierarchical logistic regression.
        
        Model:
        ------
        n_severe ~ Binomial(n_events, p_county)
        logit(p_county) = α_state + β₁·duration + β₂·customers_pct
        
        α_state ~ Normal(μ_global, σ_state)
        β₁, β₂ ~ Normal(0, 5)
        """
        print("\n" + "="*80)
        print("BUILDING HIERARCHICAL MODEL")
        print("="*80)
        
        # Prepare data
        df = self.county_agg.copy()
        
        # State index
        states = df['state'].unique()
        state_idx_map = {s: i for i, s in enumerate(states)}
        df['state_idx'] = df['state'].map(state_idx_map)
        
        # Standardize predictors
        df['duration_z'] = (df['mean_duration_hr'] - df['mean_duration_hr'].mean()) / df['mean_duration_hr'].std()
        df['customers_z'] = (df['mean_customers_pct'] - df['mean_customers_pct'].mean()) / df['mean_customers_pct'].std()
        
        print(f"  Counties: {len(df)}")
        print(f"  States: {len(states)}")
        print(f"  Total events: {df['n_events'].sum():,}")
        print(f"  Total severe: {df['n_severe'].sum():,}")
        
        # Build PyMC model
        print("\n  Building PyMC model...")
        
        with pm.Model() as model:
            # Data
            state_idx = pm.Data('state_idx', df['state_idx'].values, mutable=False)
            n_events = pm.Data('n_events', df['n_events'].values, mutable=False)
            n_severe = pm.Data('n_severe', df['n_severe'].values, mutable=False)
            duration_z = pm.Data('duration_z', df['duration_z'].values, mutable=False)
            customers_z = pm.Data('customers_z', df['customers_z'].values, mutable=False)
            
            # Hyperpriors
            μ_global = pm.Normal('μ_global', mu=0, sigma=2)
            σ_state = pm.HalfNormal('σ_state', sigma=1)
            
            # State-level intercepts
            α_state = pm.Normal('α_state', mu=μ_global, sigma=σ_state, shape=len(states))
            
            # Slopes
            β_duration = pm.Normal('β_duration', mu=0, sigma=2)
            β_customers = pm.Normal('β_customers', mu=0, sigma=2)
            
            # Linear predictor
            logit_p = (
                α_state[state_idx] +
                β_duration * duration_z +
                β_customers * customers_z
            )
            
            # Likelihood
            p_severe = pm.Deterministic('p_severe', pm.math.invlogit(logit_p))
            
            likelihood = pm.Binomial(
                'obs',
                n=n_events,
                p=p_severe,
                observed=n_severe
            )
        
        self.model = model
        self.state_idx_map = state_idx_map
        self.states = states
        
        print("  ✓ Model built")
        
        return self
    
    def sample_posterior(self, draws=2000, tune=1000, chains=4):
        """Sample from posterior."""
        print("\n" + "="*80)
        print("SAMPLING POSTERIOR")
        print("="*80)
        print(f"  Draws: {draws}")
        print(f"  Tune: {tune}")
        print(f"  Chains: {chains}")
        
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                return_inferencedata=True,
                random_seed=42
            )
        
        print("\n  ✓ Sampling complete")
        
        # Diagnostics
        print("\n  Diagnostics:")
        summary = az.summary(self.trace, var_names=['μ_global', 'σ_state', 'β_duration', 'β_customers'])
        print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']])
        
        return self
    
    def extract_county_results(self):
        """Extract county-level P(severe) estimates."""
        print("\n" + "="*80)
        print("EXTRACTING COUNTY RESULTS")
        print("="*80)
        
        # Get posterior samples
        p_severe_samples = self.trace.posterior['p_severe'].values
        
        # Shape: (chains, draws, counties)
        p_severe_samples = p_severe_samples.reshape(-1, p_severe_samples.shape[-1])
        
        # Calculate statistics
        results = self.county_agg.copy()
        results['p_severe_mean'] = p_severe_samples.mean(axis=0)
        results['p_severe_sd'] = p_severe_samples.std(axis=0)
        results['p_severe_lower'] = np.percentile(p_severe_samples, 2.5, axis=0)
        results['p_severe_upper'] = np.percentile(p_severe_samples, 97.5, axis=0)
        
        # Shrinkage
        results['shrinkage'] = np.abs(results['p_severe_mean'] - results['p_severe_raw']) / (results['p_severe_raw'] + 1e-6)
        
        # Sort by P(severe)
        results = results.sort_values('p_severe_mean', ascending=False).reset_index(drop=True)
        
        print(f"\n  Counties: {len(results)}")
        print(f"\n  Top 10 riskiest counties:")
        print(results[['fips', 'state', 'county', 'n_events', 'p_severe_mean', 'p_severe_lower', 'p_severe_upper']].head(10))
        
        self.county_results = results
        
        return results
    
    def run_sensitivity_analysis(self, output_dir='./severity_results'):
        """
        Run analysis for all severity definitions.
        """
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS: ALL THRESHOLDS")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        sensitivity_results = []
        
        for i, sev_def in enumerate(self.severity_definitions):
            print(f"\n[{i+1}/{len(self.severity_definitions)}] {sev_def['name']}")
            print("-" * 80)
            
            # Aggregate for this definition
            self.aggregate_to_county_level(severity_col=sev_def['name'])
            
            # Build model
            self.build_hierarchical_model()
            
            # Sample (fewer draws for speed)
            self.sample_posterior(draws=1000, tune=500, chains=2)
            
            # Extract results
            results = self.extract_county_results()
            
            # Save
            results.to_csv(output_path / f"county_severity_{sev_def['name']}.csv", index=False)
            
            # Aggregate to state level for comparison
            state_summary = results.groupby('state').agg({
                'n_events': 'sum',
                'n_severe': 'sum',
                'p_severe_mean': 'mean',
                'p_severe_sd': 'mean'
            }).reset_index()
            
            state_summary['threshold'] = sev_def['name']
            state_summary['duration_hr'] = sev_def['duration_hr']
            state_summary['customer_abs'] = sev_def['customer_abs']
            state_summary['customer_pct'] = sev_def['customer_pct']
            
            sensitivity_results.append(state_summary)
        
        # Combine all
        sensitivity_df = pd.concat(sensitivity_results, ignore_index=True)
        sensitivity_df.to_csv(output_path / 'sensitivity_all_thresholds.csv', index=False)
        
        print(f"\n✓ Sensitivity analysis complete")
        print(f"  Saved: {output_path / 'sensitivity_all_thresholds.csv'}")
        
        return sensitivity_df
    
    def plot_sensitivity_results(self, sensitivity_df, output_dir='./severity_results'):
        """
        Plot sensitivity analysis results.
        """
        print("\n" + "="*80)
        print("PLOTTING SENSITIVITY RESULTS")
        print("="*80)
        
        output_path = Path(output_dir)
        
        # Select top 10 states by total events
        top_states = (
            sensitivity_df.groupby('state')['n_events']
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .index
        )
        
        plot_df = sensitivity_df[sensitivity_df['state'].isin(top_states)]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: P(severe) by duration threshold
        ax = axes[0, 0]
        for state in top_states[:5]:
            state_data = plot_df[plot_df['state'] == state]
            grouped = state_data.groupby('duration_hr')['p_severe_mean'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', label=state, linewidth=2)
        
        ax.set_xlabel('Duration Threshold (hours)', fontsize=12)
        ax.set_ylabel('Mean P(severe)', fontsize=12)
        ax.set_title('Sensitivity to Duration Threshold\n(Top 5 States)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: P(severe) by customer % threshold
        ax = axes[0, 1]
        for state in top_states[:5]:
            state_data = plot_df[plot_df['state'] == state]
            grouped = state_data.groupby('customer_pct')['p_severe_mean'].mean()
            ax.plot(grouped.index * 100, grouped.values, marker='s', label=state, linewidth=2)
        
        ax.set_xlabel('Customer % Threshold', fontsize=12)
        ax.set_ylabel('Mean P(severe)', fontsize=12)
        ax.set_title('Sensitivity to Customer Impact Threshold\n(Top 5 States)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 3: Heatmap of one state
        ax = axes[1, 0]
        pivot_data = plot_df[plot_df['state'] == top_states[0]].pivot_table(
            index='duration_hr',
            columns='customer_pct',
            values='p_severe_mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
        ax.set_title(f'P(severe) Heatmap: {top_states[0]}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Customer % Threshold')
        ax.set_ylabel('Duration Threshold (hours)')
        
        # Plot 4: Ranking stability
        ax = axes[1, 1]
        
        # For each threshold, rank states
        rankings = []
        for threshold in plot_df['threshold'].unique():
            subset = plot_df[plot_df['threshold'] == threshold].sort_values('p_severe_mean', ascending=False)
            subset['rank'] = range(1, len(subset) + 1)
            rankings.append(subset[['state', 'rank', 'threshold']])
        
        rank_df = pd.concat(rankings)
        
        # Plot rank variation for top states
        for state in top_states[:5]:
            state_ranks = rank_df[rank_df['state'] == state]['rank']
            ax.scatter([state] * len(state_ranks), state_ranks, alpha=0.5, s=50)
        
        ax.set_xlabel('State', fontsize=12)
        ax.set_ylabel('Rank (1 = Highest P(severe))', fontsize=12)
        ax.set_title('Ranking Stability Across Thresholds', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plot_file = output_path / 'sensitivity_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_file}")
        plt.close()
        
        return self


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='County-level severity modeling')
    parser.add_argument('data_path', help='Path to outage data CSV')
    parser.add_argument('population_path', help='Path to population CSV')
    parser.add_argument('--output-dir', default='./severity_results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test (one threshold only)')
    
    args = parser.parse_args()
    
    # Initialize
    modeler = CountySeverityModeler(args.data_path, args.population_path)
    
    # Load data
    modeler.load_and_prepare_data()
    
    # Define thresholds
    modeler.define_severity_thresholds(
        duration_thresholds=[12, 24, 48],
        customer_abs_thresholds=[1000, 5000, 10000],
        customer_pct_thresholds=[0.02, 0.05, 0.10]
    )
    
    if args.quick:
        # Quick test: just one threshold
        print("\n🚀 QUICK MODE: Testing one threshold only")
        modeler.aggregate_to_county_level(severity_col='severe_d24_a5000_p5')
        modeler.build_hierarchical_model()
        modeler.sample_posterior(draws=500, tune=250, chains=2)
        results = modeler.extract_county_results()
        results.to_csv(f'{args.output_dir}/county_severity_quick.csv', index=False)
    else:
        # Full sensitivity analysis
        print("\n🔥 FULL MODE: Testing all threshold combinations")
        sensitivity_df = modeler.run_sensitivity_analysis(args.output_dir)
        modeler.plot_sensitivity_results(sensitivity_df, args.output_dir)
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
