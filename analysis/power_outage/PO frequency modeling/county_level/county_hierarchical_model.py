#!/usr/bin/env python3
"""
County-Level Hierarchical Bayesian Model with State-Level Pooling
=================================================================
Two-level hierarchy: State → County
- Counties borrow strength from their state
- No national level (to keep computation tractable)
- Suitable for ~3000 counties across 50 states
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


def prepare_county_data(data_path, filter_states=None):
    """
    Prepare county-level data with state grouping.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file
    filter_states : list, optional
        List of state names/abbreviations to analyze (e.g., ['TX', 'CA', 'FL'])
        If None, analyze all states
    """
    print("="*70)
    print("PREPARING COUNTY-LEVEL DATA")
    print("="*70)
    
    df = pd.read_csv(data_path)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['year'] = df['start_time'].dt.year
    
    # Filter states if specified
    if filter_states is not None:
        print(f"\nFiltering to states: {filter_states}")
        df = df[df['state'].isin(filter_states)]
        print(f"Rows after filtering: {len(df)}")
    
    # County-level aggregation
    county_agg = df.groupby(['state', 'fips']).agg(
        n_events=('fips', 'count'),
        min_year=('year', 'min'),
        max_year=('year', 'max')
    ).reset_index()
    
    county_agg['n_years'] = county_agg['max_year'] - county_agg['min_year'] + 1
    county_agg['lambda_raw'] = county_agg['n_events'] / county_agg['n_years']
    
    # Create state index mapping
    states = sorted(county_agg['state'].unique())
    state_to_idx = {state: i for i, state in enumerate(states)}
    county_agg['state_idx'] = county_agg['state'].map(state_to_idx)
    
    # Add county index
    county_agg['county_idx'] = range(len(county_agg))
    
    print(f"\nData summary:")
    print(f"  States: {len(states)}")
    print(f"  Counties: {len(county_agg)}")
    print(f"  Total events: {county_agg['n_events'].sum()}")
    print(f"  Events per county: {county_agg['n_events'].mean():.1f} ± {county_agg['n_events'].std():.1f}")
    
    # Show data sparsity
    sparse = (county_agg['n_events'] < 5).sum()
    print(f"\n  Sparse counties (n < 5): {sparse} ({100*sparse/len(county_agg):.1f}%)")
    print(f"  → These will benefit most from state-level pooling!")
    
    return county_agg, state_to_idx


def build_hierarchical_model(county_data):
    """
    Build two-level hierarchical Poisson model: State → County
    """
    print("\n" + "="*70)
    print("BUILDING HIERARCHICAL MODEL")
    print("="*70)
    
    n_states = county_data['state_idx'].max() + 1
    n_counties = len(county_data)
    
    Y_obs = county_data['n_events'].values.astype(int)
    T_obs = county_data['n_years'].values.astype(float)
    state_idx = county_data['state_idx'].values.astype(int)
    
    # Informative priors from data
    log_lambda_raw = np.log(county_data['lambda_raw'] + 0.1)
    mu_data = log_lambda_raw.mean()
    sigma_data = log_lambda_raw.std()
    
    print(f"\nModel structure:")
    print(f"  Top level: {n_states} states")
    print(f"  Bottom level: {n_counties} counties")
    print(f"\nPrior calibration (from data):")
    print(f"  log(λ) mean: {mu_data:.2f}")
    print(f"  log(λ) std: {sigma_data:.2f}")
    
    with pm.Model() as model:
        # ============================================================
        # STATE LEVEL (top level)
        # ============================================================
        # Each state has its own baseline log(λ)
        mu_state = pm.Normal(
            "mu_state",
            mu=mu_data,
            sigma=sigma_data * 2,  # Wide prior
            shape=n_states
        )
        
        # ============================================================
        # COUNTY LEVEL (bottom level)
        # ============================================================
        # How much do counties vary within their state?
        sigma_county = pm.HalfNormal("sigma_county", sigma=sigma_data)
        
        # Each county borrows from its state's mean
        log_lambda_county = pm.Normal(
            "log_lambda_county",
            mu=mu_state[state_idx],  # County inherits state's baseline
            sigma=sigma_county,
            shape=n_counties
        )
        
        # Transform to rate scale
        lambda_county = pm.Deterministic(
            "lambda_county",
            pm.math.exp(log_lambda_county)
        )
        
        # ============================================================
        # LIKELIHOOD
        # ============================================================
        pm.Poisson(
            "Y_obs",
            mu=lambda_county * T_obs,
            observed=Y_obs
        )
        
        # ============================================================
        # DERIVED QUANTITIES
        # ============================================================
        # State-level average λ (for comparison)
        lambda_state = pm.Deterministic(
            "lambda_state",
            pm.math.exp(mu_state)
        )
    
    print("\nModel built successfully!")
    print(f"  Total parameters: ~{n_states + n_counties + 1}")
    
    return model


def sample_model(model, draws=1000, tune=1000):
    """
    Sample from the posterior.
    """
    print("\n" + "="*70)
    print("SAMPLING POSTERIOR")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Draws: {draws}")
    print(f"  Tune: {tune}")
    print(f"  Target accept: 0.95")
    
    print("\nSampling (this may take 10-30 minutes for ~3000 counties)...")
    print("Progress:")
    
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            cores=4,  # Use multiple cores
            target_accept=0.95,
            return_inferencedata=True,
            random_seed=42,
            progressbar=True
        )
    
    print("\n✓ Sampling complete!")
    
    # Diagnostics
    print("\nConvergence diagnostics:")
    rhat = az.rhat(trace)
    max_rhat = max(
        rhat['mu_state'].max().values,
        rhat['sigma_county'].values,
        rhat['log_lambda_county'].max().values
    )
    print(f"  Max R̂: {max_rhat:.4f} (should be < 1.01)")
    
    if max_rhat > 1.01:
        print("  ⚠ WARNING: Some parameters have R̂ > 1.01. Consider more tuning.")
    else:
        print("  ✓ Excellent convergence!")
    
    return trace


def extract_results(trace, county_data):
    """
    Extract posterior summaries for counties and states.
    """
    print("\n" + "="*70)
    print("EXTRACTING RESULTS")
    print("="*70)
    
    # County-level results
    lambda_samples = trace.posterior['lambda_county'].values
    lambda_samples_flat = lambda_samples.reshape(-1, lambda_samples.shape[-1])
    
    county_results = []
    for i, row in county_data.iterrows():
        samples = lambda_samples_flat[:, i]
        
        county_results.append({
            'state': row['state'],
            'fips': row['fips'],
            'state_idx': row['state_idx'],
            'n_events': row['n_events'],
            'n_years': row['n_years'],
            'lambda_raw': row['lambda_raw'],
            
            # Posterior estimates
            'lambda_mean': samples.mean(),
            'lambda_median': np.median(samples),
            'lambda_sd': samples.std(),
            'lambda_ci_lower': np.percentile(samples, 2.5),
            'lambda_ci_upper': np.percentile(samples, 97.5),
            
            # Pooling metrics
            'shrinkage': abs(samples.mean() - row['lambda_raw']) / (row['lambda_raw'] + 1e-6),
            'uncertainty_reduction': 1 - samples.std() / row['lambda_raw'] if row['lambda_raw'] > 0 else 0
        })
    
    county_df = pd.DataFrame(county_results)
    
    # State-level results
    state_samples = trace.posterior['lambda_state'].values
    state_samples_flat = state_samples.reshape(-1, state_samples.shape[-1])
    
    state_results = []
    for state_idx, state in enumerate(sorted(county_data['state'].unique())):
        samples = state_samples_flat[:, state_idx]
        
        state_results.append({
            'state': state,
            'state_idx': state_idx,
            'n_counties': (county_data['state'] == state).sum(),
            'lambda_state_mean': samples.mean(),
            'lambda_state_median': np.median(samples),
            'lambda_state_sd': samples.std(),
            'lambda_state_ci_lower': np.percentile(samples, 2.5),
            'lambda_state_ci_upper': np.percentile(samples, 97.5),
        })
    
    state_df = pd.DataFrame(state_results)
    
    print(f"\n✓ Extracted {len(county_df)} county estimates")
    print(f"✓ Extracted {len(state_df)} state estimates")
    
    return county_df, state_df


def calculate_pooling_diagnostics(county_df, trace):
    """
    Calculate ICC and other pooling diagnostics.
    """
    print("\n" + "="*70)
    print("POOLING DIAGNOSTICS")
    print("="*70)
    
    # Variance components
    mu_state_samples = trace.posterior['mu_state'].values.reshape(-1, trace.posterior['mu_state'].shape[-1])
    sigma_county_samples = trace.posterior['sigma_county'].values.flatten()
    
    # Between-state variance
    var_between = mu_state_samples.var(axis=1).mean()
    
    # Within-state variance
    var_within = (sigma_county_samples ** 2).mean()
    
    # ICC (Intraclass Correlation Coefficient)
    icc = var_between / (var_between + var_within)
    
    print(f"\nVariance decomposition:")
    print(f"  Between-state variance: {var_between:.4f}")
    print(f"  Within-state variance: {var_within:.4f}")
    print(f"  ICC: {icc:.3f}")
    
    if icc > 0.1:
        print(f"  ✓ ICC > 0.1: State-level grouping is meaningful!")
    else:
        print(f"  ⚠ ICC < 0.1: Counties are very heterogeneous within states")
    
    # Pooling strength by data sparsity
    print(f"\nPooling effect by data density:")
    
    bins = [0, 5, 10, 20, np.inf]
    labels = ['Very sparse (0-4)', 'Sparse (5-9)', 'Medium (10-19)', 'Dense (20+)']
    county_df['density_bin'] = pd.cut(county_df['n_events'], bins=bins, labels=labels)
    
    for label in labels:
        subset = county_df[county_df['density_bin'] == label]
        if len(subset) > 0:
            avg_shrinkage = subset['shrinkage'].mean()
            print(f"  {label:20s}: {len(subset):4d} counties, avg shrinkage {avg_shrinkage:.1%}")
    
    return icc


def create_diagnostic_plots(county_df, state_df, output_dir):
    """
    Create comprehensive diagnostic and result plots.
    """
    print("\n" + "="*70)
    print("CREATING PLOTS")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # ============================================================
    # Plot 1: Pooling effect
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    ax.scatter(county_df['lambda_raw'], county_df['lambda_mean'], 
              alpha=0.3, s=20, c=county_df['n_events'], cmap='viridis')
    ax.plot([0, county_df['lambda_raw'].max()], [0, county_df['lambda_raw'].max()], 
           'r--', alpha=0.5, label='No pooling')
    ax.set_xlabel('Raw estimate (MLE)', fontsize=11)
    ax.set_ylabel('Bayesian estimate (pooled)', fontsize=11)
    ax.set_title('Pooling Effect on County Estimates', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ============================================================
    # Plot 2: Shrinkage by sample size
    # ============================================================
    ax = axes[0, 1]
    ax.scatter(county_df['n_events'], county_df['shrinkage'], alpha=0.3, s=20)
    ax.set_xlabel('Number of events', fontsize=11)
    ax.set_ylabel('Shrinkage (relative change)', fontsize=11)
    ax.set_title('Shrinkage vs Sample Size\n(Sparse data → More shrinkage)', 
                fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    
    # ============================================================
    # Plot 3: Top states
    # ============================================================
    ax = axes[1, 0]
    top_states = state_df.nlargest(15, 'lambda_state_mean').sort_values('lambda_state_mean')
    y_pos = np.arange(len(top_states))
    ax.barh(y_pos, top_states['lambda_state_mean'], alpha=0.7)
    ax.errorbar(top_states['lambda_state_mean'], y_pos,
               xerr=[top_states['lambda_state_mean'] - top_states['lambda_state_ci_lower'],
                     top_states['lambda_state_ci_upper'] - top_states['lambda_state_mean']],
               fmt='none', ecolor='black', alpha=0.5, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_states['state'])
    ax.set_xlabel('State-level λ (events/county/year)', fontsize=11)
    ax.set_title('Top 15 States by Average County Risk', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    # ============================================================
    # Plot 4: Uncertainty comparison
    # ============================================================
    ax = axes[1, 1]
    county_df['cv_bayes'] = county_df['lambda_sd'] / county_df['lambda_mean']
    county_df['cv_raw'] = np.sqrt(county_df['lambda_raw']) / county_df['lambda_raw']
    
    valid = (county_df['cv_raw'] < 5) & (county_df['cv_bayes'] < 5)  # Remove outliers
    ax.scatter(county_df.loc[valid, 'cv_raw'], 
              county_df.loc[valid, 'cv_bayes'],
              alpha=0.3, s=20)
    max_cv = min(county_df.loc[valid, 'cv_raw'].max(), 
                county_df.loc[valid, 'cv_bayes'].max())
    ax.plot([0, max_cv], [0, max_cv], 'r--', alpha=0.5, label='No change')
    ax.set_xlabel('Raw CV (uncertainty)', fontsize=11)
    ax.set_ylabel('Bayesian CV (uncertainty)', fontsize=11)
    ax.set_title('Uncertainty Reduction via Pooling', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_path / 'county_hierarchical_diagnostics.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_file}")
    plt.close()
    
    # ============================================================
    # Plot 5: Top risky counties
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axes[0]
    top_counties = county_df.nlargest(20, 'lambda_mean').sort_values('lambda_mean')
    y_pos = np.arange(len(top_counties))
    ax.barh(y_pos, top_counties['lambda_mean'], alpha=0.7, color='red')
    ax.errorbar(top_counties['lambda_mean'], y_pos,
               xerr=[top_counties['lambda_mean'] - top_counties['lambda_ci_lower'],
                     top_counties['lambda_ci_upper'] - top_counties['lambda_mean']],
               fmt='none', ecolor='black', alpha=0.5, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['state']}-{row['fips']}" for _, row in top_counties.iterrows()], 
                       fontsize=8)
    ax.set_xlabel('County-level λ (events/year)', fontsize=11)
    ax.set_title('Top 20 Highest Risk Counties', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    ax = axes[1]
    # Compare with raw estimates
    ax.scatter(top_counties['lambda_raw'], top_counties['lambda_mean'], s=100, alpha=0.6)
    for _, row in top_counties.iterrows():
        ax.annotate(f"{row['state']}", 
                   (row['lambda_raw'], row['lambda_mean']),
                   fontsize=7, alpha=0.7)
    max_val = max(top_counties['lambda_raw'].max(), top_counties['lambda_mean'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='No pooling')
    ax.set_xlabel('Raw estimate', fontsize=11)
    ax.set_ylabel('Bayesian estimate', fontsize=11)
    ax.set_title('Top 20: Raw vs Pooled Estimates', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_path / 'top_risky_counties.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_file}")
    plt.close()


def save_results(county_df, state_df, output_dir):
    """
    Save results to CSV files.
    """
    output_path = Path(output_dir)
    
    county_file = output_path / 'county_estimates.csv'
    county_df.to_csv(county_file, index=False)
    print(f"✓ Saved: {county_file}")
    
    state_file = output_path / 'state_estimates.csv'
    state_df.to_csv(state_file, index=False)
    print(f"✓ Saved: {state_file}")


def print_summary(county_df, state_df, icc):
    """
    Print final summary.
    """
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Model Performance:")
    print(f"  ICC (state clustering): {icc:.3f}")
    print(f"  Average pooling shrinkage: {county_df['shrinkage'].mean():.1%}")
    
    print(f"\n🏆 Top 5 States (by average county risk):")
    for _, row in state_df.nlargest(5, 'lambda_state_mean').iterrows():
        print(f"  {row['state']:15s}: λ = {row['lambda_state_mean']:5.2f} "
              f"[{row['lambda_state_ci_lower']:.2f}, {row['lambda_state_ci_upper']:.2f}] "
              f"({row['n_counties']} counties)")
    
    print(f"\n⚠️  Top 5 Riskiest Counties:")
    for _, row in county_df.nlargest(5, 'lambda_mean').iterrows():
        print(f"  {row['state']}-{row['fips']}: λ = {row['lambda_mean']:5.2f} "
              f"[{row['lambda_ci_lower']:.2f}, {row['lambda_ci_upper']:.2f}] "
              f"({row['n_events']} events in {row['n_years']} years)")
    
    print(f"\n💡 Interpretation:")
    print(f"  - λ = expected events per county per year")
    print(f"  - Counties with sparse data borrow strength from their state")
    print(f"  - Credible intervals reflect both data uncertainty and state-level variation")


def run_county_hierarchical_model(data_path, output_dir='./county_results', 
                                  draws=1000, tune=1000, filter_states=None):
    """
    Main pipeline: Run complete hierarchical model.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with columns: state, fips, start_time
    output_dir : str
        Directory to save results
    draws : int
        Number of MCMC draws
    tune : int
        Number of tuning iterations
    filter_states : list, optional
        List of states to analyze (e.g., ['TX', 'CA', 'FL'])
        If None, analyze all states
    """
    # Step 1: Prepare data
    county_data, state_to_idx = prepare_county_data(data_path, filter_states=filter_states)
    
    # Step 2: Build model
    model = build_hierarchical_model(county_data)
    
    # Step 3: Sample
    trace = sample_model(model, draws=draws, tune=tune)
    
    # Step 4: Extract results
    county_df, state_df = extract_results(trace, county_data)
    
    # Step 5: Diagnostics
    icc = calculate_pooling_diagnostics(county_df, trace)
    
    # Step 6: Plots
    create_diagnostic_plots(county_df, state_df, output_dir)
    
    # Step 7: Save
    save_results(county_df, state_df, output_dir)
    
    # Step 8: Summary
    print_summary(county_df, state_df, icc)
    
    print(f"\n{'='*70}")
    print(f"✓ ALL DONE! Results saved to: {output_dir}/")
    print(f"{'='*70}\n")
    
    return county_df, state_df, trace


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='County-level hierarchical Bayesian model with state-level pooling'
    )
    parser.add_argument('data_path', type=str, 
                       help='Path to cleaned power outage data (CSV)')
    parser.add_argument('--output-dir', type=str, default='./county_results',
                       help='Output directory for results')
    parser.add_argument('--draws', type=int, default=1000,
                       help='Number of posterior draws')
    parser.add_argument('--tune', type=int, default=1000,
                       help='Number of tuning steps')
    parser.add_argument('--states', type=str, nargs='+', default=None,
                       help='Filter to specific states (e.g., --states TX CA FL)')
    
    args = parser.parse_args()
    
    # Run the model
    county_df, state_df, trace = run_county_hierarchical_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        draws=args.draws,
        tune=args.tune,
        filter_states=args.states
    )
