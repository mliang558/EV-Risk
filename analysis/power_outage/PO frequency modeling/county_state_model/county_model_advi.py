#!/usr/bin/env python3
"""
County-Level Model with ADVI (Fast Variational Inference)
==========================================================
Uses ADVI instead of MCMC for 10-100x speedup
Trade-off: Approximate posterior instead of exact samples

Speed comparison:
- MCMC (NUTS): 1-2 hours for 254 counties
- ADVI: 5-15 minutes for 254 counties
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def prepare_county_data(data_path, filter_states=None):
    """
    Prepare county-level data with state grouping.
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
    county_agg['county_idx'] = range(len(county_agg))
    
    print(f"\nData summary:")
    print(f"  States: {len(states)}")
    print(f"  Counties: {len(county_agg)}")
    print(f"  Total events: {county_agg['n_events'].sum()}")
    print(f"  Events per county: {county_agg['n_events'].mean():.1f} ± {county_agg['n_events'].std():.1f}")
    
    sparse = (county_agg['n_events'] < 5).sum()
    print(f"  Sparse counties (n < 5): {sparse} ({100*sparse/len(county_agg):.1f}%)")
    
    return county_agg, state_to_idx


def build_hierarchical_model(county_data):
    """
    Build two-level hierarchical Poisson model.
    """
    print("\n" + "="*70)
    print("BUILDING HIERARCHICAL MODEL")
    print("="*70)
    
    # Cast to plain Python ints to satisfy PyMC shape requirements
    n_states = int(county_data['state_idx'].max() + 1)
    n_counties = int(len(county_data))
    
    Y_obs = county_data['n_events'].values.astype(int)
    T_obs = county_data['n_years'].values.astype(float)
    state_idx = county_data['state_idx'].values.astype(int)
    
    # Informative priors from data
    log_lambda_raw = np.log(county_data['lambda_raw'] + 0.1)
    mu_data = log_lambda_raw.mean()
    sigma_data = log_lambda_raw.std()
    
    print(f"\nModel structure:")
    print(f"  States: {n_states}")
    print(f"  Counties: {n_counties}")
    print(f"  Prior calibration: μ={mu_data:.2f}, σ={sigma_data:.2f}")
    
    with pm.Model() as model:
        # State level
        mu_state = pm.Normal(
            "mu_state",
            mu=mu_data,
            sigma=sigma_data * 2,
            shape=int(n_states),
        )
        
        # County level
        sigma_county = pm.HalfNormal("sigma_county", sigma=sigma_data)
        
        log_lambda_county = pm.Normal(
            "log_lambda_county",
            mu=mu_state[state_idx],
            sigma=sigma_county,
            shape=int(n_counties),
        )
        
        lambda_county = pm.Deterministic(
            "lambda_county",
            pm.math.exp(log_lambda_county)
        )
        
        # Likelihood
        pm.Poisson(
            "Y_obs",
            mu=lambda_county * T_obs,
            observed=Y_obs
        )
        
        # State-level lambda
        lambda_state = pm.Deterministic(
            "lambda_state",
            pm.math.exp(mu_state)
        )
    
    print("✓ Model built successfully!")
    
    return model


def fit_with_advi(model, n_iterations=50000):
    """
    Fit model using ADVI (fast approximate inference).
    
    ADVI is much faster than MCMC but gives approximate posterior.
    Good for:
    - Quick exploratory analysis
    - Large models (1000+ parameters)
    - When you need results fast
    """
    print("\n" + "="*70)
    print("FITTING WITH ADVI (Variational Inference)")
    print("="*70)
    print(f"\nIterations: {n_iterations}")
    print("This should take 5-15 minutes (vs 1-2 hours for MCMC)...\n")
    
    with model:
        # Fit ADVI
        approx = pm.fit(
            n=n_iterations,
            method='advi',
            progressbar=True,
            random_seed=42
        )
    
    print("\n✓ ADVI fitting complete!")
    
    # Draw samples from approximate posterior
    print("\nDrawing samples from approximate posterior...")
    with model:
        trace = approx.sample(draws=2000)
    
    print("✓ Sampling complete!")
    
    return trace, approx


def calculate_icc(trace):
    """
    Calculate ICC from ADVI trace.
    """
    print("\n" + "="*70)
    print("POOLING DIAGNOSTICS")
    print("="*70)
    
    # Extract samples
    mu_state_samples = trace.posterior['mu_state'].values.reshape(-1, trace.posterior['mu_state'].shape[-1])
    sigma_county_samples = trace.posterior['sigma_county'].values.flatten()
    
    # Between-state variance
    var_between = mu_state_samples.var(axis=1).mean()
    
    # Within-state variance
    var_within = (sigma_county_samples ** 2).mean()
    
    # ICC
    icc = var_between / (var_between + var_within)
    
    print(f"\nVariance decomposition:")
    print(f"  Between-state variance: {var_between:.4f}")
    print(f"  Within-state variance: {var_within:.4f}")
    print(f"  ICC: {icc:.3f}")
    
    if icc > 0.1:
        print(f"  ✓ ICC > 0.1: State-level grouping is meaningful!")
    else:
        print(f"  ⚠ ICC < 0.1: Counties are very heterogeneous within states")
    
    return icc


def extract_results(trace, county_data, use_hdi=False):
    """
    Extract posterior summaries.
    
    Parameters:
    -----------
    use_hdi : bool
        If True, use HDI instead of percentile intervals
    """
    print("\n" + "="*70)
    print("EXTRACTING RESULTS")
    print("="*70)
    
    if use_hdi:
        print("Using 95% HDI (Highest Density Interval)")
    else:
        print("Using 95% Percentile Interval")
    
    # County-level results
    lambda_samples = trace.posterior['lambda_county'].values
    lambda_samples_flat = lambda_samples.reshape(-1, lambda_samples.shape[-1])
    
    county_results = []
    for i, row in county_data.iterrows():
        samples = lambda_samples_flat[:, i]
        
        if use_hdi:
            # Use ArviZ HDI
            hdi = az.hdi(samples, hdi_prob=0.95)
            ci_lower, ci_upper = hdi[0], hdi[1]
        else:
            # Use percentile
            ci_lower = np.percentile(samples, 2.5)
            ci_upper = np.percentile(samples, 97.5)
        
        county_results.append({
            'state': row['state'],
            'fips': row['fips'],
            'state_idx': row['state_idx'],
            'n_events': row['n_events'],
            'n_years': row['n_years'],
            'lambda_raw': row['lambda_raw'],
            
            'lambda_mean': samples.mean(),
            'lambda_median': np.median(samples),
            'lambda_sd': samples.std(),
            'lambda_ci_lower': ci_lower,
            'lambda_ci_upper': ci_upper,
            
            'shrinkage': abs(samples.mean() - row['lambda_raw']) / (row['lambda_raw'] + 1e-6),
        })
    
    county_df = pd.DataFrame(county_results)
    
    # State-level results
    state_samples = trace.posterior['lambda_state'].values
    state_samples_flat = state_samples.reshape(-1, state_samples.shape[-1])
    
    state_results = []
    for state_idx, state in enumerate(sorted(county_data['state'].unique())):
        samples = state_samples_flat[:, state_idx]
        
        if use_hdi:
            hdi = az.hdi(samples, hdi_prob=0.95)
            ci_lower, ci_upper = hdi[0], hdi[1]
        else:
            ci_lower = np.percentile(samples, 2.5)
            ci_upper = np.percentile(samples, 97.5)
        
        state_results.append({
            'state': state,
            'state_idx': state_idx,
            'n_counties': (county_data['state'] == state).sum(),
            'lambda_state_mean': samples.mean(),
            'lambda_state_median': np.median(samples),
            'lambda_state_sd': samples.std(),
            'lambda_state_ci_lower': ci_lower,
            'lambda_state_ci_upper': ci_upper,
        })
    
    state_df = pd.DataFrame(state_results)
    
    print(f"✓ Extracted {len(county_df)} county estimates")
    print(f"✓ Extracted {len(state_df)} state estimates")
    
    return county_df, state_df


def create_diagnostic_plots(county_df, state_df, approx, output_dir):
    """
    Create diagnostic and result plots.
    """
    print("\n" + "="*70)
    print("CREATING PLOTS")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: ADVI convergence (ELBO)
    ax = axes[0, 0]
    elbo = -approx.hist
    ax.plot(elbo, alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('ELBO', fontsize=11)
    ax.set_title('ADVI Convergence (ELBO)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Plot 2: Pooling effect
    ax = axes[0, 1]
    ax.scatter(county_df['lambda_raw'], county_df['lambda_mean'], 
              alpha=0.3, s=20, c=county_df['n_events'], cmap='viridis')
    max_val = max(county_df['lambda_raw'].max(), county_df['lambda_mean'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='No pooling')
    ax.set_xlabel('Raw estimate', fontsize=11)
    ax.set_ylabel('ADVI estimate (pooled)', fontsize=11)
    ax.set_title('Pooling Effect on County Estimates', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Top states
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
    ax.set_xlabel('State-level λ', fontsize=11)
    ax.set_title('Top States by County Risk', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    # Plot 4: Top risky counties
    ax = axes[1, 1]
    top_counties = county_df.nlargest(20, 'lambda_mean').sort_values('lambda_mean')
    y_pos = np.arange(len(top_counties))
    ax.barh(y_pos, top_counties['lambda_mean'], alpha=0.7, color='red')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['state'][:10]}-{row['fips']}" for _, row in top_counties.iterrows()], 
                       fontsize=8)
    ax.set_xlabel('County λ', fontsize=11)
    ax.set_title('Top 20 Riskiest Counties', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plot_file = output_path / 'advi_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_file}")
    plt.close()


def save_results(county_df, state_df, output_dir):
    """
    Save results to CSV.
    """
    output_path = Path(output_dir)
    
    county_file = output_path / 'county_estimates_advi.csv'
    county_df.to_csv(county_file, index=False)
    print(f"✓ Saved: {county_file}")
    
    state_file = output_path / 'state_estimates_advi.csv'
    state_df.to_csv(state_file, index=False)
    print(f"✓ Saved: {state_file}")


def print_summary(county_df, state_df):
    """
    Print summary results.
    """
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n📊 Model Summary:")
    print(f"  Method: ADVI (Variational Inference)")
    print(f"  Counties analyzed: {len(county_df)}")
    print(f"  States analyzed: {len(state_df)}")
    
    print(f"\n🏆 Top 5 States (by average county risk):")
    for _, row in state_df.nlargest(5, 'lambda_state_mean').iterrows():
        print(f"  {row['state']:20s}: λ = {row['lambda_state_mean']:5.2f} "
              f"[{row['lambda_state_ci_lower']:.2f}, {row['lambda_state_ci_upper']:.2f}]")
    
    print(f"\n⚠️  Top 5 Riskiest Counties:")
    for _, row in county_df.nlargest(5, 'lambda_mean').iterrows():
        print(f"  {row['state']:20s} (FIPS {row['fips']}): λ = {row['lambda_mean']:5.2f} "
              f"[{row['lambda_ci_lower']:.2f}, {row['lambda_ci_upper']:.2f}]")
    
    print(f"\n💡 Note:")
    print(f"  - ADVI provides approximate posterior (faster but less exact than MCMC)")
    print(f"  - For publication-quality results, validate with MCMC on subset")
    print(f"  - Credible intervals may be slightly narrower than MCMC")


def run_advi_model(data_path, output_dir='./advi_results', 
                   n_iterations=50000, filter_states=None, use_hdi=False):
    """
    Main pipeline with ADVI.
    
    Parameters:
    -----------
    use_hdi : bool
        If True, use HDI for credible intervals. Default False (percentile).
    """
    import time
    start_time = time.time()
    
    # Prepare data
    county_data, state_to_idx = prepare_county_data(data_path, filter_states)
    
    # Build model
    model = build_hierarchical_model(county_data)
    
    # Fit with ADVI (fast!)
    trace, approx = fit_with_advi(model, n_iterations)

    # Persist full approximate posterior for downstream use
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    posterior_file = output_path / "county_advi_posterior.nc"
    az.to_netcdf(trace, posterior_file)
    print(f"✓ Saved full posterior samples to: {posterior_file}")
    
    # Extract results
    county_df, state_df = extract_results(trace, county_data, use_hdi=use_hdi)
    
    # Calculate ICC
    icc = calculate_icc(trace)
    
    # Plots
    create_diagnostic_plots(county_df, state_df, approx, output_dir)
    
    # Save summaries
    save_results(county_df, state_df, output_dir)
    
    # Summary
    print_summary(county_df, state_df)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ COMPLETE! Total time: {elapsed/60:.1f} minutes")
    print(f"✓ Results saved to: {output_dir}/")
    print(f"{'='*70}\n")
    
    return county_df, state_df, trace, approx


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fast county-level model using ADVI'
    )
    parser.add_argument('data_path', type=str, 
                       help='Path to cleaned data CSV')
    parser.add_argument('--output-dir', type=str, default='./advi_results',
                       help='Output directory')
    parser.add_argument('--iterations', type=int, default=50000,
                       help='ADVI iterations (default: 50000)')
    parser.add_argument('--states', type=str, nargs='+', default=None,
                       help='Filter to specific states (e.g., --states Texas California)')
    parser.add_argument('--use-hdi', action='store_true',
                       help='Use HDI instead of percentile intervals (slightly slower)')
    
    args = parser.parse_args()
    
    # Run
    county_df, state_df, trace, approx = run_advi_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_iterations=args.iterations,
        filter_states=args.states,
        use_hdi=args.use_hdi
    )
