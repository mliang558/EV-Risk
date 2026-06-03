#!/usr/bin/env python3
"""
Fast State-Level Model with Per-County Normalization
====================================================
Compromise solution: State-level Bayesian + post-hoc per-county normalization
Time: ~5 minutes instead of hours
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def quick_state_model_with_normalization(data_path, output_dir='./fast_results'):
    """
    Fast state-level model with per-county normalization.
    
    Approach:
    1. Model total state-level lambda (fast)
    2. Normalize by county count (post-hoc)
    3. Get both total and per-county metrics
    """
    
    print("="*60)
    print("FAST STATE MODEL WITH PER-COUNTY NORMALIZATION")
    print("="*60)
    
    # ============================================================
    # Load and aggregate data
    # ============================================================
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['year'] = df['start_time'].dt.year
    
    # State-level aggregation
    state_agg = df.groupby('state').agg(
        n_events=('state', 'count'),
        n_counties=('fips', 'nunique'),  # Count unique counties
        min_year=('year', 'min'),
        max_year=('year', 'max')
    ).reset_index()
    
    state_agg['n_years'] = state_agg['max_year'] - state_agg['min_year'] + 1
    state_agg['lambda_raw_total'] = state_agg['n_events'] / state_agg['n_years']
    state_agg['lambda_raw_per_county'] = state_agg['lambda_raw_total'] / state_agg['n_counties']
    
    print(f"States: {len(state_agg)}")
    print(f"Total counties: {state_agg['n_counties'].sum()}")
    
    # ============================================================
    # Bayesian model for TOTAL lambda
    # ============================================================
    print("\nBuilding Bayesian model...")
    
    n_states = len(state_agg)
    Y_obs = state_agg['n_events'].values.astype(int)
    T_obs = state_agg['n_years'].values.astype(float)
    
    # Priors from data
    log_lambda_raw = np.log(state_agg['lambda_raw_total'] + 0.1)
    mu_data = log_lambda_raw.mean()
    sigma_data = log_lambda_raw.std()
    
    with pm.Model() as model:
        mu_national = pm.Normal("mu_national", mu=mu_data, sigma=sigma_data * 2)
        sigma_state = pm.HalfNormal("sigma_state", sigma=sigma_data)
        
        log_lambda_total = pm.Normal(
            "log_lambda_total",
            mu=mu_national,
            sigma=sigma_state,
            shape=n_states
        )
        
        lambda_total = pm.Deterministic("lambda_total", pm.math.exp(log_lambda_total))
        
        pm.Poisson("Y_obs", mu=lambda_total * T_obs, observed=Y_obs)
    
    # ============================================================
    # Sample
    # ============================================================
    print("Sampling (this is fast - ~2 minutes)...")
    
    with model:
        trace = pm.sample(
            draws=1000,
            tune=1000,
            cores=1,
            target_accept=0.95,
            return_inferencedata=True,
            random_seed=42
        )
    
    print("✓ Sampling complete")
    
    # ============================================================
    # Extract and normalize
    # ============================================================
    print("\nExtracting and normalizing results...")
    
    lambda_total_posterior = trace.posterior['lambda_total']
    lambda_total_samples = lambda_total_posterior.values.reshape(-1, n_states)
    
    results = []
    for i, row in state_agg.iterrows():
        # Total lambda samples
        total_samples = lambda_total_samples[:, i]
        
        # Per-county lambda samples (divide by county count)
        per_county_samples = total_samples / row['n_counties']
        
        results.append({
            'state': row['state'],
            'n_counties': row['n_counties'],
            'n_events': row['n_events'],
            'n_years': row['n_years'],
            
            # Total lambda (absolute risk)
            'lambda_total_mean': total_samples.mean(),
            'lambda_total_median': np.median(total_samples),
            'lambda_total_sd': total_samples.std(),
            'lambda_total_ci_lower': np.percentile(total_samples, 2.5),
            'lambda_total_ci_upper': np.percentile(total_samples, 97.5),
            
            # Per-county lambda (comparable risk)
            'lambda_per_county_mean': per_county_samples.mean(),
            'lambda_per_county_median': np.median(per_county_samples),
            'lambda_per_county_sd': per_county_samples.std(),
            'lambda_per_county_ci_lower': np.percentile(per_county_samples, 2.5),
            'lambda_per_county_ci_upper': np.percentile(per_county_samples, 97.5),
            
            # Raw estimates for comparison
            'lambda_raw_total': row['lambda_raw_total'],
            'lambda_raw_per_county': row['lambda_raw_per_county']
        })
    
    results_df = pd.DataFrame(results)
    
    # ============================================================
    # Save results
    # ============================================================
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    csv_file = output_path / 'state_lambda_normalized.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved: {csv_file}")
    
    # ============================================================
    # Create plots
    # ============================================================
    print("\nCreating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Total lambda
    ax = axes[0, 0]
    plot_df = results_df.sort_values('lambda_total_mean', ascending=True).tail(20)
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df['lambda_total_mean'], alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['state'])
    ax.set_xlabel('Total λ (events/year)', fontsize=11)
    ax.set_title('Top 20: TOTAL Frequency (Absolute Risk)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    # Plot 2: Per-county lambda
    ax = axes[0, 1]
    plot_df = results_df.sort_values('lambda_per_county_mean', ascending=True).tail(20)
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df['lambda_per_county_mean'], alpha=0.7, color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['state'])
    ax.set_xlabel('Per-County λ (events/county/year)', fontsize=11)
    ax.set_title('Top 20: PER-COUNTY Frequency (Comparable Risk)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    # Plot 3: Scatter
    ax = axes[1, 0]
    scatter = ax.scatter(
        results_df['n_counties'],
        results_df['lambda_total_mean'],
        s=100,
        alpha=0.6,
        c=results_df['lambda_per_county_mean'],
        cmap='RdYlGn_r'
    )
    ax.set_xlabel('Number of Counties', fontsize=11)
    ax.set_ylabel('Total λ', fontsize=11)
    ax.set_title('Total λ vs County Count\n(Color = Per-County λ)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Per-County λ')
    
    # Add labels for outliers
    for _, row in results_df.nlargest(5, 'lambda_total_mean').iterrows():
        ax.annotate(row['state'], 
                   (row['n_counties'], row['lambda_total_mean']),
                   fontsize=8, alpha=0.7)
    
    # Plot 4: Comparison table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Top 10 comparison
    top_total = results_df.nlargest(10, 'lambda_total_mean')[['state', 'lambda_total_mean']].reset_index(drop=True)
    top_per_county = results_df.nlargest(10, 'lambda_per_county_mean')[['state', 'lambda_per_county_mean']].reset_index(drop=True)
    
    table_text = "TOP 10 COMPARISON\n" + "="*40 + "\n\n"
    table_text += "BY TOTAL λ          BY PER-COUNTY λ\n"
    table_text += "-"*40 + "\n"
    
    for i in range(10):
        table_text += f"{top_total.iloc[i]['state']:12s} {top_total.iloc[i]['lambda_total_mean']:6.0f}    "
        table_text += f"{top_per_county.iloc[i]['state']:12s} {top_per_county.iloc[i]['lambda_per_county_mean']:5.1f}\n"
    
    ax.text(0.1, 0.5, table_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    plot_file = output_path / 'state_comparison_fast.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot_file}")
    plt.close()
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\nTop 5 by TOTAL lambda (absolute risk):")
    for _, row in results_df.nlargest(5, 'lambda_total_mean').iterrows():
        print(f"  {row['state']:15s}: {row['lambda_total_mean']:7.1f} events/year "
              f"({row['n_counties']:3d} counties)")
    
    print("\nTop 5 by PER-COUNTY lambda (comparable risk):")
    for _, row in results_df.nlargest(5, 'lambda_per_county_mean').iterrows():
        print(f"  {row['state']:15s}: {row['lambda_per_county_mean']:5.2f} events/county/year "
              f"({row['n_counties']:3d} counties)")
    
    print(f"\n✓ All results in: {output_dir}/")
    
    return results_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to cleaned data')
    parser.add_argument('--output-dir', type=str, default='./fast_results')
    
    args = parser.parse_args()
    
    results = quick_state_model_with_normalization(args.data_path, args.output_dir)
