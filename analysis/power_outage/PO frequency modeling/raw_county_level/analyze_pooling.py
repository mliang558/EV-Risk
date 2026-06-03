#!/usr/bin/env python3
"""
Improved Pooling Effect Visualization
Shows where pooling actually matters
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
county_df = pd.read_csv('all_states_final/county_estimates_advi.csv')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ============================================================
# Plot 1: Pooling effect colored by sample size
# ============================================================
ax = axes[0, 0]
scatter = ax.scatter(
    county_df['lambda_raw'], 
    county_df['lambda_mean'],
    c=np.log10(county_df['n_events'] + 1),  # Log scale for color
    s=20,
    alpha=0.6,
    cmap='viridis'
)
max_val = max(county_df['lambda_raw'].max(), county_df['lambda_mean'].max())
ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='No pooling')
ax.set_xlabel('Raw estimate (MLE)', fontsize=12)
ax.set_ylabel('Bayesian estimate (pooled)', fontsize=12)
ax.set_title('Pooling Effect (color = log10(n events))', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('log10(n events)', fontsize=10)

# ============================================================
# Plot 2: Zoom in on sparse counties
# ============================================================
ax = axes[0, 1]
sparse = county_df[county_df['n_events'] < 50]
if len(sparse) > 0:
    ax.scatter(sparse['lambda_raw'], sparse['lambda_mean'], 
              c=sparse['n_events'], s=50, alpha=0.6, cmap='Reds')
    max_val = max(sparse['lambda_raw'].max(), sparse['lambda_mean'].max())
    ax.plot([0, max_val], [0, max_val], 'b--', alpha=0.5, linewidth=2, label='No pooling')
    ax.set_xlabel('Raw estimate', fontsize=12)
    ax.set_ylabel('Bayesian estimate', fontsize=12)
    ax.set_title(f'ZOOM: Sparse Counties Only (n < 50)\nN = {len(sparse)} counties', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

# ============================================================
# Plot 3: Shrinkage by data density (bins)
# ============================================================
ax = axes[1, 0]

bins = [0, 5, 10, 20, 50, 100, 1000, 10000]
labels = ['<5', '5-9', '10-19', '20-49', '50-99', '100-999', '1000+']
county_df['n_bin'] = pd.cut(county_df['n_events'], bins=bins, labels=labels)

shrinkage_by_bin = county_df.groupby('n_bin', observed=True)['shrinkage'].agg(['mean', 'count'])
shrinkage_by_bin = shrinkage_by_bin[shrinkage_by_bin['count'] > 0]

y_pos = np.arange(len(shrinkage_by_bin))
bars = ax.barh(y_pos, shrinkage_by_bin['mean'] * 100, alpha=0.7)

# Color bars by shrinkage amount
colors = plt.cm.RdYlGn_r(shrinkage_by_bin['mean'] / shrinkage_by_bin['mean'].max())
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{idx}\n(n={row['count']})" for idx, row in shrinkage_by_bin.iterrows()])
ax.set_xlabel('Average Shrinkage (%)', fontsize=12)
ax.set_ylabel('Events per County', fontsize=12)
ax.set_title('Pooling Strength by Data Density', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# ============================================================
# Plot 4: Before/After uncertainty
# ============================================================
ax = axes[1, 1]

# Calculate CV (coefficient of variation)
county_df['cv_raw'] = np.sqrt(county_df['lambda_raw']) / (county_df['lambda_raw'] + 0.1)
county_df['cv_bayes'] = county_df['lambda_sd'] / county_df['lambda_mean']

# Remove extreme outliers
valid = (county_df['cv_raw'] < 2) & (county_df['cv_bayes'] < 2)
plot_df = county_df[valid]

scatter = ax.scatter(
    plot_df['cv_raw'], 
    plot_df['cv_bayes'],
    c=np.log10(plot_df['n_events'] + 1),
    s=20,
    alpha=0.5,
    cmap='viridis'
)
max_cv = min(plot_df['cv_raw'].max(), plot_df['cv_bayes'].max())
ax.plot([0, max_cv], [0, max_cv], 'r--', alpha=0.5, linewidth=2, label='No change')
ax.set_xlabel('Raw CV (uncertainty)', fontsize=12)
ax.set_ylabel('Bayesian CV (uncertainty)', fontsize=12)
ax.set_title('Uncertainty Reduction\n(points below line = improvement)', 
            fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pooling_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: pooling_detailed_analysis.png")

# ============================================================
# Print statistics
# ============================================================
print("\n" + "="*70)
print("POOLING EFFECT STATISTICS")
print("="*70)

print(f"\n📊 Overall:")
print(f"  Total counties: {len(county_df)}")
print(f"  Average shrinkage: {county_df['shrinkage'].mean():.1%}")

print(f"\n🔍 By data density:")
for idx, row in shrinkage_by_bin.iterrows():
    print(f"  {idx:10s}: {row['mean']*100:5.1f}% shrinkage ({int(row['count'])} counties)")

sparse = county_df[county_df['n_events'] < 10]
if len(sparse) > 0:
    print(f"\n⚠️  Sparse counties (n < 10):")
    print(f"  Count: {len(sparse)}")
    print(f"  Average shrinkage: {sparse['shrinkage'].mean():.1%}")
    print(f"  Max shrinkage: {sparse['shrinkage'].max():.1%}")

print(f"\n✓ Dense counties (n > 100):")
dense = county_df[county_df['n_events'] > 100]
print(f"  Count: {len(dense)}")
print(f"  Average shrinkage: {dense['shrinkage'].mean():.1%}")
