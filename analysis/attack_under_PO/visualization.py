"""
Visualization Tools
可视化模拟结果

This module provides visualization functions for analyzing simulation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_loss_distribution(mc_results: Dict, state_name: str, 
                           save_path: str = None, show: bool = True):
    """
    Plot annual loss distribution (histogram and CDF)
    
    Parameters:
    -----------
    mc_results : dict
        Monte Carlo simulation results
    state_name : str
        State name for plot title
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    losses = mc_results['loss_distribution']
    
    # Histogram
    axes[0].hist(losses, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(mc_results['mean_annual_loss'], color='red', 
                    linestyle='--', linewidth=2, label=f"Mean: {mc_results['mean_annual_loss']:.2f}")
    axes[0].axvline(mc_results['VaR_95'], color='orange', 
                    linestyle='--', linewidth=2, label=f"95% VaR: {mc_results['VaR_95']:.2f}")
    axes[0].set_xlabel('Annual Efficiency Loss (hour-weighted)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{state_name} - Annual Loss Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # CDF
    sorted_losses = np.sort(losses)
    cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    axes[1].plot(sorted_losses, cdf, linewidth=2, color='steelblue')
    axes[1].axhline(0.95, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1].axvline(mc_results['VaR_95'], color='orange', linestyle='--', linewidth=2,
                    label=f"95% VaR: {mc_results['VaR_95']:.2f}")
    axes[1].set_xlabel('Annual Efficiency Loss', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].set_title(f'{state_name} - Cumulative Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_state_comparison(comparison_df: pd.DataFrame, 
                         save_path: str = None, show: bool = True):
    """
    Plot multi-state comparison
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison results from run_multi_state_comparison()
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sort by normalized loss
    df_sorted = comparison_df.sort_values('Composite_Normalized_Loss', ascending=False)
    
    colors = sns.color_palette("viridis", len(df_sorted))
    
    # 1. Raw annual loss
    axes[0, 0].barh(df_sorted['State'], df_sorted['Mean_Annual_Loss'], color=colors)
    axes[0, 0].set_xlabel('Mean Annual Loss', fontsize=11)
    axes[0, 0].set_title('Raw Annual Loss by State', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. Normalized loss (composite)
    axes[0, 1].barh(df_sorted['State'], df_sorted['Composite_Normalized_Loss'], color=colors)
    axes[0, 1].set_xlabel('Composite Normalized Loss', fontsize=11)
    axes[0, 1].set_title('Normalized Loss by State (Comparable)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 3. Loss per station
    axes[1, 0].barh(df_sorted['State'], df_sorted['Loss_Per_Station'], color=colors)
    axes[1, 0].set_xlabel('Loss per Charging Station', fontsize=11)
    axes[1, 0].set_title('Loss per Charging Station', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 4. VaR comparison
    axes[1, 1].barh(df_sorted['State'], df_sorted['VaR_95'], color=colors)
    axes[1, 1].set_xlabel('95% Value at Risk', fontsize=11)
    axes[1, 1].set_title('Extreme Loss Scenarios (95% VaR)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_vulnerability_ranking(comparison_df: pd.DataFrame,
                               save_path: str = None, show: bool = True):
    """
    Plot vulnerability ranking across states
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison results
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort by vulnerability rank
    df_sorted = comparison_df.sort_values('Vulnerability_Rank')
    
    # Color code by rank
    colors = sns.color_palette("RdYlGn_r", len(df_sorted))
    
    bars = ax.barh(df_sorted['State'], df_sorted['Composite_Normalized_Loss'], color=colors)
    
    # Add rank labels
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['Composite_Normalized_Loss'], i, 
               f"  Rank {int(row['Vulnerability_Rank'])}", 
               va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Composite Normalized Loss', fontsize=12)
    ax.set_title('State Vulnerability Ranking\n(Higher = More Vulnerable)', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_risk_metrics(comparison_df: pd.DataFrame,
                     save_path: str = None, show: bool = True):
    """
    Plot risk metrics comparison (Mean vs VaR vs CVaR)
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison results
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    states = comparison_df['State']
    x = np.arange(len(states))
    width = 0.25
    
    # Plot bars
    ax.bar(x - width, comparison_df['Mean_Annual_Loss'], width, 
          label='Mean Loss', alpha=0.8, color='steelblue')
    ax.bar(x, comparison_df['VaR_95'], width, 
          label='95% VaR', alpha=0.8, color='orange')
    ax.bar(x + width, comparison_df['CVaR_95'], width, 
          label='95% CVaR', alpha=0.8, color='red')
    
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Loss Metric', fontsize=12)
    ax.set_title('Risk Metrics Comparison Across States', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_county_vulnerability(simulator, county_loss_dict: Dict[str, float],
                             top_n: int = 15,
                             save_path: str = None, show: bool = True):
    """
    Plot county-level vulnerability
    
    Parameters:
    -----------
    simulator : RealisticOutageSimulator
        Simulator instance
    county_loss_dict : dict
        Dictionary mapping county names to expected loss contributions
    top_n : int
        Number of top counties to display
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort counties by loss
    sorted_counties = sorted(county_loss_dict.items(), key=lambda x: x[1], reverse=True)
    top_counties = sorted_counties[:top_n]
    
    counties, losses = zip(*top_counties)
    
    # Create color gradient
    colors = sns.color_palette("Reds_r", len(counties))
    
    bars = ax.barh(counties, losses, color=colors)
    
    # Add value labels
    for i, (county, loss) in enumerate(top_counties):
        ax.text(loss, i, f'  {loss:.2f}', va='center', fontsize=9)
    
    ax.set_xlabel('Expected Annual Loss Contribution', fontsize=12)
    ax.set_ylabel('County', fontsize=12)
    ax.set_title(f'{simulator.state} - Top {top_n} Most Vulnerable Counties', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_sensitivity_analysis(sensitivity_results: Dict[str, Dict],
                              save_path: str = None, show: bool = True):
    """
    Plot sensitivity analysis results
    
    Parameters:
    -----------
    sensitivity_results : dict
        Dictionary with different scenario results
        Format: {'scenario_name': mc_results_dict, ...}
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    scenarios = list(sensitivity_results.keys())
    means = [sensitivity_results[s]['mean_annual_loss'] for s in scenarios]
    vars_95 = [sensitivity_results[s]['VaR_95'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    
    # Mean comparison
    axes[0].bar(x, means, color='steelblue', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Annual Loss', fontsize=11)
    axes[0].set_title('Sensitivity Analysis - Mean Loss', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(means):
        axes[0].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    # VaR comparison
    axes[1].bar(x, vars_95, color='orange', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[1].set_ylabel('95% VaR', fontsize=11)
    axes[1].set_title('Sensitivity Analysis - 95% VaR', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(vars_95):
        axes[1].text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_summary_dashboard(mc_results: Dict, state_name: str,
                            save_path: str = None, show: bool = True):
    """
    Create comprehensive summary dashboard
    
    Parameters:
    -----------
    mc_results : dict
        Monte Carlo simulation results
    state_name : str
        State name
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    losses = mc_results['loss_distribution']
    
    # Main histogram (top left, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(losses, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(mc_results['mean_annual_loss'], color='red', linestyle='--', linewidth=2)
    ax1.axvline(mc_results['VaR_95'], color='orange', linestyle='--', linewidth=2)
    ax1.set_xlabel('Annual Loss')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{state_name} - Loss Distribution', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Statistics box (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = f"""
    Key Statistics:
    
    Mean Loss: {mc_results['mean_annual_loss']:.2f}
    Median Loss: {mc_results['median_annual_loss']:.2f}
    Std Dev: {mc_results['std_annual_loss']:.2f}
    
    95% VaR: {mc_results['VaR_95']:.2f}
    95% CVaR: {mc_results['CVaR_95']:.2f}
    
    Avg Events/Year: {mc_results['mean_n_events']:.1f}
    """
    ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    # CDF (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    sorted_losses = np.sort(losses)
    cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    ax3.plot(sorted_losses, cdf, linewidth=2, color='steelblue')
    ax3.axhline(0.95, color='orange', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Loss')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('CDF', fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Box plot (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.boxplot(losses, vert=True)
    ax4.set_ylabel('Loss')
    ax4.set_title('Box Plot', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Q-Q plot (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    from scipy import stats
    stats.probplot(losses, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot (Normal)', fontweight='bold')
    ax5.grid(alpha=0.3)
    
    # Kernel density (bottom, spanning all columns)
    ax6 = fig.add_subplot(gs[2, :])
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(losses)
    x_range = np.linspace(losses.min(), losses.max(), 200)
    ax6.plot(x_range, kde(x_range), linewidth=2, color='steelblue')
    ax6.fill_between(x_range, kde(x_range), alpha=0.3, color='steelblue')
    ax6.axvline(mc_results['mean_annual_loss'], color='red', linestyle='--', 
               linewidth=2, label='Mean')
    ax6.axvline(mc_results['VaR_95'], color='orange', linestyle='--', 
               linewidth=2, label='95% VaR')
    ax6.set_xlabel('Annual Loss')
    ax6.set_ylabel('Density')
    ax6.set_title('Kernel Density Estimation', fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    fig.suptitle(f'{state_name} - Comprehensive Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    print("Visualization Tools Module")
    print("="*60)
    print("Available functions:")
    print("  - plot_loss_distribution()")
    print("  - plot_state_comparison()")
    print("  - plot_vulnerability_ranking()")
    print("  - plot_risk_metrics()")
    print("  - plot_county_vulnerability()")
    print("  - plot_sensitivity_analysis()")
    print("  - create_summary_dashboard()")
    print("="*60)
