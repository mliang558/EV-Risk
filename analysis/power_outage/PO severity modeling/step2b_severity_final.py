#!/usr/bin/env python3
"""
Step 2B: State-Level Severity Modeling with Interaction Effects
================================================================

Model P(severe event | state) considering:
1. Duration (continuous, log-transformed)
2. Customer impact percentage (continuous, 0-1)
3. Their interaction (longer events may affect more people)

Output: State-level severity probabilities for risk assessment
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.special import expit, logit
import warnings
warnings.filterwarnings('ignore')


class StateSeverityModeler:
    """
    Model severity at state level with duration-customer interaction.
    """
    
    def __init__(self, data_path, population_path):
        """
        Initialize modeler.
        
        Parameters:
        -----------
        data_path : str
            Path to cleaned outage data CSV
            Required columns: fips, state, county, start_time, duration_min, mean_customers
        population_path : str
            Path to county population data CSV
            Required columns: STATE, COUNTY, POPESTIMATE2022
        """
        self.data_path = data_path
        self.population_path = population_path
        self.df = None
        self.state_results = None
        self.trace = None
        self.model = None
        
    def load_and_prepare_data(self):
        """Load data and merge with population."""
        print("="*60)
        print("STEP 2B: STATE-LEVEL SEVERITY MODELING")
        print("WITH DURATION-CUSTOMER INTERACTION")
        print("="*60)
        
        # Load outage data
        print("\n[1/4] Loading outage data...")
        self.df = pd.read_csv(self.data_path)
        print(f"  Loaded {len(self.df)} outage events")
        
        # Convert timestamps
        self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        self.df['duration_hr'] = self.df['duration_min'] / 60
        
        # Ensure fips is 5-digit string
        self.df['fips'] = self.df['fips'].astype(str).str.zfill(5)
        
        # Load population data
        print("\n[2/4] Loading population data...")
        pop = pd.read_csv(self.population_path, encoding='latin-1')
        
        # Create fips in population data
        pop['STATE'] = pop['STATE'].astype(str).str.zfill(2)
        pop['COUNTY'] = pop['COUNTY'].astype(str).str.zfill(3)
        pop['fips'] = pop['STATE'] + pop['COUNTY']
        
        print(f"  Loaded population for {pop['fips'].nunique()} counties")
        
        # Merge
        print("\n[3/4] Merging data...")
        self.df = self.df.merge(
            pop[['fips', 'POPESTIMATE2022']],
            on='fips',
            how='left'
        )
        
        # Check merge success
        missing_pop = self.df['POPESTIMATE2022'].isna().sum()
        if missing_pop > 0:
            print(f"  ⚠️  Warning: {missing_pop} events missing population data")
            print(f"     Will drop these events")
            self.df = self.df.dropna(subset=['POPESTIMATE2022'])
        
        # Calculate customer impact percentage
        self.df['customers_pct'] = self.df['mean_customers'] / self.df['POPESTIMATE2022']
        self.df['customers_pct'] = self.df['customers_pct'].clip(0, 1)  # Cap at 100%
        
        print(f"\n[4/4] Data prepared:")
        print(f"  Final events: {len(self.df)}")
        print(f"  States: {self.df['state'].nunique()}")
        print(f"  Date range: {self.df['start_time'].min().date()} to {self.df['start_time'].max().date()}")
        print(f"\n  Duration (hours):")
        print(f"    Min: {self.df['duration_hr'].min():.2f}")
        print(f"    Median: {self.df['duration_hr'].median():.2f}")
        print(f"    Mean: {self.df['duration_hr'].mean():.2f}")
        print(f"    Max: {self.df['duration_hr'].max():.2f}")
        print(f"\n  Customer impact (% of county):")
        print(f"    Min: {self.df['customers_pct'].min():.4f}")
        print(f"    Median: {self.df['customers_pct'].median():.4f}")
        print(f"    Mean: {self.df['customers_pct'].mean():.4f}")
        print(f"    Max: {self.df['customers_pct'].max():.4f}")
        
        return self
    
    def define_severity_multiple_thresholds(self, 
                                           duration_thresholds=[2, 4, 8],
                                           customer_pct_thresholds=[0.005, 0.01, 0.05]):
        """
        Define severity for multiple threshold combinations.
        
        Creates severity labels for different definitions:
        - Duration only
        - Customer impact only  
        - Combined (duration AND impact)
        
        Parameters:
        -----------
        duration_thresholds : list
            Hours thresholds (e.g., [2, 4, 8] for 2hr, 4hr, 8hr)
        customer_pct_thresholds : list
            Percentage thresholds (e.g., [0.005, 0.01, 0.05] for 0.5%, 1%, 5%)
        """
        print("\n" + "="*60)
        print("DEFINING SEVERITY THRESHOLDS")
        print("="*60)
        
        self.duration_thresholds = duration_thresholds
        self.customer_pct_thresholds = customer_pct_thresholds
        
        # Duration-based severity
        for dur in duration_thresholds:
            col_name = f'severe_dur_{int(dur)}h'
            self.df[col_name] = (self.df['duration_hr'] > dur).astype(int)
            n_severe = self.df[col_name].sum()
            pct = self.df[col_name].mean() * 100
            print(f"  Duration > {dur}h: {n_severe:6d} events ({pct:5.2f}%)")
        
        # Customer impact severity
        for cust_pct in customer_pct_thresholds:
            col_name = f'severe_cust_{int(cust_pct*1000)}bps'  # basis points
            self.df[col_name] = (self.df['customers_pct'] > cust_pct).astype(int)
            n_severe = self.df[col_name].sum()
            pct = self.df[col_name].mean() * 100
            print(f"  Impact > {cust_pct*100:.2f}%: {n_severe:6d} events ({pct:5.2f}%)")
        
        # Combined severity (example: 2h AND 1%)
        self.df['severe_combined'] = (
            (self.df['duration_hr'] > 2) & 
            (self.df['customers_pct'] > 0.01)
        ).astype(int)
        n_severe = self.df['severe_combined'].sum()
        pct = self.df['severe_combined'].mean() * 100
        print(f"\n  Combined (>2h AND >1%): {n_severe:6d} events ({pct:5.2f}%)")
        
        return self
    
    def aggregate_to_state_level(self):
        """
        Aggregate to state level.
        
        For each state:
        - Total events
        - Number of severe events (by each definition)
        - Mean duration and customer impact
        - Empirical P(severe)
        """
        print("\n" + "="*60)
        print("AGGREGATING TO STATE LEVEL")
        print("="*60)
        
        # Find severity columns
        severity_cols = [col for col in self.df.columns if col.startswith('severe_')]
        
        # Aggregation
        agg_dict = {
            'state': 'count',  # n_events
            'duration_hr': ['mean', 'median', 'std'],
            'customers_pct': ['mean', 'median', 'std']
        }
        
        # Add severity columns
        for col in severity_cols:
            agg_dict[col] = 'sum'
        
        state_agg = self.df.groupby('state').agg(agg_dict).reset_index()
        
        # Flatten column names
        state_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                            for col in state_agg.columns]
        state_agg = state_agg.rename(columns={'state_count': 'n_events'})
        
        # Calculate empirical probabilities
        for col in severity_cols:
            col_sum = f'{col}_sum'
            col_prob = f'p_{col}_raw'
            if col_sum in state_agg.columns:
                state_agg[col_prob] = state_agg[col_sum] / state_agg['n_events']
        
        self.state_agg = state_agg
        
        print(f"  Aggregated to {len(state_agg)} states")
        print(f"\n  State-level statistics:")
        print(f"    Events per state: {state_agg['n_events'].mean():.0f} ± {state_agg['n_events'].std():.0f}")
        print(f"    Mean duration: {state_agg['duration_hr_mean'].mean():.2f}h ± {state_agg['duration_hr_mean'].std():.2f}h")
        print(f"    Mean impact: {state_agg['customers_pct_mean'].mean():.4f} ± {state_agg['customers_pct_mean'].std():.4f}")
        
        if 'p_severe_combined_raw' in state_agg.columns:
            print(f"\n  P(severe combined) across states:")
            print(f"    Mean: {state_agg['p_severe_combined_raw'].mean():.3f}")
            print(f"    Min: {state_agg['p_severe_combined_raw'].min():.3f}")
            print(f"    Max: {state_agg['p_severe_combined_raw'].max():.3f}")
        
        return self
    
    def build_interaction_model(self, severity_type='combined'):
        """
        Build Bayesian model with duration-customer interaction.
        
        Model structure:
        At state level:
          National hyperprior: μ_national, σ_state
          State: logit(p_severe[s]) ~ Normal(μ_national, σ_state)
          Observation: n_severe[s] ~ Binomial(n_events[s], p_severe[s])
        
        Note: Interaction is implicit - we model P(severe) where severe is 
        defined as (duration > T) AND (customers_pct > C), which captures 
        the joint probability.
        
        Parameters:
        -----------
        severity_type : str
            Which severity definition to use: 'combined', 'severe_dur_2h', etc.
        """
        print("\n" + "="*60)
        print(f"BUILDING BAYESIAN MODEL: {severity_type}")
        print("="*60)
        
        # Get data
        n_events = self.state_agg['n_events'].values.astype(int)
        n_states = len(self.state_agg)
        
        # Get severe counts
        if severity_type == 'combined':
            n_severe = self.state_agg['severe_combined_sum'].values.astype(int)
            p_raw_col = 'p_severe_combined_raw'
        else:
            n_severe = self.state_agg[f'{severity_type}_sum'].values.astype(int)
            p_raw_col = f'p_{severity_type}_raw'
        
        # Empirical logit for prior
        p_empirical = self.state_agg[p_raw_col].values
        p_empirical_clipped = np.clip(p_empirical, 0.01, 0.99)
        logit_empirical = np.log(p_empirical_clipped / (1 - p_empirical_clipped))
        mu_logit_data = logit_empirical.mean()
        sigma_logit_data = logit_empirical.std()
        
        print(f"\n  Data summary:")
        print(f"    States: {n_states}")
        print(f"    Total events: {n_events.sum()}")
        print(f"    Total severe: {n_severe.sum()} ({n_severe.sum()/n_events.sum()*100:.2f}%)")
        print(f"\n  Data-driven prior:")
        print(f"    logit(p) ~ Normal({mu_logit_data:.3f}, {sigma_logit_data:.3f})")
        print(f"    Implies p ~ {expit(mu_logit_data):.3f} ± {sigma_logit_data:.3f} (on logit scale)")
        
        # Build model
        with pm.Model() as self.model:
            # National-level hyperprior
            mu_logit = pm.Normal(
                "mu_logit", 
                mu=mu_logit_data, 
                sigma=sigma_logit_data * 2  # Inflate uncertainty
            )
            sigma_state = pm.HalfNormal("sigma_state", sigma=1.0)
            
            # State-level logit(p_severe)
            logit_p = pm.Normal(
                "logit_p",
                mu=mu_logit,
                sigma=sigma_state,
                shape=n_states
            )
            
            # Transform to probability
            p_severe = pm.Deterministic("p_severe", pm.math.sigmoid(logit_p))
            
            # Likelihood: n_severe ~ Binomial(n_events, p_severe)
            pm.Binomial("n_severe_obs", n=n_events, p=p_severe, observed=n_severe)
        
        print(f"\n  Model built successfully")
        print(f"  Parameters: {n_states + 2} (mu_logit, sigma_state, {n_states} state logit_p)")
        
        self.severity_type = severity_type
        
        return self
    
    def sample_posterior(self, n_samples=1000, n_tune=1000):
        """Sample from posterior distribution."""
        print("\n" + "="*60)
        print("SAMPLING POSTERIOR")
        print("="*60)
        print(f"  Draws: {n_samples}")
        print(f"  Tune: {n_tune}")
        print(f"  Cores: 1 (Windows compatible)")
        print(f"\n  This will take ~2-3 minutes...")
        
        with self.model:
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                cores=1,
                target_accept=0.95,
                return_inferencedata=True,
                random_seed=42
            )
        
        print("\n✓ Sampling complete!")
        
        # Convergence diagnostics
        print("\n  Convergence diagnostics:")
        rhat = az.rhat(self.trace, var_names=['p_severe'])
        ess = az.ess(self.trace, var_names=['p_severe'])
        
        print(f"    R-hat: max={rhat.p_severe.max().values:.4f}, mean={rhat.p_severe.mean().values:.4f}")
        print(f"    ESS: min={ess.p_severe.min().values:.0f}, mean={ess.p_severe.mean().values:.0f}")
        
        if rhat.p_severe.max().values > 1.01:
            print("    ⚠️  Warning: Some R-hat > 1.01, consider more samples")
        else:
            print("    ✓ All parameters converged well")
        
        return self
    
    def extract_state_severity(self):
        """Extract state-level severity probabilities with credible intervals."""
        print("\n" + "="*60)
        print("EXTRACTING STATE-LEVEL RESULTS")
        print("="*60)
        
        # Get posterior samples
        p_severe_posterior = self.trace.posterior['p_severe']
        p_severe_samples = p_severe_posterior.values.reshape(-1, len(self.state_agg))
        
        # Extract summary statistics
        results = []
        for i, row in self.state_agg.iterrows():
            samples = p_severe_samples[:, i]
            
            # Get raw probability
            if self.severity_type == 'combined':
                p_raw = row['p_severe_combined_raw']
            else:
                p_raw = row[f'p_{self.severity_type}_raw']
            
            results.append({
                'state': row['state'],
                'n_events': row['n_events'],
                'mean_duration_hr': row['duration_hr_mean'],
                'mean_customers_pct': row['customers_pct_mean'],
                
                # Posterior estimates
                'p_severe_mean': samples.mean(),
                'p_severe_median': np.median(samples),
                'p_severe_sd': samples.std(),
                'p_severe_ci_lower': np.percentile(samples, 2.5),
                'p_severe_ci_upper': np.percentile(samples, 97.5),
                
                # Raw empirical for comparison
                'p_severe_raw': p_raw,
                
                # Shrinkage
                'shrinkage': samples.mean() - p_raw
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('p_severe_mean', ascending=False)
        
        print(f"\n  Top 10 states by P(severe):")
        print(results_df[['state', 'p_severe_mean', 'p_severe_ci_lower', 'p_severe_ci_upper']]
              .head(10).to_string(index=False))
        
        print(f"\n  Summary statistics:")
        print(f"    Mean P(severe): {results_df['p_severe_mean'].mean():.3f}")
        print(f"    Std P(severe): {results_df['p_severe_mean'].std():.3f}")
        print(f"    Range: [{results_df['p_severe_mean'].min():.3f}, {results_df['p_severe_mean'].max():.3f}]")
        
        self.state_results = results_df
        
        return results_df
    
    def analyze_duration_customer_relationship(self, output_dir='./severity_results'):
        """
        Analyze the relationship between duration and customer impact.
        This helps understand the implicit interaction.
        """
        print("\n" + "="*60)
        print("ANALYZING DURATION-CUSTOMER RELATIONSHIP")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Calculate correlation at event level
        corr = self.df[['duration_hr', 'customers_pct']].corr().iloc[0, 1]
        print(f"\n  Event-level correlation: {corr:.3f}")
        
        # Correlation by state
        state_corrs = []
        for state in self.df['state'].unique():
            state_df = self.df[self.df['state'] == state]
            if len(state_df) > 10:  # At least 10 events
                corr_state = state_df[['duration_hr', 'customers_pct']].corr().iloc[0, 1]
                state_corrs.append({'state': state, 'correlation': corr_state})
        
        corr_df = pd.DataFrame(state_corrs).sort_values('correlation', ascending=False)
        print(f"\n  State-level correlations:")
        print(f"    Mean: {corr_df['correlation'].mean():.3f}")
        print(f"    Std: {corr_df['correlation'].std():.3f}")
        print(f"    Range: [{corr_df['correlation'].min():.3f}, {corr_df['correlation'].max():.3f}]")
        
        print(f"\n  Top 5 states with strongest positive correlation:")
        print(corr_df.head(5).to_string(index=False))
        
        # Create scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Overall scatter (log scale for duration)
        ax = axes[0, 0]
        sample = self.df.sample(min(10000, len(self.df)))  # Sample for speed
        ax.scatter(sample['duration_hr'], sample['customers_pct'], 
                  alpha=0.3, s=10)
        ax.set_xlabel('Duration (hours)', fontsize=11)
        ax.set_ylabel('Customer Impact (% of county)', fontsize=11)
        ax.set_title(f'Duration vs Customer Impact\nCorrelation: {corr:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # Plot 2: Hexbin density
        ax = axes[0, 1]
        ax.hexbin(np.log10(self.df['duration_hr'] + 0.01), 
                 np.log10(self.df['customers_pct'] + 0.0001),
                 gridsize=30, cmap='YlOrRd')
        ax.set_xlabel('log10(Duration hours)', fontsize=11)
        ax.set_ylabel('log10(Customer %)', fontsize=11)
        ax.set_title('Density: Duration vs Impact', fontsize=12, fontweight='bold')
        
        # Plot 3: By state comparison
        ax = axes[1, 0]
        state_medians = self.df.groupby('state').agg({
            'duration_hr': 'median',
            'customers_pct': 'median'
        }).reset_index()
        ax.scatter(state_medians['duration_hr'], state_medians['customers_pct'],
                  alpha=0.6, s=100)
        # Label top 5 states by correlation
        top_states = corr_df.head(5)['state'].values
        for state in top_states:
            state_data = state_medians[state_medians['state'] == state]
            if len(state_data) > 0:
                ax.annotate(state,
                          (state_data['duration_hr'].values[0], 
                           state_data['customers_pct'].values[0]),
                          fontsize=8, alpha=0.7)
        ax.set_xlabel('Median Duration (hours)', fontsize=11)
        ax.set_ylabel('Median Customer %', fontsize=11)
        ax.set_title('State-Level Medians', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Plot 4: Correlation by state
        ax = axes[1, 1]
        ax.bar(range(len(corr_df)), corr_df['correlation'].values, alpha=0.7)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('States (sorted by correlation)', fontsize=11)
        ax.set_ylabel('Correlation', fontsize=11)
        ax.set_title('Duration-Customer Correlation by State', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = output_path / 'duration_customer_interaction.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ Saved interaction analysis: {plot_file}")
        plt.close()
        
        # Save correlation data
        corr_file = output_path / 'state_correlations.csv'
        corr_df.to_csv(corr_file, index=False)
        print(f"  ✓ Saved correlations: {corr_file}")
        
        return corr_df
    
    def plot_results(self, output_dir='./severity_results'):
        """Create comprehensive visualization plots."""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: State ranking with CI
        ax = axes[0, 0]
        plot_df = self.state_results.sort_values('p_severe_mean', ascending=True).tail(20)
        y_pos = np.arange(len(plot_df))
        ax.errorbar(
            plot_df['p_severe_mean'],
            y_pos,
            xerr=[
                plot_df['p_severe_mean'] - plot_df['p_severe_ci_lower'],
                plot_df['p_severe_ci_upper'] - plot_df['p_severe_mean']
            ],
            fmt='o', markersize=6, capsize=3, alpha=0.7, color='steelblue'
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['state'])
        ax.set_xlabel('P(severe event)', fontsize=11)
        ax.set_title('Top 20 States by Severity Probability\n(with 95% CI)', 
                    fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        
        # Plot 2: Raw vs Bayesian (shrinkage)
        ax = axes[0, 1]
        scatter = ax.scatter(
            self.state_results['p_severe_raw'],
            self.state_results['p_severe_mean'],
            s=self.state_results['n_events']/10,
            alpha=0.6,
            c=self.state_results['n_events'],
            cmap='viridis'
        )
        max_val = max(self.state_results['p_severe_raw'].max(),
                     self.state_results['p_severe_mean'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel('Raw P(severe)', fontsize=11)
        ax.set_ylabel('Bayesian P(severe)', fontsize=11)
        ax.set_title('Shrinkage Effect\n(Size = # events)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='# Events')
        
        # Plot 3: Distribution of P(severe)
        ax = axes[0, 2]
        ax.hist(self.state_results['p_severe_mean'], bins=20, alpha=0.7, 
               edgecolor='black', color='steelblue')
        ax.axvline(self.state_results['p_severe_mean'].mean(),
                  color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {self.state_results["p_severe_mean"].mean():.3f}')
        ax.set_xlabel('P(severe)', fontsize=11)
        ax.set_ylabel('Number of States', fontsize=11)
        ax.set_title('Distribution of Severity Probabilities', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # Plot 4: P(severe) vs Mean Duration
        ax = axes[1, 0]
        ax.scatter(self.state_results['mean_duration_hr'],
                  self.state_results['p_severe_mean'],
                  s=100, alpha=0.6)
        # Add labels for top 5
        for _, row in self.state_results.head(5).iterrows():
            ax.annotate(row['state'],
                       (row['mean_duration_hr'], row['p_severe_mean']),
                       fontsize=8, alpha=0.7)
        ax.set_xlabel('Mean Duration (hours)', fontsize=11)
        ax.set_ylabel('P(severe)', fontsize=11)
        ax.set_title('Severity vs Duration', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Plot 5: P(severe) vs Mean Customer Impact
        ax = axes[1, 1]
        ax.scatter(self.state_results['mean_customers_pct']*100,
                  self.state_results['p_severe_mean'],
                  s=100, alpha=0.6, color='green')
        for _, row in self.state_results.head(5).iterrows():
            ax.annotate(row['state'],
                       (row['mean_customers_pct']*100, row['p_severe_mean']),
                       fontsize=8, alpha=0.7)
        ax.set_xlabel('Mean Customer Impact (% of county)', fontsize=11)
        ax.set_ylabel('P(severe)', fontsize=11)
        ax.set_title('Severity vs Customer Impact', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Plot 6: Uncertainty vs Sample Size
        ax = axes[1, 2]
        ci_width = (self.state_results['p_severe_ci_upper'] - 
                   self.state_results['p_severe_ci_lower'])
        ax.scatter(self.state_results['n_events'], ci_width, alpha=0.6)
        ax.set_xlabel('Number of Events', fontsize=11)
        ax.set_ylabel('95% CI Width', fontsize=11)
        ax.set_title('Uncertainty vs Sample Size', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_path / f'state_severity_{self.severity_type}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_file}")
        plt.close()
        
        return self
    
    def save_results(self, output_dir='./severity_results'):
        """Save all results to CSV files."""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Main results
        csv_file = output_path / f'state_severity_{self.severity_type}.csv'
        self.state_results.to_csv(csv_file, index=False)
        print(f"  ✓ Main results: {csv_file}")
        
        # Summary report
        summary_file = output_path / 'severity_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("STATE-LEVEL SEVERITY MODELING RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Severity definition: {self.severity_type}\n")
            f.write(f"Total states: {len(self.state_results)}\n")
            f.write(f"Total events analyzed: {self.state_results['n_events'].sum()}\n\n")
            
            f.write("Summary statistics:\n")
            f.write(f"  Mean P(severe): {self.state_results['p_severe_mean'].mean():.4f}\n")
            f.write(f"  Std P(severe): {self.state_results['p_severe_mean'].std():.4f}\n")
            f.write(f"  Min P(severe): {self.state_results['p_severe_mean'].min():.4f}\n")
            f.write(f"  Max P(severe): {self.state_results['p_severe_mean'].max():.4f}\n\n")
            
            f.write("Top 10 states by P(severe):\n")
            for i, row in self.state_results.head(10).iterrows():
                f.write(f"  {i+1:2d}. {row['state']:15s}: {row['p_severe_mean']:.4f} "
                       f"[{row['p_severe_ci_lower']:.4f}, {row['p_severe_ci_upper']:.4f}]\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("INTERPRETATION:\n")
            f.write("  P(severe) = Probability that an outage event is severe\n")
            f.write("  Severe = Duration > 2h AND Customer impact > 1% of county\n")
            f.write("  Use this metric to assess state-level risk in combination with\n")
            f.write("  frequency (lambda) and network vulnerability.\n")
        
        print(f"  ✓ Summary: {summary_file}")
        
        return self
    
    def run_full_pipeline(self, output_dir='./severity_results'):
        """Run the complete severity modeling pipeline."""
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Define severity thresholds
        self.define_severity_multiple_thresholds()
        
        # Step 3: Aggregate to state level
        self.aggregate_to_state_level()
        
        # Step 4: Analyze duration-customer relationship
        self.analyze_duration_customer_relationship(output_dir)
        
        # Step 5: Build Bayesian model
        self.build_interaction_model(severity_type='combined')
        
        # Step 6: Sample posterior
        self.sample_posterior()
        
        # Step 7: Extract results
        results = self.extract_state_severity()
        
        # Step 8: Plot
        self.plot_results(output_dir)
        
        # Step 9: Save
        self.save_results(output_dir)
        
        print("\n" + "="*60)
        print("✓ STEP 2B COMPLETE!")
        print("="*60)
        print(f"\nOutputs saved to: {output_dir}/")
        print(f"  1. state_severity_combined.csv - Main results")
        print(f"  2. state_severity_combined.png - Visualizations")
        print(f"  3. duration_customer_interaction.png - Interaction analysis")
        print(f"  4. state_correlations.csv - State-level correlations")
        print(f"  5. severity_summary.txt - Text summary")
        
        return results


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Step 2B: State-level severity modeling with interaction'
    )
    parser.add_argument(
        'data_path',
        help='Path to cleaned outage data CSV'
    )
    parser.add_argument(
        'population_path',
        help='Path to county population CSV'
    )
    parser.add_argument(
        '--output-dir',
        default='./severity_results',
        help='Output directory (default: ./severity_results)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    modeler = StateSeverityModeler(args.data_path, args.population_path)
    results = modeler.run_full_pipeline(args.output_dir)
    
    print(f"\n✓ Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
