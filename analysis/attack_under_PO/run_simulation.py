"""
Main Simulation Runner
运行完整的停电影响模拟分析

This is the main script to run the complete analysis pipeline.
"""

import sys
from realistic_outage_simulator import RealisticOutageSimulator, run_multi_state_comparison
from visualization import (
    plot_loss_distribution,
    plot_state_comparison,
    plot_vulnerability_ranking,
    plot_risk_metrics,
    create_summary_dashboard
)
import pandas as pd


def run_single_state_analysis(state_name: str, data_path: str,
                              n_simulations: int = 1000,
                              recovery_model: str = 'constant',
                              output_dir: str = 'results'):
    """
    Run complete analysis for a single state
    
    Parameters:
    -----------
    state_name : str
        State name
    data_path : str
        Path to prepared data pickle file
    n_simulations : int
        Number of Monte Carlo simulations
    recovery_model : str
        Recovery model: 'constant' or 'exponential'
    output_dir : str
        Directory to save results
    """
    
    print("\n" + "="*70)
    print(f"SINGLE STATE ANALYSIS: {state_name}")
    print("="*70)
    
    # Initialize simulator
    print("\nStep 1: Initializing simulator...")
    simulator = RealisticOutageSimulator(state_name, data_path)
    
    # Run Monte Carlo simulation
    print(f"\nStep 2: Running {n_simulations} Monte Carlo simulations...")
    mc_results = simulator.monte_carlo_simulation(
        n_simulations=n_simulations,
        recovery_model=recovery_model,
        verbose=True
    )
    
    # Generate report
    print("\nStep 3: Generating statistical report...")
    report = simulator.generate_report(mc_results)
    
    # Print summary
    print("\n" + "-"*70)
    print("SIMULATION RESULTS SUMMARY")
    print("-"*70)
    print(report.T.to_string())
    print("-"*70)
    
    # Save report
    report_path = f'{output_dir}/{state_name}_report.csv'
    report.to_csv(report_path, index=False)
    print(f"\n✓ Report saved to: {report_path}")
    
    # Create visualizations
    print("\nStep 4: Creating visualizations...")
    
    # Loss distribution
    plot_loss_distribution(
        mc_results, 
        state_name,
        save_path=f'{output_dir}/{state_name}_loss_distribution.png',
        show=False
    )
    
    # Summary dashboard
    create_summary_dashboard(
        mc_results,
        state_name,
        save_path=f'{output_dir}/{state_name}_dashboard.png',
        show=False
    )
    
    print("\n" + "="*70)
    print(f"ANALYSIS COMPLETE FOR {state_name}")
    print(f"Results saved to: {output_dir}/")
    print("="*70)
    
    return simulator, mc_results, report


def run_multi_state_analysis(state_configs: dict,
                             n_simulations: int = 1000,
                             recovery_model: str = 'constant',
                             output_dir: str = 'results'):
    """
    Run comparative analysis across multiple states
    
    Parameters:
    -----------
    state_configs : dict
        Dictionary mapping state names to data paths
        Format: {'California': 'data/ca.pkl', 'Texas': 'data/tx.pkl', ...}
    n_simulations : int
        Number of Monte Carlo simulations per state
    recovery_model : str
        Recovery model
    output_dir : str
        Directory to save results
    """
    
    print("\n" + "="*70)
    print(f"MULTI-STATE COMPARATIVE ANALYSIS")
    print(f"States to analyze: {', '.join(state_configs.keys())}")
    print("="*70)
    
    # Run comparison
    print("\nRunning comparative analysis...")
    comparison_df = run_multi_state_comparison(
        state_configs,
        n_simulations=n_simulations,
        recovery_model=recovery_model
    )
    
    # Print comparison table
    print("\n" + "-"*70)
    print("COMPARATIVE RESULTS")
    print("-"*70)
    display_columns = [
        'State', 'Mean_Annual_Loss', 'VaR_95', 'Loss_Per_Station',
        'Composite_Normalized_Loss', 'Vulnerability_Rank'
    ]
    print(comparison_df[display_columns].to_string(index=False))
    print("-"*70)
    
    # Save comparison report
    comparison_path = f'{output_dir}/multi_state_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Comparison report saved to: {comparison_path}")
    
    # Create visualizations
    print("\nCreating comparative visualizations...")
    
    # State comparison
    plot_state_comparison(
        comparison_df,
        save_path=f'{output_dir}/state_comparison.png',
        show=False
    )
    
    # Vulnerability ranking
    plot_vulnerability_ranking(
        comparison_df,
        save_path=f'{output_dir}/vulnerability_ranking.png',
        show=False
    )
    
    # Risk metrics
    plot_risk_metrics(
        comparison_df,
        save_path=f'{output_dir}/risk_metrics_comparison.png',
        show=False
    )
    
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print("="*70)
    
    return comparison_df


def run_sensitivity_analysis(state_name: str, data_path: str,
                             scenarios: dict = None,
                             n_simulations: int = 500,
                             output_dir: str = 'results'):
    """
    Run sensitivity analysis with different scenarios
    
    Parameters:
    -----------
    state_name : str
        State name
    data_path : str
        Path to prepared data
    scenarios : dict
        Dictionary of scenario names and recovery models
        If None, uses default scenarios
    n_simulations : int
        Number of simulations per scenario
    output_dir : str
        Directory to save results
    """
    
    if scenarios is None:
        scenarios = {
            'Constant Recovery': 'constant',
            'Exponential Recovery': 'exponential'
        }
    
    print("\n" + "="*70)
    print(f"SENSITIVITY ANALYSIS: {state_name}")
    print(f"Scenarios: {', '.join(scenarios.keys())}")
    print("="*70)
    
    simulator = RealisticOutageSimulator(state_name, data_path)
    
    results = {}
    
    for scenario_name, recovery_model in scenarios.items():
        print(f"\nRunning scenario: {scenario_name}...")
        
        mc_results = simulator.monte_carlo_simulation(
            n_simulations=n_simulations,
            recovery_model=recovery_model,
            verbose=False
        )
        
        results[scenario_name] = mc_results
        
        print(f"  Mean Loss: {mc_results['mean_annual_loss']:.4f}")
        print(f"  95% VaR: {mc_results['VaR_95']:.4f}")
    
    # Create comparison visualization
    from visualization import plot_sensitivity_analysis
    
    plot_sensitivity_analysis(
        results,
        save_path=f'{output_dir}/{state_name}_sensitivity_analysis.png',
        show=False
    )
    
    # Save results
    sensitivity_df = pd.DataFrame({
        'Scenario': list(results.keys()),
        'Mean_Loss': [r['mean_annual_loss'] for r in results.values()],
        'Std_Loss': [r['std_annual_loss'] for r in results.values()],
        'VaR_95': [r['VaR_95'] for r in results.values()],
        'CVaR_95': [r['CVaR_95'] for r in results.values()]
    })
    
    sensitivity_path = f'{output_dir}/{state_name}_sensitivity.csv'
    sensitivity_df.to_csv(sensitivity_path, index=False)
    print(f"\n✓ Sensitivity analysis saved to: {sensitivity_path}")
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)
    
    return results


def main():
    """
    Main function - run complete analysis pipeline
    """
    
    import os
    
    # Create results directory
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}/")
    
    print("\n" + "="*70)
    print("REALISTIC OUTAGE IMPACT SIMULATION")
    print("="*70)
    
    print("\nChoose analysis type:")
    print("  1. Single state analysis")
    print("  2. Multi-state comparison")
    print("  3. Sensitivity analysis")
    print("  4. Run example with sample data")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        # Single state analysis
        state_name = input("Enter state name: ").strip()
        data_path = input("Enter data file path: ").strip()
        n_sims = int(input("Number of simulations (default 1000): ").strip() or "1000")
        
        run_single_state_analysis(
            state_name=state_name,
            data_path=data_path,
            n_simulations=n_sims,
            output_dir=output_dir
        )
    
    elif choice == "2":
        # Multi-state comparison
        n_states = int(input("How many states to compare? ").strip())
        state_configs = {}
        
        for i in range(n_states):
            print(f"\nState {i+1}:")
            name = input("  State name: ").strip()
            path = input("  Data file path: ").strip()
            state_configs[name] = path
        
        n_sims = int(input("\nNumber of simulations per state (default 1000): ").strip() or "1000")
        
        run_multi_state_analysis(
            state_configs=state_configs,
            n_simulations=n_sims,
            output_dir=output_dir
        )
    
    elif choice == "3":
        # Sensitivity analysis
        state_name = input("Enter state name: ").strip()
        data_path = input("Enter data file path: ").strip()
        n_sims = int(input("Number of simulations per scenario (default 500): ").strip() or "500")
        
        run_sensitivity_analysis(
            state_name=state_name,
            data_path=data_path,
            n_simulations=n_sims,
            output_dir=output_dir
        )
    
    elif choice == "4":
        # Run example with sample data
        print("\nRunning example analysis with sample data...")
        print("First, creating sample data...")
        
        from data_preparation import create_sample_data
        sample_path = 'sample_state_data.pkl'
        create_sample_data(sample_path)
        
        print("\nRunning single state analysis on sample data...")
        run_single_state_analysis(
            state_name='California (Sample)',
            data_path=sample_path,
            n_simulations=100,  # Fewer simulations for demo
            output_dir=output_dir
        )
        
        print("\n✓ Example analysis complete!")
        print(f"Check the '{output_dir}/' directory for results.")
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE")
    print(f"All results saved to: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
