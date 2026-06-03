"""
Example Usage Script
演示如何使用RealisticOutageSimulator的各种功能

This script demonstrates various ways to use the simulation framework.
"""

import os
import numpy as np
from realistic_outage_simulator import RealisticOutageSimulator, run_multi_state_comparison
from data_preparation import create_sample_data
from visualization import (
    plot_loss_distribution,
    plot_state_comparison,
    create_summary_dashboard
)


def example_1_basic_single_state():
    """
    Example 1: Basic single state analysis
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Single State Analysis")
    print("="*70)
    
    # Create sample data
    print("\nCreating sample data...")
    data_path = 'sample_california.pkl'
    create_sample_data(data_path)
    
    # Initialize simulator
    print("\nInitializing simulator...")
    simulator = RealisticOutageSimulator('California', data_path)
    
    # Run simulation with fewer iterations for demo
    print("\nRunning Monte Carlo simulation (100 iterations for demo)...")
    results = simulator.monte_carlo_simulation(n_simulations=100, verbose=False)
    
    # Print results
    print("\nResults:")
    print(f"  Mean Annual Loss: {results['mean_annual_loss']:.4f}")
    print(f"  Standard Deviation: {results['std_annual_loss']:.4f}")
    print(f"  95% VaR: {results['VaR_95']:.4f}")
    print(f"  95% CVaR: {results['CVaR_95']:.4f}")
    print(f"  Expected Events per Year: {results['mean_n_events']:.1f}")
    
    # Generate report
    print("\nGenerating detailed report...")
    report = simulator.generate_report(results)
    print("\nDetailed Report:")
    print(report.T.to_string())
    
    # Create visualization
    print("\nCreating visualization...")
    os.makedirs('results', exist_ok=True)
    plot_loss_distribution(results, 'California', 
                          save_path='results/example1_loss_dist.png',
                          show=False)
    print("✓ Visualization saved to: results/example1_loss_dist.png")


def example_2_different_recovery_models():
    """
    Example 2: Compare different recovery models
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparing Recovery Models")
    print("="*70)
    
    # Use existing sample data
    data_path = 'sample_california.pkl'
    if not os.path.exists(data_path):
        create_sample_data(data_path)
    
    simulator = RealisticOutageSimulator('California', data_path)
    
    # Test different recovery models
    print("\nTesting Constant Recovery Model...")
    results_constant = simulator.monte_carlo_simulation(
        n_simulations=100,
        recovery_model='constant',
        verbose=False
    )
    
    print("Testing Exponential Recovery Model...")
    results_exponential = simulator.monte_carlo_simulation(
        n_simulations=100,
        recovery_model='exponential',
        verbose=False
    )
    
    # Compare results
    print("\nComparison:")
    print(f"Constant Recovery:")
    print(f"  Mean Loss: {results_constant['mean_annual_loss']:.4f}")
    print(f"  95% VaR: {results_constant['VaR_95']:.4f}")
    print(f"\nExponential Recovery:")
    print(f"  Mean Loss: {results_exponential['mean_annual_loss']:.4f}")
    print(f"  95% VaR: {results_exponential['VaR_95']:.4f}")
    print(f"\nDifference:")
    diff = results_constant['mean_annual_loss'] - results_exponential['mean_annual_loss']
    print(f"  Mean Loss: {diff:.4f} ({diff/results_constant['mean_annual_loss']*100:.1f}%)")
    
    # Visualize
    from visualization import plot_sensitivity_analysis
    os.makedirs('results', exist_ok=True)
    plot_sensitivity_analysis(
        {'Constant Recovery': results_constant, 'Exponential Recovery': results_exponential},
        save_path='results/example2_recovery_comparison.png',
        show=False
    )
    print("\n✓ Comparison visualization saved to: results/example2_recovery_comparison.png")


def example_3_analyze_single_event():
    """
    Example 3: Analyze a single outage event in detail
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Single Outage Event Analysis")
    print("="*70)
    
    data_path = 'sample_california.pkl'
    if not os.path.exists(data_path):
        create_sample_data(data_path)
    
    simulator = RealisticOutageSimulator('California', data_path)
    
    # Simulate a single outage in San Francisco
    print("\nSimulating single outage event in San Francisco...")
    event = simulator.simulate_single_outage_event('San Francisco')
    
    print("\nEvent Details:")
    print(f"  County: {event['county']}")
    print(f"  Duration: {event['duration']:.2f} hours")
    print(f"  Affected Customers: {event['affected_customers']:,}")
    print(f"  Impact Radius: {event['radius_km']:.2f} km")
    print(f"  Charging Stations Affected: {event['affected_stations']}")
    print(f"  Efficiency Loss: {event['efficiency_loss']:.4f}")
    
    # Simulate multiple events to see variation
    print("\nSimulating 10 events to see variation...")
    losses = []
    for i in range(10):
        event = simulator.simulate_single_outage_event('San Francisco')
        losses.append(event['efficiency_loss'])
        print(f"  Event {i+1}: Loss = {event['efficiency_loss']:.4f}, "
              f"Duration = {event['duration']:.1f}h, "
              f"Stations = {event['affected_stations']}")
    
    print(f"\nStatistics across 10 events:")
    print(f"  Mean Loss: {np.mean(losses):.4f}")
    print(f"  Std Dev: {np.std(losses):.4f}")
    print(f"  Min Loss: {np.min(losses):.4f}")
    print(f"  Max Loss: {np.max(losses):.4f}")


def example_4_county_vulnerability():
    """
    Example 4: Analyze county-level vulnerability
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: County-Level Vulnerability Analysis")
    print("="*70)
    
    data_path = 'sample_california.pkl'
    if not os.path.exists(data_path):
        create_sample_data(data_path)
    
    simulator = RealisticOutageSimulator('California', data_path)
    
    print("\nAnalyzing vulnerability for each county...")
    county_losses = {}
    
    for county in simulator.lambda_county.keys():
        # Simulate multiple events per county to get expected loss
        losses = []
        for _ in range(20):
            event = simulator.simulate_single_outage_event(county)
            losses.append(event['efficiency_loss'])
        
        expected_loss = np.mean(losses)
        annual_expected_loss = expected_loss * simulator.lambda_county[county]
        
        county_losses[county] = annual_expected_loss
        
        print(f"\n{county}:")
        print(f"  Outage Rate: {simulator.lambda_county[county]:.2f} events/year")
        print(f"  Avg Loss per Event: {expected_loss:.4f}")
        print(f"  Annual Expected Loss: {annual_expected_loss:.4f}")
    
    # Rank counties
    print("\n" + "-"*70)
    print("County Vulnerability Ranking:")
    print("-"*70)
    ranked = sorted(county_losses.items(), key=lambda x: x[1], reverse=True)
    for i, (county, loss) in enumerate(ranked, 1):
        print(f"{i}. {county}: {loss:.4f}")
    
    # Visualize
    from visualization import plot_county_vulnerability
    os.makedirs('results', exist_ok=True)
    plot_county_vulnerability(
        simulator, 
        county_losses,
        save_path='results/example4_county_vulnerability.png',
        show=False
    )
    print("\n✓ County vulnerability plot saved to: results/example4_county_vulnerability.png")


def example_5_comprehensive_dashboard():
    """
    Example 5: Create comprehensive analysis dashboard
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Comprehensive Analysis Dashboard")
    print("="*70)
    
    data_path = 'sample_california.pkl'
    if not os.path.exists(data_path):
        create_sample_data(data_path)
    
    print("\nRunning comprehensive analysis...")
    simulator = RealisticOutageSimulator('California', data_path)
    results = simulator.monte_carlo_simulation(n_simulations=200, verbose=False)
    
    print("\nCreating comprehensive dashboard...")
    os.makedirs('results', exist_ok=True)
    create_summary_dashboard(
        results,
        'California',
        save_path='results/example5_comprehensive_dashboard.png',
        show=False
    )
    
    print("\n✓ Dashboard created: results/example5_comprehensive_dashboard.png")
    print("\nThe dashboard includes:")
    print("  - Loss distribution histogram")
    print("  - Statistical summary")
    print("  - Cumulative distribution function")
    print("  - Box plot")
    print("  - Q-Q plot for normality test")
    print("  - Kernel density estimation")


def example_6_extract_detailed_events():
    """
    Example 6: Extract and analyze detailed event logs
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Detailed Event Log Analysis")
    print("="*70)
    
    data_path = 'sample_california.pkl'
    if not os.path.exists(data_path):
        create_sample_data(data_path)
    
    simulator = RealisticOutageSimulator('California', data_path)
    
    print("\nSimulating one year of outages...")
    annual_result = simulator.simulate_annual_outages(verbose=True)
    
    print(f"\nAnnual Summary:")
    print(f"  Total Events: {annual_result['n_events']}")
    print(f"  Total Annual Loss: {annual_result['total_annual_loss']:.4f}")
    print(f"  Average Loss per Event: {annual_result['average_loss_per_event']:.4f}")
    
    # Analyze event log
    events = annual_result['event_log']
    
    if len(events) > 0:
        print(f"\nDetailed Event Analysis:")
        
        # Find worst event
        worst_event = max(events, key=lambda x: x['efficiency_loss'])
        print(f"\nWorst Event:")
        print(f"  County: {worst_event['county']}")
        print(f"  Duration: {worst_event['duration']:.2f} hours")
        print(f"  Affected Customers: {worst_event['affected_customers']:,}")
        print(f"  Impact Radius: {worst_event['radius_km']:.2f} km")
        print(f"  Stations Affected: {worst_event['affected_stations']}")
        print(f"  Loss: {worst_event['efficiency_loss']:.4f}")
        
        # Statistics
        durations = [e['duration'] for e in events]
        customers = [e['affected_customers'] for e in events]
        radii = [e['radius_km'] for e in events]
        
        print(f"\nEvent Statistics:")
        print(f"  Duration (hours):")
        print(f"    Mean: {np.mean(durations):.2f}")
        print(f"    Range: [{np.min(durations):.2f}, {np.max(durations):.2f}]")
        print(f"  Affected Customers:")
        print(f"    Mean: {np.mean(customers):.0f}")
        print(f"    Range: [{np.min(customers):.0f}, {np.max(customers):.0f}]")
        print(f"  Impact Radius (km):")
        print(f"    Mean: {np.mean(radii):.2f}")
        print(f"    Range: [{np.min(radii):.2f}, {np.max(radii):.2f}]")


def main():
    """
    Run all examples
    """
    print("\n" + "="*70)
    print("REALISTIC OUTAGE SIMULATOR - USAGE EXAMPLES")
    print("="*70)
    
    print("\nThis script demonstrates various features of the simulation framework.")
    print("Choose an example to run:")
    print()
    print("  1. Basic single state analysis")
    print("  2. Compare recovery models")
    print("  3. Analyze single events")
    print("  4. County vulnerability analysis")
    print("  5. Comprehensive dashboard")
    print("  6. Detailed event logs")
    print("  7. Run all examples")
    print()
    
    choice = input("Enter choice (1-7): ").strip()
    
    examples = {
        '1': example_1_basic_single_state,
        '2': example_2_different_recovery_models,
        '3': example_3_analyze_single_event,
        '4': example_4_county_vulnerability,
        '5': example_5_comprehensive_dashboard,
        '6': example_6_extract_detailed_events,
    }
    
    if choice in examples:
        examples[choice]()
    elif choice == '7':
        print("\nRunning all examples...")
        for i, func in examples.items():
            func()
            print("\n" + "="*70)
    else:
        print("Invalid choice.")
        return
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("Check the 'results/' directory for output files.")
    print("="*70)


if __name__ == "__main__":
    main()
