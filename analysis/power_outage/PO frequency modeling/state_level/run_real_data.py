#!/usr/bin/env python3
"""
Simple script to run state frequency modeling with your real data.
Just configure the paths and run!
"""

from state_frequency_modeling import StateFrequencyModeler

def main():
    """Run modeling on real power outage data."""
    
    # ============================================================
    # CONFIGURATION - Edit these paths
    # ============================================================
    
    # Path to your cleaned data file
    DATA_PATH = "cleaned_data_1117.csv"
    
    # Where to save results
    OUTPUT_DIR = "./real_results"
    
    # ============================================================
    # Run the pipeline
    # ============================================================
    
    print("="*60)
    print("Running State Frequency Modeling")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # Initialize modeler (will use config.yaml if it exists)
    modeler = StateFrequencyModeler(
        data_path=DATA_PATH,
        config_path='config.yaml'  # Or None to use defaults
    )
    
    # Run the full pipeline
    results = modeler.run_full_pipeline(output_dir=OUTPUT_DIR)
    
    # Display results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nTop 5 states by outage frequency:")
    print(results[['state', 'lambda_mean', 'n_events']].head(5).to_string(index=False))
    
    print(f"\n✓ All results saved to: {OUTPUT_DIR}/")
    print(f"  - state_frequency_estimates.csv")
    print(f"  - summary_statistics.txt")
    print(f"  - 4 visualization plots")


if __name__ == '__main__':
    main()
