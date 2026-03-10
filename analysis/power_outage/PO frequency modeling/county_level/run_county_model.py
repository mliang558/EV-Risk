#!/usr/bin/env python3
"""
Quick run script for county-level model
"""

from county_to_state_modeling import CountyStateFrequencyModeler

def main():
    print("="*60)
    print("COUNTY-LEVEL BAYESIAN MODEL")
    print("Aggregating to State Level for Fair Comparison")
    print("="*60)
    
    # Configuration
    DATA_PATH = "cleaned_data_1117.csv"
    OUTPUT_DIR = "./county_model_results"
    
    # For quick testing, use fewer samples
    # For final analysis, use n_samples=2000, n_tune=1000
    N_SAMPLES = 500   # Quick test (5-10 min)
    N_TUNE = 500      # Quick test
    
    print(f"\nSettings:")
    print(f"  Data: {DATA_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Samples: {N_SAMPLES} (tune: {N_TUNE})")
    print(f"  ⚠️  This will take 10-30 minutes for 3000+ counties")
    
    # Initialize modeler
    modeler = CountyStateFrequencyModeler(DATA_PATH)
    
    # Run full pipeline
    results = modeler.run_full_pipeline(
        output_dir=OUTPUT_DIR,
        n_samples=N_SAMPLES,
        n_tune=N_TUNE
    )
    
    # Display key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    print("\nTop 5 states by TOTAL frequency (absolute risk):")
    top_total = results.nlargest(5, 'lambda_total_mean')
    for _, row in top_total.iterrows():
        print(f"  {row['state']:15s}: {row['lambda_total_mean']:7.1f} events/year "
              f"({row['n_counties']:3d} counties)")
    
    print("\nTop 5 states by PER-COUNTY frequency (comparable risk):")
    top_per_county = results.nlargest(5, 'lambda_per_county_mean')
    for _, row in top_per_county.iterrows():
        print(f"  {row['state']:15s}: {row['lambda_per_county_mean']:5.2f} events/county/year "
              f"({row['n_counties']:3d} counties)")
    
    print(f"\n✓ Full results saved to: {OUTPUT_DIR}/")
    print(f"  - state_level_from_county_model.csv")
    print(f"  - state_comparison_county_model.png")
    print(f"  - state_comparison_summary.txt")


if __name__ == '__main__':
    main()
