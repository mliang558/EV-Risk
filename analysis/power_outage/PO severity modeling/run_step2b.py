#!/usr/bin/env python3
"""
Simple Run Script for Step 2B: Severity Modeling
=================================================
"""

from step2b_severity_final import StateSeverityModeler

# ============================================================
# CONFIGURATION
# ============================================================

# Input files
DATA_PATH = "cleaned_data_1117.csv"
POPULATION_PATH = "co-est2024-alldata.csv"  # 修改为你的population文件路径

# Output directory
OUTPUT_DIR = "./step2b_severity_results"

# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':
    print("="*70)
    print(" " * 15 + "STEP 2B: SEVERITY MODELING")
    print(" " * 10 + "Duration-Customer Interaction Analysis")
    print("="*70)
    
    print(f"\n📁 Inputs:")
    print(f"  - Outage data: {DATA_PATH}")
    print(f"  - Population: {POPULATION_PATH}")
    print(f"\n📁 Output: {OUTPUT_DIR}/")
    print()
    
    # Initialize and run
    modeler = StateSeverityModeler(DATA_PATH, POPULATION_PATH)
    results = modeler.run_full_pipeline(OUTPUT_DIR)
    
    # Display key results
    print("\n" + "="*70)
    print("📊 KEY RESULTS")
    print("="*70)
    
    print("\nTop 10 States by Severity Probability:")
    print("-" * 70)
    print(f"{'Rank':<6} {'State':<15} {'P(severe)':<12} {'95% CI':<20} {'# Events':<10}")
    print("-" * 70)
    
    for rank, (_, row) in enumerate(results.head(10).iterrows(), 1):
        ci_str = f"[{row['p_severe_ci_lower']:.3f}, {row['p_severe_ci_upper']:.3f}]"
        print(f"{rank:<6} {row['state']:<15} {row['p_severe_mean']:<12.3f} {ci_str:<20} {row['n_events']:<10.0f}")
    
    print("\n" + "="*70)
    print("✅ STEP 2B COMPLETE!")
    print("="*70)
    
    print(f"\n📂 Check outputs in: {OUTPUT_DIR}/")
    print("   ✓ state_severity_combined.csv")
    print("   ✓ state_severity_combined.png") 
    print("   ✓ duration_customer_interaction.png")
    print("   ✓ state_correlations.csv")
    print("   ✓ severity_summary.txt")
    
    print("\n💡 Next step: Combine with Step 2A (frequency) for risk scoring")
