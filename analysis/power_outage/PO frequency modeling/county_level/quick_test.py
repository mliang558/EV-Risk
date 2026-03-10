#!/usr/bin/env python3
"""
FAST TEST VERSION - County Hierarchical Model
==============================================
Quick test run with reduced parameters for faster iteration
"""

import subprocess
import sys

print("="*70)
print("FAST TEST CONFIGURATION")
print("="*70)
print("\nThis version uses:")
print("  • draws=500 (instead of 1000)")
print("  • tune=500 (instead of 1000)")
print("  • Should complete in ~30-60 minutes instead of 4-5 hours")
print("\nFor production runs, use the full county_hierarchical_model.py")
print("="*70)

# Get data path
if len(sys.argv) < 2:
    print("\nUsage:")
    print("  python quick_test.py cleaned_data_1117.csv")
    print("  python quick_test.py cleaned_data_1117.csv --states TX CA FL")
    sys.exit(1)

data_path = sys.argv[1]

# Build command
cmd = [
    "python",
    "county_hierarchical_model.py",
    data_path,
    "--draws", "500",
    "--tune", "500",
    "--output-dir", "./quick_test_results"
]

# Add state filter if provided
if "--states" in sys.argv:
    states_idx = sys.argv.index("--states")
    states = sys.argv[states_idx+1:]
    cmd.extend(["--states"] + states)
    print(f"\n🎯 Filtering to states: {states}")

print(f"\nRunning: {' '.join(cmd)}\n")

# Run
subprocess.run(cmd)
