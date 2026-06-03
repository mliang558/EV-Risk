#!/usr/bin/env python3
"""
Quick data inspector - check what's in your data file
"""

import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_data.py cleaned_data_1117.csv")
    sys.exit(1)

data_path = sys.argv[1]

print("="*70)
print("DATA INSPECTOR")
print("="*70)

# Load data
print(f"\nLoading: {data_path}")
df = pd.read_csv(data_path)

print(f"\n📊 Basic Info:")
print(f"  Total rows: {len(df):,}")
print(f"  Columns: {list(df.columns)}")

print(f"\n🗺️  State column:")
if 'state' in df.columns:
    unique_states = df['state'].unique()
    print(f"  Unique states: {len(unique_states)}")
    print(f"  State names (first 20):")
    for state in sorted(unique_states)[:20]:
        count = (df['state'] == state).sum()
        print(f"    {state:30s}: {count:,} events")
    
    if len(unique_states) > 20:
        print(f"    ... and {len(unique_states) - 20} more states")
else:
    print("  ❌ No 'state' column found!")
    print(f"  Available columns: {list(df.columns)}")

print(f"\n🏘️  County column:")
if 'fips' in df.columns:
    print(f"  Unique FIPS codes: {df['fips'].nunique()}")
    print(f"  Sample FIPS codes: {df['fips'].unique()[:10].tolist()}")
else:
    print("  ❌ No 'fips' column found!")

print(f"\n📅 Time column:")
time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
if time_cols:
    print(f"  Found time columns: {time_cols}")
    for col in time_cols:
        print(f"    {col}: {df[col].iloc[0]} (sample)")
else:
    print("  ⚠️  No obvious time column found")

print(f"\n📋 First few rows:")
print(df.head(3))

print(f"\n" + "="*70)
print("RECOMMENDATIONS:")
print("="*70)

if 'state' in df.columns:
    # Suggest correct state name
    sample_state = sorted(df['state'].unique())[0]
    print(f"\n✅ Use this state name format: --states {sample_state}")
    print(f"\nExample command:")
    print(f"  python county_hierarchical_model.py {data_path} --states {sample_state} --draws 500 --tune 500")
    
    # Show top 5 states by event count
    top_states = df['state'].value_counts().head(5)
    print(f"\n📊 Top 5 states by event count:")
    for state, count in top_states.items():
        print(f"  {state:30s}: {count:,} events")
