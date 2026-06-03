"""
快速检查 area_km2 列
"""

import pandas as pd
import numpy as np

csv_path = input("Enter CSV path: ").strip()

print("\nLoading CSV...")
df = pd.read_csv(csv_path, encoding='latin-1')

print(f"\n{'='*70}")
print("COLUMN CHECK")
print(f"{'='*70}")

# 查找area相关列
area_cols = [col for col in df.columns if 'area' in col.lower()]
print(f"\nColumns containing 'area':")
for col in area_cols:
    print(f"  - '{col}'")

# 检查area_km2
if 'area_km2' in df.columns:
    print(f"\n{'='*70}")
    print("area_km2 ANALYSIS")
    print(f"{'='*70}")
    
    col = df['area_km2']
    
    print(f"\nData type: {col.dtype}")
    print(f"Total values: {len(col)}")
    print(f"Non-null: {col.notna().sum()}")
    print(f"Null: {col.isna().sum()} ({col.isna().sum()/len(col)*100:.1f}%)")
    
    # 尝试转换为数值
    if col.dtype == 'object':
        print(f"\n⚠ Column is object type (string), trying to convert...")
        col_numeric = pd.to_numeric(col, errors='coerce')
        newly_null = col_numeric.isna().sum() - col.isna().sum()
        if newly_null > 0:
            print(f"  → {newly_null} values couldn't be converted to numbers")
            print(f"  → Sample non-numeric values:")
            non_numeric = df[pd.to_numeric(col, errors='coerce').isna() & col.notna()]['area_km2'].head()
            for val in non_numeric:
                print(f"     '{val}'")
        col = col_numeric
    
    # 统计
    valid = col.dropna()
    if len(valid) > 0:
        print(f"\nValid numeric values: {len(valid)}")
        print(f"  Min: {valid.min():.2f}")
        print(f"  Max: {valid.max():.2f}")
        print(f"  Mean: {valid.mean():.2f}")
        print(f"  Sum: {valid.sum():,.2f}")
    
    # 检查异常
    if len(valid) > 0:
        print(f"\nAnomalies:")
        print(f"  Zero: {(col == 0).sum()}")
        print(f"  Negative: {(col < 0).sum()}")
        print(f"  > 1M km²: {(col > 1000000).sum()}")
    
    # 按州检查
    if 'STATE' in df.columns or 'STNAME' in df.columns:
        state_col = 'STNAME' if 'STNAME' in df.columns else 'STATE'
        print(f"\n{'='*70}")
        print("BY STATE")
        print(f"{'='*70}")
        
        state_stats = df.groupby(state_col).agg({
            'area_km2': ['count', lambda x: x.isna().sum(), 'sum']
        })
        state_stats.columns = ['Total', 'Missing', 'Sum_km2']
        state_stats['Missing_pct'] = (state_stats['Missing'] / state_stats['Total'] * 100).round(1)
        
        print("\nStates with missing area:")
        missing_states = state_stats[state_stats['Missing'] > 0].sort_values('Missing', ascending=False)
        if len(missing_states) > 0:
            print(missing_states.head(10))
        else:
            print("  None! All states have complete area data")
        
        # 检查康涅狄格
        ct_variations = ['Connecticut', 'CT', 'CONNECTICUT', '09']
        for ct in ct_variations:
            if ct in state_stats.index:
                print(f"\nConnecticut data:")
                print(state_stats.loc[ct])
                break

else:
    print(f"\n❌ 'area_km2' column not found!")
    print(f"\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
