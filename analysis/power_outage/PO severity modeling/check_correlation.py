#!/usr/bin/env python3
"""
检查Duration和Customer Percentage的相关性
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.special import logit
import warnings
import sys
warnings.filterwarnings('ignore')

print("="*80)
print("Duration vs Customer % 相关性分析")
print("="*80)

# 加载原始数据
print("\n[1/5] 加载原始数据...")

# 获取数据文件路径
if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = 'full_pop_2022.csv'

print(f"  数据文件: {data_path}")
df = pd.read_csv(data_path)

# 数据清理
df['duration_hr'] = df['duration_min'] / 60
df['fips'] = df['fips'].astype(str).str.zfill(5)
df = df.dropna(subset=['mean_customers', 'duration_min', 'POPESTIMATE2022'])
df = df[(df['duration_hr'] > 0) & (df['mean_customers'] > 0)]
df['customers_pct'] = (df['mean_customers'] / df['POPESTIMATE2022']).clip(0, 1)

print(f"  总样本数: {len(df):,}")
print(f"  Duration范围: {df['duration_hr'].min():.2f} - {df['duration_hr'].max():.2f} 小时")
print(f"  Customer %范围: {df['customers_pct'].min():.4f} - {df['customers_pct'].max():.4f}")

# 基础统计
print("\n[2/5] 基础统计")
print(f"\nDuration (小时):")
print(f"  均值: {df['duration_hr'].mean():.2f}")
print(f"  中位数: {df['duration_hr'].median():.2f}")
print(f"  标准差: {df['duration_hr'].std():.2f}")

print(f"\nCustomer %:")
print(f"  均值: {df['customers_pct'].mean():.4f} ({df['customers_pct'].mean()*100:.2f}%)")
print(f"  中位数: {df['customers_pct'].median():.4f} ({df['customers_pct'].median()*100:.2f}%)")
print(f"  标准差: {df['customers_pct'].std():.4f}")

# 相关性分析
print("\n[3/5] 相关性分析")
print("="*80)

# 原始尺度
pearson_r, p_pearson = pearsonr(df['duration_hr'], df['customers_pct'])
spearman_r, p_spearman = spearmanr(df['duration_hr'], df['customers_pct'])

print("\n原始尺度 (duration_hr vs customers_pct):")
print(f"  Pearson correlation:  {pearson_r:7.4f} (p={p_pearson:.3e})")
print(f"  Spearman correlation: {spearman_r:7.4f} (p={p_spearman:.3e})")

# 对数尺度
df_log = df[(df['duration_hr'] > 0) & (df['customers_pct'] > 0)].copy()
df_log['log_duration'] = np.log(df_log['duration_hr'])
df_log['log_customers_pct'] = np.log(df_log['customers_pct'])

pearson_log, p_log_p = pearsonr(df_log['log_duration'], df_log['log_customers_pct'])
spearman_log, p_log_s = spearmanr(df_log['log_duration'], df_log['log_customers_pct'])

print("\n对数尺度 (log(duration) vs log(customers_pct)):")
print(f"  Pearson correlation:  {pearson_log:7.4f} (p={p_log_p:.3e})")
print(f"  Spearman correlation: {spearman_log:7.4f} (p={p_log_s:.3e})")

# 解释
print("\n[4/5] 相关性解释")
print("="*80)

def interpret_correlation(r, name=""):
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = "几乎无相关"
        emoji = "✅"
    elif abs_r < 0.3:
        strength = "弱相关"
        emoji = "✅"
    elif abs_r < 0.5:
        strength = "中等相关"
        emoji = "⚠️"
    elif abs_r < 0.7:
        strength = "强相关"
        emoji = "❌"
    else:
        strength = "非常强相关"
        emoji = "❌❌"
    
    direction = "正" if r > 0 else "负"
    print(f"\n{emoji} {name}:")
    print(f"  r = {r:.4f} → {strength} ({direction}相关)")
    
    if abs_r < 0.3:
        print(f"  ✅ 共线性不严重，可以同时作为预测变量")
    elif abs_r < 0.5:
        print(f"  ⚠️  中等共线性，需要注意多重共线性问题")
        print(f"     建议：检查VIF，或使用正则化")
    else:
        print(f"  ❌ 严重共线性！不应同时作为预测变量")
        print(f"     建议：只用其中一个，或用PCA降维")
    
    return abs_r

r1 = interpret_correlation(pearson_r, "原始尺度 Pearson")
r2 = interpret_correlation(spearman_r, "原始尺度 Spearman")
r3 = interpret_correlation(pearson_log, "对数尺度 Pearson")

# 决策建议
print("\n" + "="*80)
print("建模建议")
print("="*80)

max_r = max(r1, r2, r3)

if max_r < 0.3:
    print("\n✅ 好消息！")
    print("Duration和Customer%相关性较弱，可以继续当前模型架构。")
    print("但仍需注意：你用它们定义了'severe'，存在循环论证问题。")
    print("\n建议：")
    print("  1. 重新框架化为描述性分析（不强调因果）")
    print("  2. 或改用独立的响应变量（经济损失、投诉数等）")
    
elif max_r < 0.5:
    print("\n⚠️  注意！")
    print("Duration和Customer%存在中等程度相关性。")
    print("加上循环论证问题，当前模型的解释性受限。")
    print("\n建议：")
    print("  1. 计算VIF (Variance Inflation Factor)")
    print("  2. 考虑只用一个变量，或用它们的交互项")
    print("  3. 最好：改用独立的响应变量")
    
else:
    print("\n❌ 严重问题！")
    print("Duration和Customer%高度相关！")
    print("这意味着：")
    print("  1. 它们包含高度重叠的信息")
    print("  2. 同时放入模型会导致严重的多重共线性")
    print("  3. 系数估计不稳定，解释性差")
    print("\n强烈建议：")
    print("  1. 不要同时使用它们作为预测变量")
    print("  2. 重新设计模型（见modeling_alternatives.txt）")
    print("  3. 考虑用它们的组合（如duration × customers）作为新的响应变量")

# 可视化
print("\n[5/5] 生成可视化...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 子图1: 原始尺度散点图
ax = axes[0, 0]
ax.scatter(df['duration_hr'], df['customers_pct']*100, alpha=0.3, s=10)
ax.set_xlabel('Duration (hours)', fontsize=11)
ax.set_ylabel('Customer Impact (%)', fontsize=11)
ax.set_title(f'Raw Scale\nPearson r = {pearson_r:.3f}', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# 子图2: 对数-对数尺度
ax = axes[0, 1]
ax.scatter(df['duration_hr'], df['customers_pct']*100, alpha=0.3, s=10)
ax.set_xlabel('Duration (hours)', fontsize=11)
ax.set_ylabel('Customer Impact (%)', fontsize=11)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(f'Log-Log Scale\nPearson r = {pearson_log:.3f}', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# 子图3: 密度热图
ax = axes[0, 2]
# 对数尺度的2D直方图
h = ax.hist2d(np.log10(df['duration_hr']+0.1), 
               np.log10(df['customers_pct']*100+0.01), 
               bins=50, cmap='YlOrRd', cmin=1)
ax.set_xlabel('Log10(Duration hours)', fontsize=11)
ax.set_ylabel('Log10(Customer %)', fontsize=11)
ax.set_title('Density Heatmap', fontsize=12, fontweight='bold')
plt.colorbar(h[3], ax=ax, label='Count')

# 子图4: Duration分布
ax = axes[1, 0]
ax.hist(df['duration_hr'], bins=100, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Duration (hours)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Duration Distribution', fontsize=12, fontweight='bold')
ax.set_xlim(0, df['duration_hr'].quantile(0.95))
ax.grid(alpha=0.3)

# 子图5: Customer % 分布
ax = axes[1, 1]
ax.hist(df['customers_pct']*100, bins=100, color='orange', alpha=0.7, edgecolor='black')
ax.set_xlabel('Customer Impact (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Customer Impact Distribution', fontsize=12, fontweight='bold')
ax.set_xlim(0, df['customers_pct'].quantile(0.95)*100)
ax.grid(alpha=0.3)

# 子图6: 分位数散点图（减少过度绘制）
ax = axes[1, 2]
# 按duration分组，计算customer%的中位数
duration_bins = pd.qcut(df['duration_hr'], q=50, duplicates='drop')
grouped = df.groupby(duration_bins)['customers_pct'].agg(['median', 'mean', 'std', 'count'])
bin_centers = df.groupby(duration_bins)['duration_hr'].mean()

ax.errorbar(bin_centers, grouped['median']*100, 
            yerr=grouped['std']*100, 
            fmt='o-', color='steelblue', capsize=3, alpha=0.7,
            label='Median ± SD')
ax.set_xlabel('Duration (hours)', fontsize=11)
ax.set_ylabel('Customer Impact (%)', fontsize=11)
ax.set_title('Median Customer % by Duration Bin', fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle('Duration vs Customer Impact: Correlation Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ correlation_analysis.png")

# 额外分析：按严重程度阈值分组
print("\n" + "="*80)
print("额外分析：按严重程度阈值分层")
print("="*80)

thresholds = [
    ('Tier 1', 2, 0.005),
    ('Tier 2', 4, 0.01),
    ('Tier 3', 12, 0.02),
    ('Tier 4', 24, 0.05),
]

print("\n在不同阈值下，Duration和Customer%在'严重'组的相关性：")
print("-"*80)

for name, dur_thresh, pct_thresh in thresholds:
    severe = df[(df['duration_hr'] >= dur_thresh) & (df['customers_pct'] >= pct_thresh)]
    if len(severe) > 10:
        r, p = pearsonr(severe['duration_hr'], severe['customers_pct'])
        pct = len(severe) / len(df) * 100
        print(f"\n{name} (≥{dur_thresh}h & ≥{pct_thresh*100}%):")
        print(f"  样本数: {len(severe):,} ({pct:.2f}%)")
        print(f"  相关系数: {r:.4f} (p={p:.3e})")

print("\n" + "="*80)
print("✅ 分析完成！")
print("="*80)
print("\n查看可视化: correlation_analysis.png")
print("查看建议: modeling_alternatives.txt")
