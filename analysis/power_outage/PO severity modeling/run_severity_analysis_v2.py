#!/usr/bin/env python3
"""
县级停电严重程度分析 - 改进版
- 添加2小时阈值
- 更清晰的可视化
- 阈值选择指南
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_and_prepare_data(data_path='full_pop_2022.csv'):
    """加载并准备数据"""
    print("="*80)
    print("县级停电严重程度分析 - 改进版")
    print("="*80)
    
    print(f"\n[1/3] 加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  总记录数: {len(df):,}")
    
    # 转换时间和计算时长
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['duration_hr'] = df['duration_min'] / 60
    df['fips'] = df['fips'].astype(str).str.zfill(5)
    
    # 清理数据
    print(f"\n[2/3] 数据清理")
    before = len(df)
    df = df.dropna(subset=['mean_customers', 'duration_min', 'POPESTIMATE2022'])
    df = df[(df['duration_hr'] > 0) & (df['mean_customers'] > 0)]
    print(f"  删除异常值: {before - len(df):,} 条")
    print(f"  保留记录: {len(df):,} 条")
    print(f"  县数量: {df['fips'].nunique()}")
    print(f"  州数量: {df['state'].nunique()}")
    
    # 计算受影响客户百分比
    df['customers_pct'] = (df['mean_customers'] / df['POPESTIMATE2022']).clip(0, 1)
    
    print(f"\n[3/3] 数据分布（用于指导阈值选择）")
    print(f"\n  停电时长 (小时):")
    for p in [50, 75, 90, 95, 99]:
        val = df['duration_hr'].quantile(p/100)
        print(f"    {p:2d}分位: {val:6.2f}小时")
    
    print(f"\n  受影响客户百分比:")
    for p in [50, 75, 90, 95, 99]:
        val = df['customers_pct'].quantile(p/100)
        print(f"    {p:2d}分位: {val:.4f} ({val*100:5.2f}%)")
    
    return df


def define_core_severity_levels(df):
    """
    定义核心严重程度标准（减少组合数量）
    
    策略：
    1. 单一条件：仅用于探索性分析
    2. 组合条件：4个推荐标准（实际使用）
    """
    print("\n" + "="*80)
    print("定义核心严重程度标准")
    print("="*80)
    
    severity_definitions = []
    
    # ========================================================================
    # 第1部分：探索性标准（单一条件）
    # ========================================================================
    print("\n第1部分: 探索性标准 (单一条件 - 用于理解数据分布)")
    print("-" * 80)
    
    # 时长标准（5个关键点）
    duration_thresholds = [2, 4, 8, 12, 24]
    print("\n基于时长:")
    for dur in duration_thresholds:
        col_name = f'dur_{int(dur)}h'
        df[col_name] = (df['duration_hr'] >= dur).astype(int)
        n = df[col_name].sum()
        pct = (n / len(df)) * 100
        severity_definitions.append({
            'category': 'exploratory',
            'type': 'duration',
            'name': col_name,
            'duration_hr': dur,
            'customer_pct': None,
            'n_severe': n,
            'pct_severe': pct
        })
        print(f"  ≥{dur:2d}小时: {n:8,} ({pct:5.2f}%)")
    
    # 客户百分比标准（5个关键点）
    pct_thresholds = [0.005, 0.01, 0.02, 0.05, 0.10]
    print("\n基于客户百分比:")
    for pct in pct_thresholds:
        col_name = f'pct_{int(pct*100)}'
        df[col_name] = (df['customers_pct'] >= pct).astype(int)
        n = df[col_name].sum()
        pct_events = (n / len(df)) * 100
        severity_definitions.append({
            'category': 'exploratory',
            'type': 'percentage',
            'name': col_name,
            'duration_hr': None,
            'customer_pct': pct,
            'n_severe': n,
            'pct_severe': pct_events
        })
        print(f"  ≥{pct*100:4.1f}%: {n:8,} ({pct_events:5.2f}%)")
    
    # ========================================================================
    # 第2部分：推荐标准（组合条件）- 这是重点！
    # ========================================================================
    print("\n" + "="*80)
    print("第2部分: 推荐标准 (组合条件 - 实际使用) ⭐")
    print("="*80)
    
    recommended_standards = [
        # (时长, 百分比, 名称, 描述, 使用场景)
        (2,  0.005, 'tier1_alert',      '🟢 Tier 1 - 快速预警',    '捕获需要关注的事件'),
        (4,  0.01,  'tier2_moderate',   '🟡 Tier 2 - 中等影响',    '需要积极响应的事件'),
        (12, 0.02,  'tier3_serious',    '🟠 Tier 3 - 严重事件',    '需要紧急处理的事件'),
        (24, 0.05,  'tier4_critical',   '🔴 Tier 4 - 重大灾害',    '灾难性事件，最高优先级'),
    ]
    
    print("\n推荐的4级严重程度体系:")
    print("-" * 80)
    print(f"{'等级':<20} {'条件':<25} {'使用场景':<30}")
    print("-" * 80)
    
    for dur, pct, name, desc, usage in recommended_standards:
        col_name = name
        df[col_name] = (
            (df['duration_hr'] >= dur) & 
            (df['customers_pct'] >= pct)
        ).astype(int)
        n = df[col_name].sum()
        pct_events = (n / len(df)) * 100
        
        severity_definitions.append({
            'category': 'recommended',
            'type': 'combined',
            'name': col_name,
            'duration_hr': dur,
            'customer_pct': pct,
            'n_severe': n,
            'pct_severe': pct_events,
            'description': desc,
            'usage': usage
        })
        
        print(f"{desc:<20} ≥{dur}h & ≥{pct*100:.1f}%    {usage:<30}")
        print(f"{'':20} → {n:,} 事件 ({pct_events:.2f}%)")
        print()
    
    print("="*80)
    print(f"总结: {len(severity_definitions)} 个标准")
    print(f"  - 探索性标准: 10 个（用于理解数据）")
    print(f"  - 推荐标准: 4 个（用于实际应用）⭐")
    print("="*80)
    
    return df, severity_definitions, recommended_standards


def aggregate_to_county(df, severity_definitions, min_events=5):
    """聚合到县级别"""
    print("\n" + "="*80)
    print("聚合到县级别")
    print("="*80)
    
    severity_cols = [d['name'] for d in severity_definitions]
    
    agg_dict = {
        'state': 'first',
        'county': 'first',
        'fips': 'count',
        'duration_hr': ['mean', 'median', 'max'],
        'mean_customers': ['mean', 'median', 'max'],
        'customers_pct': ['mean', 'median', 'max'],
        'POPESTIMATE2022': 'first'
    }
    
    for col in severity_cols:
        agg_dict[col] = 'sum'
    
    county_agg = df.groupby('fips').agg(agg_dict).reset_index()
    
    # 展平列名
    county_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in county_agg.columns]
    
    county_agg = county_agg.rename(columns={
        'fips_count': 'n_events',
        'state_first': 'state',
        'county_first': 'county',
        'POPESTIMATE2022_first': 'population'
    })
    
    # 计算发生率
    for col in severity_cols:
        if col in county_agg.columns:
            county_agg[f'{col}_rate'] = county_agg[col] / county_agg['n_events']
    
    county_agg = county_agg[county_agg['n_events'] >= min_events].copy()
    county_agg = county_agg.sort_values('n_events', ascending=False).reset_index(drop=True)
    
    print(f"\n  县数量: {len(county_agg)}")
    print(f"  总事件数: {county_agg['n_events'].sum():,}")
    
    return county_agg


def create_threshold_selection_guide(df, severity_definitions, output_path):
    """创建阈值选择可视化指南"""
    print("\n" + "="*80)
    print("生成阈值选择指南")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 图1: 时长分布 + 阈值线
    ax = axes[0, 0]
    ax.hist(df['duration_hr'].clip(upper=50), bins=100, edgecolor='black', 
           alpha=0.7, color='skyblue')
    for dur in [2, 4, 12, 24]:
        pct = (df['duration_hr'] >= dur).mean() * 100
        ax.axvline(dur, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(dur, ax.get_ylim()[1]*0.9, f'{dur}h\n({pct:.1f}%)', 
               ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.set_xlabel('Duration (hours, capped at 50)', fontsize=11)
    ax.set_ylabel('Number of Events', fontsize=11)
    ax.set_title('Duration Distribution with Thresholds', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 图2: 客户百分比分布 + 阈值线
    ax = axes[0, 1]
    ax.hist(df['customers_pct'].clip(upper=0.2)*100, bins=100, 
           edgecolor='black', alpha=0.7, color='lightcoral')
    for pct in [0.5, 1, 2, 5]:
        pct_val = pct / 100
        pct_events = (df['customers_pct'] >= pct_val).mean() * 100
        ax.axvline(pct, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(pct, ax.get_ylim()[1]*0.9, f'{pct}%\n({pct_events:.1f}%)', 
               ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.set_xlabel('Customer Impact (% of county, capped at 20%)', fontsize=11)
    ax.set_ylabel('Number of Events', fontsize=11)
    ax.set_title('Customer Impact Distribution with Thresholds', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 图3: 2D散点图 - 时长 vs 客户百分比
    ax = axes[0, 2]
    sample = df.sample(min(10000, len(df)), random_state=42)
    ax.scatter(sample['duration_hr'].clip(upper=50), 
              sample['customers_pct'].clip(upper=0.2)*100,
              alpha=0.3, s=10, c='gray')
    
    # 画出4个推荐标准的边界
    colors = ['green', 'gold', 'orange', 'red']
    labels = ['Tier 1 (2h, 0.5%)', 'Tier 2 (4h, 1%)', 
             'Tier 3 (12h, 2%)', 'Tier 4 (24h, 5%)']
    thresholds = [(2, 0.5), (4, 1), (12, 2), (24, 5)]
    
    for (dur, pct), color, label in zip(thresholds, colors, labels):
        ax.axvline(dur, color=color, linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(pct, color=color, linestyle='--', linewidth=2, alpha=0.7)
        ax.plot(dur, pct, 'o', color=color, markersize=10, label=label)
    
    ax.set_xlabel('Duration (hours, capped at 50)', fontsize=11)
    ax.set_ylabel('Customer Impact (%, capped at 20%)', fontsize=11)
    ax.set_title('Duration vs Impact (4-Tier Boundaries)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # 图4: 不同阈值下的事件捕获率
    ax = axes[1, 0]
    
    # 提取单一条件的数据
    dur_data = [(d['duration_hr'], d['pct_severe']) 
                for d in severity_definitions if d['type'] == 'duration']
    pct_data = [(d['customer_pct']*100, d['pct_severe']) 
                for d in severity_definitions if d['type'] == 'percentage']
    
    if dur_data:
        dur_x, dur_y = zip(*sorted(dur_data))
        ax.plot(dur_x, dur_y, 'o-', linewidth=2, markersize=8, 
               label='Duration only', color='steelblue')
    
    ax2 = ax.twiny()
    if pct_data:
        pct_x, pct_y = zip(*sorted(pct_data))
        ax2.plot(pct_x, pct_y, 's-', linewidth=2, markersize=8,
                label='Customer % only', color='coral')
    
    ax.set_xlabel('Duration Threshold (hours)', fontsize=11, color='steelblue')
    ax2.set_xlabel('Customer % Threshold', fontsize=11, color='coral')
    ax.set_ylabel('% of Events Captured', fontsize=11)
    ax.set_title('Event Capture Rate by Threshold', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelcolor='steelblue')
    ax2.tick_params(axis='x', labelcolor='coral')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    ax2.legend(loc='upper left')
    
    # 图5: 推荐4级标准的对比
    ax = axes[1, 1]
    recommended = [d for d in severity_definitions if d['category'] == 'recommended']
    names = [d['description'].split('-')[1].strip() for d in recommended]
    counts = [d['n_severe'] for d in recommended]
    pcts = [d['pct_severe'] for d in recommended]
    colors_bar = ['green', 'gold', 'orange', 'red']
    
    bars = ax.barh(names, counts, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Events', fontsize=11)
    ax.set_title('Recommended 4-Tier System Comparison', fontsize=12, fontweight='bold')
    
    # 添加百分比标签
    for bar, pct in zip(bars, pcts):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f'{pct:.2f}%', ha='left', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    # 图6: 阈值选择决策树
    ax = axes[1, 2]
    ax.axis('off')
    
    decision_text = """
阈值选择决策指南
━━━━━━━━━━━━━━━━━━━━━

问：我应该用哪个标准？

1️⃣ 早期预警系统
   → Tier 1 (2h & 0.5%)
   ✓ 捕获最多事件
   ✓ 适合预防性监控
   
2️⃣ 日常运维监控  
   → Tier 2 (4h & 1%)
   ✓ 平衡敏感性和特异性
   ✓ 推荐作为默认标准
   
3️⃣ 严重事件响应
   → Tier 3 (12h & 2%)
   ✓ 聚焦真正严重事件
   ✓ 资源分配优先级
   
4️⃣ 灾难级别分析
   → Tier 4 (24h & 5%)
   ✓ 仅最严重事件
   ✓ 政策研究、灾后评估

💡 建议：多层次使用
   - 监控用 Tier 2
   - 分析用 Tier 3
   - 研究用所有层级
    """
    
    ax.text(0.1, 0.95, decision_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_file = output_path / 'threshold_selection_guide.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ {plot_file}")
    plt.close()


def create_county_analysis_plots(county_results, recommended_standards, output_path):
    """创建县级分析可视化（聚焦推荐标准）"""
    print("\n" + "="*80)
    print("生成县级分析可视化")
    print("="*80)
    
    # 为每个推荐标准创建一个分析图
    for dur, pct, name, desc, usage in recommended_standards:
        rate_col = f'{name}_rate'
        
        if rate_col not in county_results.columns:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{desc} - County-Level Analysis\n'
                    f'Threshold: ≥{dur}h AND ≥{pct*100}% customers affected',
                    fontsize=14, fontweight='bold')
        
        # 子图1: 严重事件率分布
        ax = axes[0, 0]
        ax.hist(county_results[rate_col], bins=50, edgecolor='black', 
               alpha=0.7, color='steelblue')
        mean_rate = county_results[rate_col].mean()
        median_rate = county_results[rate_col].median()
        ax.axvline(mean_rate, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_rate:.3f}')
        ax.axvline(median_rate, color='blue', linestyle='--', linewidth=2,
                  label=f'Median: {median_rate:.3f}')
        ax.set_xlabel('Severe Event Rate', fontsize=11)
        ax.set_ylabel('Number of Counties', fontsize=11)
        ax.set_title('Distribution of Severity Rates', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # 子图2: 事件数 vs 严重率
        ax = axes[0, 1]
        scatter = ax.scatter(county_results['n_events'], 
                           county_results[rate_col],
                           alpha=0.5, s=50, 
                           c=county_results[rate_col],
                           cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xlabel('Total Events (log scale)', fontsize=11)
        ax.set_ylabel('Severe Event Rate', fontsize=11)
        ax.set_title('Event Count vs Severity Rate', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Rate')
        
        # 子图3: 前20高风险县
        ax = axes[1, 0]
        top20 = county_results.nlargest(20, rate_col)
        labels = [f"{row['county']}, {row['state']}" for _, row in top20.iterrows()]
        colors_risk = plt.cm.RdYlGn_r(top20[rate_col])
        ax.barh(range(len(top20)), top20[rate_col], color=colors_risk, alpha=0.8)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Severe Event Rate', fontsize=11)
        ax.set_title('Top 20 Highest Risk Counties', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 子图4: 按州的平均严重率
        ax = axes[1, 1]
        state_avg = county_results.groupby('state')[rate_col].mean().sort_values(ascending=False).head(20)
        state_avg.plot(kind='barh', ax=ax, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Average Severe Event Rate', fontsize=11)
        ax.set_title('Top 20 States by Average Severity', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_path / f'county_analysis_{name}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ {plot_file}")
        plt.close()


def create_comparison_dashboard(county_results, recommended_standards, output_path):
    """创建4级标准对比仪表板"""
    print("\n" + "="*80)
    print("生成4级标准对比仪表板")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 顶部: 4个标准的并排对比
    for i, (dur, pct, name, desc, usage) in enumerate(recommended_standards):
        rate_col = f'{name}_rate'
        if rate_col not in county_results.columns:
            continue
        
        ax = fig.add_subplot(gs[0, i])
        
        # 绘制分布
        color_map = ['green', 'gold', 'orange', 'red'][i]
        ax.hist(county_results[rate_col], bins=30, 
               edgecolor='black', alpha=0.7, color=color_map)
        
        mean_val = county_results[rate_col].mean()
        ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2)
        
        ax.set_title(f'{desc}\n≥{dur}h & ≥{pct*100}%', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Rate', fontsize=10)
        ax.set_ylabel('Counties', fontsize=10)
        ax.text(0.95, 0.95, f'Mean: {mean_val:.3f}', 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(alpha=0.3, axis='y')
    
    # 中部左: 县数排名变化
    ax = fig.add_subplot(gs[1, :2])
    
    # 找出在至少一个标准中排名前20的县
    all_top_counties = set()
    for dur, pct, name, desc, usage in recommended_standards:
        rate_col = f'{name}_rate'
        if rate_col in county_results.columns:
            top = county_results.nlargest(20, rate_col)
            all_top_counties.update(top['fips'].values)
    
    # 计算这些县在各标准下的排名
    if all_top_counties:
        county_subset = county_results[county_results['fips'].isin(all_top_counties)].copy()
        
        for dur, pct, name, desc, usage in recommended_standards:
            rate_col = f'{name}_rate'
            if rate_col in county_subset.columns:
                county_subset[f'{name}_rank'] = county_subset[rate_col].rank(ascending=False, method='first')
        
        # 绘制排名变化（选择前10个县）
        top10_fips = county_subset.nlargest(10, f'{recommended_standards[0][2]}_rate')['fips'].values
        
        for fips in top10_fips:
            row = county_subset[county_subset['fips'] == fips].iloc[0]
            label = f"{row['county']}, {row['state']}"
            ranks = []
            for dur, pct, name, desc, usage in recommended_standards:
                rank_col = f'{name}_rank'
                if rank_col in county_subset.columns:
                    ranks.append(row[rank_col])
            
            ax.plot(range(len(ranks)), ranks, 'o-', label=label, linewidth=2, markersize=8)
        
        ax.set_xticks(range(len(recommended_standards)))
        ax.set_xticklabels([desc.split('-')[1].strip() for dur, pct, name, desc, usage in recommended_standards], 
                          rotation=0)
        ax.set_ylabel('Rank (lower is worse)', fontsize=11)
        ax.set_title('County Ranking Changes Across Severity Tiers', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
    
    # 中部右: 各标准的事件捕获统计
    ax = fig.add_subplot(gs[1, 2:])
    
    stats_data = []
    for dur, pct, name, desc, usage in recommended_standards:
        if name in county_results.columns:
            total_severe = county_results[name].sum()
            total_events = county_results['n_events'].sum()
            capture_rate = (total_severe / total_events) * 100
            
            stats_data.append({
                'Tier': desc.split('-')[1].strip(),
                'Severe Events': total_severe,
                'Capture Rate (%)': capture_rate
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        x = np.arange(len(stats_df))
        width = 0.35
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, stats_df['Severe Events'], width, 
                      label='Severe Events', color='steelblue', alpha=0.7)
        bars2 = ax2.bar(x + width/2, stats_df['Capture Rate (%)'], width,
                       label='Capture Rate (%)', color='coral', alpha=0.7)
        
        ax.set_xlabel('Severity Tier', fontsize=11)
        ax.set_ylabel('Number of Severe Events', fontsize=11, color='steelblue')
        ax2.set_ylabel('Capture Rate (%)', fontsize=11, color='coral')
        ax.set_title('Event Capture Statistics by Tier', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df['Tier'])
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(alpha=0.3, axis='y')
    
    # 底部: 州级汇总热力图
    ax = fig.add_subplot(gs[2, :])
    
    # 计算各州在不同标准下的平均严重率
    state_data = []
    for state in county_results['state'].unique():
        state_counties = county_results[county_results['state'] == state]
        row = {'State': state}
        for dur, pct, name, desc, usage in recommended_standards:
            rate_col = f'{name}_rate'
            if rate_col in state_counties.columns:
                row[desc.split('-')[1].strip()] = state_counties[rate_col].mean()
        state_data.append(row)
    
    if state_data:
        state_df = pd.DataFrame(state_data).set_index('State')
        
        # 只显示前25个州（按Tier 2排序）
        tier2_col = recommended_standards[1][3].split('-')[1].strip()
        if tier2_col in state_df.columns:
            state_df = state_df.nlargest(25, tier2_col)
            
            sns.heatmap(state_df, annot=True, fmt='.3f', cmap='YlOrRd',
                       ax=ax, cbar_kws={'label': 'Average Severe Event Rate'})
            ax.set_title('State-Level Severity Comparison (Top 25 States by Tier 2)',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Severity Tier', fontsize=11)
            ax.set_ylabel('State', fontsize=11)
    
    plt.suptitle('4-Tier Severity System: Comprehensive Comparison Dashboard',
                fontsize=16, fontweight='bold', y=0.995)
    
    plot_file = output_path / 'tier_comparison_dashboard.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ {plot_file}")
    plt.close()


def save_results(df, severity_definitions, county_results, recommended_standards, output_dir):
    """保存结果（聚焦推荐标准）"""
    print("\n" + "="*80)
    print("保存结果")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 1. 所有严重程度定义
    severity_df = pd.DataFrame(severity_definitions)
    file1 = output_path / 'all_severity_definitions.csv'
    severity_df.to_csv(file1, index=False)
    print(f"  ✓ {file1}")
    
    # 2. 推荐标准汇总
    recommended_df = severity_df[severity_df['category'] == 'recommended'].copy()
    file2 = output_path / 'RECOMMENDED_standards.csv'
    recommended_df.to_csv(file2, index=False)
    print(f"  ✓ {file2} ⭐")
    
    # 3. 县级完整结果
    file3 = output_path / 'county_severity_full.csv'
    county_results.to_csv(file3, index=False)
    print(f"  ✓ {file3}")
    
    # 4. 县级简化结果（仅推荐标准）
    core_cols = ['fips', 'state', 'county', 'n_events', 'population',
                 'duration_hr_mean', 'customers_pct_mean']
    for dur, pct, name, desc, usage in recommended_standards:
        if name in county_results.columns:
            core_cols.append(name)
            core_cols.append(f'{name}_rate')
    
    county_simple = county_results[core_cols].copy()
    file4 = output_path / 'county_severity_SIMPLE.csv'
    county_simple.to_csv(file4, index=False)
    print(f"  ✓ {file4} ⭐")
    
    # 5. 州级汇总
    severity_cols_in_county = [col for col in county_results.columns 
                               if any(col == d['name'] for d in severity_definitions)]
    
    state_cols = ['state', 'n_events', 'population'] + severity_cols_in_county
    state_agg = county_results[state_cols].groupby('state').sum().reset_index()
    
    for col in severity_cols_in_county:
        if col in state_agg.columns:
            state_agg[f'{col}_rate'] = state_agg[col] / state_agg['n_events']
    
    state_agg = state_agg.sort_values('n_events', ascending=False)
    file5 = output_path / 'state_severity_summary.csv'
    state_agg.to_csv(file5, index=False)
    print(f"  ✓ {file5}")
    
    # 6. 文本报告
    file6 = output_path / 'SEVERITY_ANALYSIS_REPORT.txt'
    with open(file6, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("县级停电严重程度分析 - 最终报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("📊 数据概览\n")
        f.write("-"*80 + "\n")
        f.write(f"总事件数: {len(df):,}\n")
        f.write(f"分析县数: {len(county_results):,}\n")
        f.write(f"分析州数: {county_results['state'].nunique()}\n\n")
        
        f.write("⭐ 推荐的4级严重程度体系\n")
        f.write("="*80 + "\n\n")
        
        for dur, pct, name, desc, usage in recommended_standards:
            matches = [d for d in severity_definitions if d['name'] == name]
            if matches:
                d = matches[0]
                f.write(f"{desc}\n")
                f.write(f"{'-'*60}\n")
                f.write(f"定义: 停电时长 ≥ {dur}小时 AND 受影响客户 ≥ {pct*100}%\n")
                f.write(f"使用场景: {usage}\n")
                f.write(f"识别事件: {d['n_severe']:,} ({d['pct_severe']:.2f}%)\n")
                
                # 添加县级统计
                rate_col = f'{name}_rate'
                if rate_col in county_results.columns:
                    mean_rate = county_results[rate_col].mean()
                    median_rate = county_results[rate_col].median()
                    f.write(f"县级平均严重率: {mean_rate:.3f}\n")
                    f.write(f"县级中位严重率: {median_rate:.3f}\n")
                    
                    # 高风险县
                    high_risk = county_results[county_results[rate_col] > 0.3]
                    f.write(f"高风险县 (严重率>30%): {len(high_risk)}\n")
                
                f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("💡 使用建议\n")
        f.write("="*80 + "\n\n")
        f.write("1. 默认监控标准: Tier 2 (4h & 1%)\n")
        f.write("   - 适合日常运维和监控仪表板\n")
        f.write("   - 平衡了敏感性和特异性\n\n")
        f.write("2. 资源分配决策: Tier 3 (12h & 2%)\n")
        f.write("   - 聚焦真正需要紧急响应的事件\n")
        f.write("   - 用于优先级排序\n\n")
        f.write("3. 政策研究分析: 使用所有4个层级\n")
        f.write("   - 了解不同严重程度的分布\n")
        f.write("   - 评估改进措施的效果\n\n")
        f.write("4. 早期预警系统: Tier 1 (2h & 0.5%)\n")
        f.write("   - 捕获更多潜在严重事件\n")
        f.write("   - 预防性措施和预警\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("📁 输出文件说明\n")
        f.write("="*80 + "\n\n")
        f.write("重点查看:\n")
        f.write("  ⭐ RECOMMENDED_standards.csv - 推荐的4个标准\n")
        f.write("  ⭐ county_severity_SIMPLE.csv - 县级结果(简化版)\n")
        f.write("  ⭐ tier_comparison_dashboard.png - 4级对比仪表板\n\n")
        f.write("详细数据:\n")
        f.write("  • all_severity_definitions.csv - 所有标准定义\n")
        f.write("  • county_severity_full.csv - 县级完整结果\n")
        f.write("  • state_severity_summary.csv - 州级汇总\n\n")
        f.write("可视化:\n")
        f.write("  • threshold_selection_guide.png - 阈值选择指南\n")
        f.write("  • county_analysis_[tier].png - 各层级县级分析(4个文件)\n")
    
    print(f"  ✓ {file6} ⭐")
    
    return output_path


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = 'full_pop_2022.csv'
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = './severity_results'
    
    print(f"\n使用数据文件: {data_file}")
    print(f"输出目录: {output_dir}\n")
    
    try:
        # 步骤1: 加载数据
        df = load_and_prepare_data(data_file)
        
        # 步骤2: 定义核心严重程度标准（包含2小时阈值）
        df, severity_definitions, recommended_standards = define_core_severity_levels(df)
        
        # 步骤3: 聚合到县级
        county_results = aggregate_to_county(df, severity_definitions, min_events=5)
        
        # 步骤4: 保存结果
        output_path = save_results(df, severity_definitions, county_results, 
                                   recommended_standards, output_dir)
        
        # 步骤5: 可视化
        create_threshold_selection_guide(df, severity_definitions, output_path)
        create_county_analysis_plots(county_results, recommended_standards, output_path)
        create_comparison_dashboard(county_results, recommended_standards, output_path)
        
        print("\n" + "="*80)
        print("✅ 分析完成！")
        print("="*80)
        print(f"\n📁 输出文件 (在 {output_dir}/ 目录):\n")
        print("重点查看 ⭐:")
        print("  1. RECOMMENDED_standards.csv - 推荐的4级标准定义")
        print("  2. county_severity_SIMPLE.csv - 县级结果(仅包含推荐标准)")
        print("  3. SEVERITY_ANALYSIS_REPORT.txt - 分析报告和使用建议")
        print("  4. tier_comparison_dashboard.png - 4级对比仪表板")
        print("  5. threshold_selection_guide.png - 阈值选择可视化指南")
        print("\n详细数据:")
        print("  6. all_severity_definitions.csv - 所有标准(探索性+推荐)")
        print("  7. county_severity_full.csv - 县级完整结果")
        print("  8. state_severity_summary.csv - 州级汇总")
        print("  9. county_analysis_[tier1-4].png - 4个县级详细分析图")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
