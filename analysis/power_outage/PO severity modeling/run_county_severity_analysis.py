#!/usr/bin/env python3
"""
县级停电严重程度分析 - 直接运行版本
数据文件: full_pop_2022.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data(data_path='full_pop_2022.csv'):
    """加载并准备数据"""
    print("="*80)
    print("县级停电严重程度分析")
    print("="*80)
    
    print(f"\n[1/4] 加载数据: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  总记录数: {len(df):,}")
    print(f"  字段: {list(df.columns)}")
    
    # 转换时间
    df['start_time'] = pd.to_datetime(df['start_time'])
    if 'end_time' in df.columns:
        df['end_time'] = pd.to_datetime(df['end_time'])
    
    # 计算小时数
    df['duration_hr'] = df['duration_min'] / 60
    
    # 确保fips是5位字符串
    df['fips'] = df['fips'].astype(str).str.zfill(5)
    
    print(f"\n[2/4] 数据清理")
    print(f"  县数量: {df['fips'].nunique()}")
    print(f"  州数量: {df['state'].nunique()}")
    
    # 删除缺失值
    before = len(df)
    df = df.dropna(subset=['mean_customers', 'duration_min', 'POPESTIMATE2022'])
    after = len(df)
    if before > after:
        print(f"  删除缺失值: {before - after} 条记录")
    
    # 删除异常值（duration_hr > 0 且 mean_customers > 0）
    df = df[(df['duration_hr'] > 0) & (df['mean_customers'] > 0)]
    print(f"  删除异常值后: {len(df):,} 条记录")
    
    # 计算受影响客户百分比
    df['customers_pct'] = (df['mean_customers'] / df['POPESTIMATE2022']).clip(0, 1)
    
    print(f"\n[3/4] 数据统计")
    print(f"  停电时长 (小时):")
    print(f"    最小值: {df['duration_hr'].min():.2f}")
    print(f"    中位数: {df['duration_hr'].median():.2f}")
    print(f"    平均值: {df['duration_hr'].mean():.2f}")
    print(f"    最大值: {df['duration_hr'].max():.2f}")
    print(f"    75分位: {df['duration_hr'].quantile(0.75):.2f}")
    print(f"    90分位: {df['duration_hr'].quantile(0.90):.2f}")
    print(f"    95分位: {df['duration_hr'].quantile(0.95):.2f}")
    
    print(f"\n  受影响客户数:")
    print(f"    最小值: {df['mean_customers'].min():.0f}")
    print(f"    中位数: {df['mean_customers'].median():.0f}")
    print(f"    平均值: {df['mean_customers'].mean():.0f}")
    print(f"    最大值: {df['mean_customers'].max():.0f}")
    print(f"    75分位: {df['mean_customers'].quantile(0.75):.0f}")
    print(f"    90分位: {df['mean_customers'].quantile(0.90):.0f}")
    print(f"    95分位: {df['mean_customers'].quantile(0.95):.0f}")
    
    print(f"\n  受影响客户百分比 (占县人口):")
    print(f"    最小值: {df['customers_pct'].min():.4f} ({df['customers_pct'].min()*100:.2f}%)")
    print(f"    中位数: {df['customers_pct'].median():.4f} ({df['customers_pct'].median()*100:.2f}%)")
    print(f"    平均值: {df['customers_pct'].mean():.4f} ({df['customers_pct'].mean()*100:.2f}%)")
    print(f"    最大值: {df['customers_pct'].max():.4f} ({df['customers_pct'].max()*100:.2f}%)")
    print(f"    75分位: {df['customers_pct'].quantile(0.75):.4f} ({df['customers_pct'].quantile(0.75)*100:.2f}%)")
    print(f"    90分位: {df['customers_pct'].quantile(0.90):.4f} ({df['customers_pct'].quantile(0.90)*100:.2f}%)")
    print(f"    95分位: {df['customers_pct'].quantile(0.95):.4f} ({df['customers_pct'].quantile(0.95)*100:.2f}%)")
    
    return df


def define_severity_thresholds(df):
    """定义多种严重程度阈值"""
    print("\n[4/4] 定义严重程度等级")
    print("="*80)
    
    severity_definitions = []
    
    # 基于数据分布，设置更合理的阈值
    # 时长: 使用75、90、95分位数作为参考
    duration_thresholds = [4, 12, 24, 48]
    
    # 客户百分比: 0.5%, 1%, 2%, 5%
    customer_pct_thresholds = [0.005, 0.01, 0.02, 0.05]
    
    # 绝对客户数: 基于数据的75、90、95分位数
    customer_abs_thresholds = [500, 1000, 2500, 5000]
    
    # 方法1: 仅基于时长
    print("\n方法1: 仅基于停电时长")
    print("-" * 80)
    for dur in duration_thresholds:
        col_name = f'severe_dur_{int(dur)}h'
        df[col_name] = (df['duration_hr'] >= dur).astype(int)
        n_severe = df[col_name].sum()
        pct_severe = (n_severe / len(df)) * 100
        
        severity_definitions.append({
            'method': 'duration_only',
            'name': col_name,
            'duration_hr': dur,
            'customer_pct': None,
            'customer_abs': None,
            'n_severe': n_severe,
            'pct_severe': pct_severe
        })
        
        print(f"  时长 ≥ {dur:2d}小时: {n_severe:8,} 次 ({pct_severe:6.2f}%)")
    
    # 方法2: 仅基于客户百分比
    print("\n方法2: 仅基于受影响客户百分比")
    print("-" * 80)
    for pct in customer_pct_thresholds:
        col_name = f'severe_pct_{int(pct*1000)}bps'  # basis points
        df[col_name] = (df['customers_pct'] >= pct).astype(int)
        n_severe = df[col_name].sum()
        pct_severe = (n_severe / len(df)) * 100
        
        severity_definitions.append({
            'method': 'percentage_only',
            'name': col_name,
            'duration_hr': None,
            'customer_pct': pct,
            'customer_abs': None,
            'n_severe': n_severe,
            'pct_severe': pct_severe
        })
        
        print(f"  客户% ≥ {pct*100:5.2f}%: {n_severe:8,} 次 ({pct_severe:6.2f}%)")
    
    # 方法3: 仅基于绝对客户数
    print("\n方法3: 仅基于受影响客户绝对数")
    print("-" * 80)
    for abs_val in customer_abs_thresholds:
        col_name = f'severe_abs_{abs_val}'
        df[col_name] = (df['mean_customers'] >= abs_val).astype(int)
        n_severe = df[col_name].sum()
        pct_severe = (n_severe / len(df)) * 100
        
        severity_definitions.append({
            'method': 'absolute_only',
            'name': col_name,
            'duration_hr': None,
            'customer_pct': None,
            'customer_abs': abs_val,
            'n_severe': n_severe,
            'pct_severe': pct_severe
        })
        
        print(f"  客户数 ≥ {abs_val:5,}: {n_severe:8,} 次 ({pct_severe:6.2f}%)")
    
    # 方法4: 组合条件 (推荐)
    print("\n方法4: 组合条件 (时长 AND 客户百分比) - 推荐")
    print("-" * 80)
    
    # 选择关键组合
    key_combinations = [
        (4, 0.005, '宽松-早期预警'),
        (12, 0.01, '中等-日常监控'),
        (24, 0.02, '标准-重要事件'),
        (48, 0.05, '严格-重大事故'),
    ]
    
    for dur, pct, desc in key_combinations:
        col_name = f'severe_comb_d{int(dur)}_p{int(pct*100)}'
        df[col_name] = (
            (df['duration_hr'] >= dur) & 
            (df['customers_pct'] >= pct)
        ).astype(int)
        n_severe = df[col_name].sum()
        pct_severe = (n_severe / len(df)) * 100
        
        severity_definitions.append({
            'method': 'combined',
            'name': col_name,
            'duration_hr': dur,
            'customer_pct': pct,
            'customer_abs': None,
            'n_severe': n_severe,
            'pct_severe': pct_severe,
            'description': desc
        })
        
        print(f"  时长≥{dur:2d}h AND 客户%≥{pct*100:5.2f}% ({desc:12s}): {n_severe:8,} 次 ({pct_severe:6.2f}%)")
    
    # 方法5: 组合条件 (时长 AND 绝对客户数)
    print("\n方法5: 组合条件 (时长 AND 绝对客户数)")
    print("-" * 80)
    
    abs_combinations = [
        (12, 1000),
        (24, 2500),
        (48, 5000),
    ]
    
    for dur, abs_val in abs_combinations:
        col_name = f'severe_comb_d{int(dur)}_a{abs_val}'
        df[col_name] = (
            (df['duration_hr'] >= dur) & 
            (df['mean_customers'] >= abs_val)
        ).astype(int)
        n_severe = df[col_name].sum()
        pct_severe = (n_severe / len(df)) * 100
        
        severity_definitions.append({
            'method': 'combined_abs',
            'name': col_name,
            'duration_hr': dur,
            'customer_pct': None,
            'customer_abs': abs_val,
            'n_severe': n_severe,
            'pct_severe': pct_severe
        })
        
        print(f"  时长≥{dur:2d}h AND 客户≥{abs_val:5,}: {n_severe:8,} 次 ({pct_severe:6.2f}%)")
    
    print(f"\n总共定义了 {len(severity_definitions)} 种严重程度标准")
    
    return df, severity_definitions


def aggregate_to_county(df, severity_definitions, min_events=5):
    """聚合到县级别"""
    print("\n" + "="*80)
    print("聚合到县级别")
    print("="*80)
    
    # 获取所有严重程度列
    severity_cols = [d['name'] for d in severity_definitions]
    
    # 聚合
    agg_dict = {
        'state': 'first',
        'county': 'first',
        'fips': 'count',
        'duration_hr': ['mean', 'median', 'max', 'std'],
        'mean_customers': ['mean', 'median', 'max', 'std'],
        'customers_pct': ['mean', 'median', 'max', 'std'],
        'POPESTIMATE2022': 'first'
    }
    
    for col in severity_cols:
        agg_dict[col] = 'sum'
    
    county_agg = df.groupby('fips').agg(agg_dict).reset_index()
    
    # 展平列名
    county_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in county_agg.columns]
    
    # 重命名
    county_agg = county_agg.rename(columns={
        'fips_count': 'n_events',
        'state_first': 'state',
        'county_first': 'county',
        'POPESTIMATE2022_first': 'population'
    })
    
    # 计算各严重程度的发生率
    for col in severity_cols:
        if col in county_agg.columns:
            county_agg[f'{col}_rate'] = county_agg[col] / county_agg['n_events']
    
    # 过滤
    county_agg = county_agg[county_agg['n_events'] >= min_events].copy()
    
    print(f"\n县级统计 (至少{min_events}个事件):")
    print(f"  县数量: {len(county_agg)}")
    print(f"  总事件数: {county_agg['n_events'].sum():,}")
    print(f"  平均事件数/县: {county_agg['n_events'].mean():.1f}")
    print(f"  中位事件数/县: {county_agg['n_events'].median():.0f}")
    
    # 排序
    county_agg = county_agg.sort_values('n_events', ascending=False).reset_index(drop=True)
    
    return county_agg


def save_results(df, severity_definitions, county_results, output_dir='./severity_results'):
    """保存结果"""
    print("\n" + "="*80)
    print("保存结果")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 1. 严重程度定义汇总
    severity_df = pd.DataFrame(severity_definitions)
    file1 = output_path / 'severity_definitions.csv'
    severity_df.to_csv(file1, index=False)
    print(f"  ✓ {file1}")
    
    # 2. 县级详细结果
    file2 = output_path / 'county_severity_full.csv'
    county_results.to_csv(file2, index=False)
    print(f"  ✓ {file2}")
    
    # 3. 州级汇总
    state_cols = ['state', 'n_events', 'population']
    state_cols.extend([d['name'] for d in severity_definitions])
    
    state_agg = county_results[state_cols].groupby('state').sum().reset_index()
    
    # 计算州级比率
    for d in severity_definitions:
        col = d['name']
        state_agg[f'{col}_rate'] = state_agg[col] / state_agg['n_events']
    
    state_agg = state_agg.sort_values('n_events', ascending=False)
    file3 = output_path / 'state_severity_summary.csv'
    state_agg.to_csv(file3, index=False)
    print(f"  ✓ {file3}")
    
    # 4. 文本报告
    file4 = output_path / 'severity_report.txt'
    with open(file4, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("县级停电严重程度分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. 数据概览\n")
        f.write("-"*80 + "\n")
        f.write(f"总事件数: {len(df):,}\n")
        f.write(f"分析县数: {len(county_results):,}\n")
        f.write(f"分析州数: {county_results['state'].nunique()}\n\n")
        
        f.write("2. 推荐的严重程度标准\n")
        f.write("-"*80 + "\n\n")
        
        recommended = [d for d in severity_definitions if d['method'] == 'combined']
        for i, d in enumerate(recommended, 1):
            desc = d.get('description', '')
            f.write(f"标准{i}: {d['name']}\n")
            f.write(f"  描述: {desc}\n")
            f.write(f"  时长阈值: ≥{d['duration_hr']}小时\n")
            f.write(f"  客户%阈值: ≥{d['customer_pct']*100}%\n")
            f.write(f"  识别事件: {d['n_severe']:,} ({d['pct_severe']:.2f}%)\n\n")
        
        f.write("\n3. 前20个高风险县 (按事件数)\n")
        f.write("-"*80 + "\n\n")
        
        top20 = county_results.head(20)
        for i, row in top20.iterrows():
            f.write(f"{i+1}. {row['county']}, {row['state']} (FIPS: {row['fips']})\n")
            f.write(f"   事件数: {row['n_events']:,}\n")
            f.write(f"   人口: {row['population']:,}\n")
            f.write(f"   平均时长: {row['duration_hr_mean']:.1f}小时\n")
            f.write(f"   平均受影响客户: {row['mean_customers_mean']:.0f} ({row['customers_pct_mean']*100:.2f}%)\n\n")
    
    print(f"  ✓ {file4}")
    
    return output_path


def plot_results(df, severity_definitions, county_results, output_path):
    """绘制可视化图表"""
    print("\n" + "="*80)
    print("生成可视化图表")
    print("="*80)
    
    # 图1: 严重程度定义概览
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    severity_df = pd.DataFrame(severity_definitions)
    
    # 子图1: 按方法分类的严重事件数
    ax = axes[0, 0]
    method_summary = severity_df.groupby('method')['n_severe'].sum().sort_values(ascending=True)
    method_summary.plot(kind='barh', ax=ax, color='steelblue', alpha=0.7)
    ax.set_xlabel('Total Severe Events', fontsize=12)
    ax.set_title('Severe Events by Method Type', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 子图2: 各定义的覆盖率
    ax = axes[0, 1]
    top_defs = severity_df.nlargest(15, 'pct_severe')
    colors = plt.cm.RdYlGn_r(top_defs['pct_severe'] / 100)
    ax.barh(range(len(top_defs)), top_defs['pct_severe'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_defs)))
    ax.set_yticklabels(top_defs['name'].str.replace('severe_', ''), fontsize=8)
    ax.set_xlabel('Percentage of Events Classified as Severe (%)', fontsize=12)
    ax.set_title('Coverage Rate by Definition (Top 15)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 子图3: 县级事件数分布
    ax = axes[1, 0]
    ax.hist(county_results['n_events'], bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.set_xlabel('Number of Events per County', fontsize=12)
    ax.set_ylabel('Number of Counties', fontsize=12)
    ax.set_title('Distribution of Events Across Counties', fontsize=13, fontweight='bold')
    ax.axvline(county_results['n_events'].median(), color='red', linestyle='--', 
              label=f'Median: {county_results["n_events"].median():.0f}')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 子图4: 前20个县
    ax = axes[1, 1]
    top20 = county_results.head(20)
    labels = [f"{row['county']}, {row['state']}" for _, row in top20.iterrows()]
    ax.barh(range(len(top20)), top20['n_events'], color='teal', alpha=0.7)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Number of Outage Events', fontsize=12)
    ax.set_title('Top 20 Counties by Event Count', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot1 = output_path / 'severity_overview.png'
    plt.savefig(plot1, dpi=300, bbox_inches='tight')
    print(f"  ✓ {plot1}")
    plt.close()
    
    # 图2: 组合条件热力图
    combined_defs = [d for d in severity_definitions if d['method'] == 'combined']
    if len(combined_defs) >= 4:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        durations = sorted(list(set([d['duration_hr'] for d in combined_defs if d['duration_hr']])))
        pcts = sorted(list(set([d['customer_pct'] for d in combined_defs if d['customer_pct']])))
        
        if durations and pcts:
            matrix = np.zeros((len(durations), len(pcts)))
            for d in combined_defs:
                if d['duration_hr'] and d['customer_pct']:
                    i = durations.index(d['duration_hr'])
                    j = pcts.index(d['customer_pct'])
                    matrix[i, j] = d['pct_severe']
            
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                       xticklabels=[f'{p*100:.2f}%' for p in pcts],
                       yticklabels=[f'{d}h' for d in durations],
                       cbar_kws={'label': 'Severe Event Percentage (%)'})
            ax.set_xlabel('Customer Impact Threshold', fontsize=12)
            ax.set_ylabel('Duration Threshold', fontsize=12)
            ax.set_title('Severity Heatmap: Combined Conditions', fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            plot2 = output_path / 'severity_heatmap.png'
            plt.savefig(plot2, dpi=300, bbox_inches='tight')
            print(f"  ✓ {plot2}")
            plt.close()
    
    # 图3: 使用推荐标准的县级分析
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 使用中等标准: severe_comb_d12_p1
    standard_col = 'severe_comb_d12_p1'
    rate_col = f'{standard_col}_rate'
    
    if rate_col in county_results.columns:
        # 子图1: 事件数 vs 严重事件率
        ax = axes[0, 0]
        scatter = ax.scatter(county_results['n_events'], county_results[rate_col],
                           alpha=0.5, s=50, c=county_results[rate_col], 
                           cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xlabel('Total Events', fontsize=12)
        ax.set_ylabel('Severe Event Rate', fontsize=12)
        ax.set_title(f'Events vs Severity Rate\n(Standard: {standard_col})', 
                    fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Rate')
        
        # 子图2: 严重事件率分布
        ax = axes[0, 1]
        ax.hist(county_results[rate_col], bins=30, edgecolor='black', 
               alpha=0.7, color='orange')
        ax.set_xlabel('Severe Event Rate', fontsize=12)
        ax.set_ylabel('Number of Counties', fontsize=12)
        ax.set_title('Distribution of Severity Rates', fontsize=13, fontweight='bold')
        ax.axvline(county_results[rate_col].mean(), color='red', 
                  linestyle='--', label=f'Mean: {county_results[rate_col].mean():.3f}')
        ax.axvline(county_results[rate_col].median(), color='blue', 
                  linestyle='--', label=f'Median: {county_results[rate_col].median():.3f}')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # 子图3: 高风险县（rate > 0.3）
        ax = axes[1, 0]
        high_risk = county_results[county_results[rate_col] > 0.3].nlargest(15, rate_col)
        if len(high_risk) > 0:
            labels = [f"{row['county']}, {row['state']}" for _, row in high_risk.iterrows()]
            ax.barh(range(len(high_risk)), high_risk[rate_col], color='crimson', alpha=0.7)
            ax.set_yticks(range(len(high_risk)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('Severe Event Rate', fontsize=12)
            ax.set_title('Top 15 High-Risk Counties (Rate > 0.3)', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        # 子图4: 按州的平均严重程度
        ax = axes[1, 1]
        state_severity = county_results.groupby('state')[rate_col].mean().sort_values(ascending=False).head(20)
        state_severity.plot(kind='barh', ax=ax, color='purple', alpha=0.7)
        ax.set_xlabel('Average Severe Event Rate', fontsize=12)
        ax.set_title('Top 20 States by Average Severity Rate', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot3 = output_path / 'county_severity_analysis.png'
    plt.savefig(plot3, dpi=300, bbox_inches='tight')
    print(f"  ✓ {plot3}")
    plt.close()


def main():
    """主函数"""
    import sys
    
    # 获取数据文件路径
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = 'full_pop_2022.csv'
    
    # 输出目录
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = './severity_results'
    
    print(f"\n使用数据文件: {data_file}")
    print(f"输出目录: {output_dir}\n")
    
    try:
        # 步骤1: 加载数据
        df = load_and_prepare_data(data_file)
        
        # 步骤2: 定义严重程度
        df, severity_definitions = define_severity_thresholds(df)
        
        # 步骤3: 聚合到县级
        county_results = aggregate_to_county(df, severity_definitions, min_events=5)
        
        # 步骤4: 保存结果
        output_path = save_results(df, severity_definitions, county_results, output_dir)
        
        # 步骤5: 绘图
        plot_results(df, severity_definitions, county_results, output_path)
        
        print("\n" + "="*80)
        print("✓ 分析完成！")
        print("="*80)
        print(f"\n输出文件保存在: {output_dir}/")
        print("  1. severity_definitions.csv - 严重程度定义汇总")
        print("  2. county_severity_full.csv - 县级详细结果")
        print("  3. state_severity_summary.csv - 州级汇总")
        print("  4. severity_report.txt - 文本分析报告")
        print("  5. severity_overview.png - 概览图表")
        print("  6. severity_heatmap.png - 热力图")
        print("  7. county_severity_analysis.png - 县级分析图")
        
    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到文件 '{data_file}'")
        print("请确保数据文件在当前目录，或指定正确的路径")
        print(f"\n使用方法: python {sys.argv[0]} <数据文件路径> [输出目录]")
        print(f"示例: python {sys.argv[0]} full_pop_2022.csv ./results")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
