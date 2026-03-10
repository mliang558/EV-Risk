#!/usr/bin/env python3
"""
Step 2B: County-Level Severity Modeling
========================================

计算县级别的停电严重程度，使用多个阈值定义

字段说明:
- fips: 县代码
- state: 州名
- county: 县名
- start_time, end_time: 停电时间
- min_customers, max_customers, mean_customers: 受影响客户数
- duration, duration_min: 停电时长
- POPESTIMATE2022: 县人口
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CountySeverityAnalyzer:
    """县级严重程度分析器"""
    
    def __init__(self, data_path):
        """
        初始化
        
        Parameters:
        -----------
        data_path : str
            包含停电数据的CSV文件路径
            必需字段: fips, state, county, start_time, end_time, 
                     mean_customers, duration_min, POPESTIMATE2022
        """
        self.data_path = data_path
        self.df = None
        self.county_results = None
        
    def load_data(self):
        """加载数据"""
        print("="*80)
        print("县级停电严重程度分析")
        print("="*80)
        
        print("\n[1/3] 加载数据...")
        self.df = pd.read_csv(self.data_path)
        print(f"  总记录数: {len(self.df):,}")
        
        # 转换时间
        if 'start_time' in self.df.columns:
            self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        
        # 计算小时数
        if 'duration_min' in self.df.columns:
            self.df['duration_hr'] = self.df['duration_min'] / 60
        elif 'duration' in self.df.columns:
            # 假设duration是timedelta
            self.df['duration_hr'] = self.df['duration'].dt.total_seconds() / 3600
        
        # 确保fips是5位字符串
        self.df['fips'] = self.df['fips'].astype(str).str.zfill(5)
        
        print(f"  县数量: {self.df['fips'].nunique()}")
        print(f"  州数量: {self.df['state'].nunique()}")
        
        # 检查必需字段
        required_fields = ['fips', 'state', 'county', 'mean_customers', 
                          'duration_hr', 'POPESTIMATE2022']
        missing = [f for f in required_fields if f not in self.df.columns]
        if missing:
            raise ValueError(f"缺少必需字段: {missing}")
        
        # 删除缺失值
        before = len(self.df)
        self.df = self.df.dropna(subset=['mean_customers', 'duration_hr', 'POPESTIMATE2022'])
        after = len(self.df)
        if before > after:
            print(f"  删除缺失值: {before - after} 条记录")
        
        # 计算受影响客户百分比
        self.df['customers_pct'] = (
            self.df['mean_customers'] / self.df['POPESTIMATE2022']
        ).clip(0, 1)  # 限制在0-100%
        
        print(f"\n[2/3] 数据统计:")
        print(f"  停电时长 (小时):")
        print(f"    最小: {self.df['duration_hr'].min():.2f}")
        print(f"    中位数: {self.df['duration_hr'].median():.2f}")
        print(f"    平均: {self.df['duration_hr'].mean():.2f}")
        print(f"    最大: {self.df['duration_hr'].max():.2f}")
        print(f"\n  受影响客户数:")
        print(f"    最小: {self.df['mean_customers'].min():.0f}")
        print(f"    中位数: {self.df['mean_customers'].median():.0f}")
        print(f"    平均: {self.df['mean_customers'].mean():.0f}")
        print(f"    最大: {self.df['mean_customers'].max():.0f}")
        print(f"\n  受影响客户百分比:")
        print(f"    最小: {self.df['customers_pct'].min():.4f} ({self.df['customers_pct'].min()*100:.2f}%)")
        print(f"    中位数: {self.df['customers_pct'].median():.4f} ({self.df['customers_pct'].median()*100:.2f}%)")
        print(f"    平均: {self.df['customers_pct'].mean():.4f} ({self.df['customers_pct'].mean()*100:.2f}%)")
        print(f"    最大: {self.df['customers_pct'].max():.4f} ({self.df['customers_pct'].max()*100:.2f}%)")
        
        return self
    
    def define_severity_levels(self,
                               duration_thresholds=[4, 12, 24, 48],
                               customer_pct_thresholds=[0.01, 0.02, 0.05, 0.10],
                               customer_abs_thresholds=[500, 1000, 5000, 10000]):
        """
        定义多个严重程度阈值
        
        Parameters:
        -----------
        duration_thresholds : list
            停电时长阈值（小时）
            默认: [4, 12, 24, 48] - 4小时、12小时、24小时、48小时
        
        customer_pct_thresholds : list
            受影响客户百分比阈值
            默认: [0.01, 0.02, 0.05, 0.10] - 1%, 2%, 5%, 10%
        
        customer_abs_thresholds : list
            受影响客户绝对数阈值
            默认: [500, 1000, 5000, 10000]
        """
        print("\n[3/3] 定义严重程度等级...")
        print("="*80)
        
        self.severity_definitions = []
        
        # 方法1: 仅基于时长
        print("\n方法1: 仅基于停电时长")
        print("-" * 80)
        for dur in duration_thresholds:
            col_name = f'severe_dur_{int(dur)}h'
            self.df[col_name] = (self.df['duration_hr'] >= dur).astype(int)
            n_severe = self.df[col_name].sum()
            pct_severe = (n_severe / len(self.df)) * 100
            
            self.severity_definitions.append({
                'method': 'duration_only',
                'column': col_name,
                'duration_threshold': dur,
                'customer_pct_threshold': None,
                'customer_abs_threshold': None,
                'n_severe': n_severe,
                'pct_severe': pct_severe
            })
            
            print(f"  时长 ≥ {dur:2d}小时: {n_severe:6,} 次 ({pct_severe:5.2f}%)")
        
        # 方法2: 仅基于受影响客户百分比
        print("\n方法2: 仅基于受影响客户百分比")
        print("-" * 80)
        for pct in customer_pct_thresholds:
            col_name = f'severe_pct_{int(pct*100)}pct'
            self.df[col_name] = (self.df['customers_pct'] >= pct).astype(int)
            n_severe = self.df[col_name].sum()
            pct_severe = (n_severe / len(self.df)) * 100
            
            self.severity_definitions.append({
                'method': 'percentage_only',
                'column': col_name,
                'duration_threshold': None,
                'customer_pct_threshold': pct,
                'customer_abs_threshold': None,
                'n_severe': n_severe,
                'pct_severe': pct_severe
            })
            
            print(f"  客户% ≥ {pct*100:4.1f}%: {n_severe:6,} 次 ({pct_severe:5.2f}%)")
        
        # 方法3: 仅基于受影响客户绝对数
        print("\n方法3: 仅基于受影响客户绝对数")
        print("-" * 80)
        for abs_val in customer_abs_thresholds:
            col_name = f'severe_abs_{abs_val}'
            self.df[col_name] = (self.df['mean_customers'] >= abs_val).astype(int)
            n_severe = self.df[col_name].sum()
            pct_severe = (n_severe / len(self.df)) * 100
            
            self.severity_definitions.append({
                'method': 'absolute_only',
                'column': col_name,
                'duration_threshold': None,
                'customer_pct_threshold': None,
                'customer_abs_threshold': abs_val,
                'n_severe': n_severe,
                'pct_severe': pct_severe
            })
            
            print(f"  客户数 ≥ {abs_val:6,}: {n_severe:6,} 次 ({pct_severe:5.2f}%)")
        
        # 方法4: 组合条件 (时长 AND 百分比)
        print("\n方法4: 组合条件 (时长 AND 客户百分比)")
        print("-" * 80)
        # 只选择几个代表性组合，避免太多
        key_combinations = [
            (4, 0.01),   # 4小时 且 1%
            (12, 0.02),  # 12小时 且 2%
            (24, 0.05),  # 24小时 且 5%
            (48, 0.10),  # 48小时 且 10%
        ]
        
        for dur, pct in key_combinations:
            col_name = f'severe_combined_d{int(dur)}_p{int(pct*100)}'
            self.df[col_name] = (
                (self.df['duration_hr'] >= dur) & 
                (self.df['customers_pct'] >= pct)
            ).astype(int)
            n_severe = self.df[col_name].sum()
            pct_severe = (n_severe / len(self.df)) * 100
            
            self.severity_definitions.append({
                'method': 'combined',
                'column': col_name,
                'duration_threshold': dur,
                'customer_pct_threshold': pct,
                'customer_abs_threshold': None,
                'n_severe': n_severe,
                'pct_severe': pct_severe
            })
            
            print(f"  时长≥{dur:2d}h AND 客户%≥{pct*100:4.1f}%: {n_severe:6,} 次 ({pct_severe:5.2f}%)")
        
        # 保存定义
        self.severity_def_df = pd.DataFrame(self.severity_definitions)
        
        print(f"\n总共定义了 {len(self.severity_definitions)} 种严重程度标准")
        
        return self
    
    def aggregate_to_county(self):
        """聚合到县级别"""
        print("\n" + "="*80)
        print("聚合到县级别")
        print("="*80)
        
        # 获取所有严重程度列
        severity_cols = [d['column'] for d in self.severity_definitions]
        
        # 聚合字典
        agg_dict = {
            'state': 'first',
            'county': 'first',
            'fips': 'count',  # 事件数
            'duration_hr': ['mean', 'median', 'max'],
            'mean_customers': ['mean', 'median', 'max'],
            'customers_pct': ['mean', 'median', 'max'],
            'POPESTIMATE2022': 'first'
        }
        
        # 添加严重程度列（求和）
        for col in severity_cols:
            agg_dict[col] = 'sum'
        
        # 聚合
        county_agg = self.df.groupby('fips').agg(agg_dict).reset_index()
        
        # 展平列名
        county_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in county_agg.columns]
        
        # 重命名
        rename_dict = {
            'fips_count': 'n_events',
            'state_first': 'state',
            'county_first': 'county',
            'POPESTIMATE2022_first': 'population'
        }
        county_agg = county_agg.rename(columns=rename_dict)
        
        # 计算每种定义下的严重事件比例
        for d in self.severity_definitions:
            col = d['column']
            if col in county_agg.columns:
                pct_col = f'{col}_rate'
                county_agg[pct_col] = county_agg[col] / county_agg['n_events']
        
        # 过滤：至少5个事件
        county_agg = county_agg[county_agg['n_events'] >= 5].copy()
        
        print(f"\n县级统计 (至少5个事件):")
        print(f"  县数量: {len(county_agg)}")
        print(f"  总事件数: {county_agg['n_events'].sum():,}")
        print(f"  平均事件数/县: {county_agg['n_events'].mean():.1f}")
        
        # 排序
        county_agg = county_agg.sort_values('n_events', ascending=False).reset_index(drop=True)
        
        self.county_results = county_agg
        
        return self
    
    def create_summary_report(self, output_dir='./severity_results'):
        """创建汇总报告"""
        print("\n" + "="*80)
        print("生成汇总报告")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 报告1: 严重程度定义汇总
        summary_file = output_path / 'severity_definitions_summary.csv'
        self.severity_def_df.to_csv(summary_file, index=False)
        print(f"  ✓ 严重程度定义: {summary_file}")
        
        # 报告2: 县级汇总
        county_file = output_path / 'county_severity_summary.csv'
        self.county_results.to_csv(county_file, index=False)
        print(f"  ✓ 县级汇总: {county_file}")
        
        # 报告3: 州级汇总
        state_summary = self.county_results.groupby('state').agg({
            'n_events': 'sum',
            'population': 'sum'
        }).reset_index()
        
        # 添加各种严重程度的统计
        for d in self.severity_definitions:
            col = d['column']
            if col in self.county_results.columns:
                state_summary[col] = self.county_results.groupby('state')[col].sum().values
                state_summary[f'{col}_rate'] = state_summary[col] / state_summary['n_events']
        
        state_file = output_path / 'state_severity_summary.csv'
        state_summary.to_csv(state_file, index=False)
        print(f"  ✓ 州级汇总: {state_file}")
        
        # 报告4: 文本摘要
        text_file = output_path / 'severity_analysis_summary.txt'
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("县级停电严重程度分析报告\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. 数据概览\n")
            f.write("-" * 80 + "\n")
            f.write(f"总事件数: {len(self.df):,}\n")
            f.write(f"县数量: {self.df['fips'].nunique()}\n")
            f.write(f"州数量: {self.df['state'].nunique()}\n")
            f.write(f"分析的县数量 (≥5事件): {len(self.county_results)}\n\n")
            
            f.write("2. 严重程度定义\n")
            f.write("-" * 80 + "\n")
            for i, d in enumerate(self.severity_definitions, 1):
                f.write(f"\n定义 {i}: {d['column']}\n")
                f.write(f"  方法: {d['method']}\n")
                if d['duration_threshold']:
                    f.write(f"  时长阈值: {d['duration_threshold']} 小时\n")
                if d['customer_pct_threshold']:
                    f.write(f"  客户%阈值: {d['customer_pct_threshold']*100}%\n")
                if d['customer_abs_threshold']:
                    f.write(f"  客户数阈值: {d['customer_abs_threshold']:,}\n")
                f.write(f"  严重事件数: {d['n_severe']:,} ({d['pct_severe']:.2f}%)\n")
            
            f.write("\n\n3. 建议使用的严重程度标准\n")
            f.write("-" * 80 + "\n")
            f.write("基于分析，建议以下标准作为不同场景的严重程度定义：\n\n")
            f.write("场景1 - 保守标准（高严重性）:\n")
            f.write("  severe_combined_d48_p10: 时长≥48小时 且 客户%≥10%\n\n")
            f.write("场景2 - 中等标准:\n")
            f.write("  severe_combined_d24_p5: 时长≥24小时 且 客户%≥5%\n\n")
            f.write("场景3 - 宽松标准（早期预警）:\n")
            f.write("  severe_combined_d4_p1: 时长≥4小时 且 客户%≥1%\n\n")
            
            f.write("\n4. 前10个高风险县 (按事件数)\n")
            f.write("-" * 80 + "\n")
            top10 = self.county_results.head(10)
            for i, row in top10.iterrows():
                f.write(f"\n{i+1}. {row['county']}, {row['state']} (FIPS: {row['fips']})\n")
                f.write(f"   事件数: {row['n_events']}\n")
                f.write(f"   人口: {row['population']:,}\n")
                f.write(f"   平均时长: {row['duration_hr_mean']:.1f} 小时\n")
                f.write(f"   平均受影响客户: {row['mean_customers_mean']:.0f}\n")
                f.write(f"   平均受影响%: {row['customers_pct_mean']*100:.2f}%\n")
        
        print(f"  ✓ 文本摘要: {text_file}")
        
        return self
    
    def plot_results(self, output_dir='./severity_results'):
        """绘制可视化结果"""
        print("\n" + "="*80)
        print("生成可视化图表")
        print("="*80)
        
        output_path = Path(output_dir)
        
        # 图1: 不同严重程度定义的比较
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 按方法分组
        methods = self.severity_def_df['method'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        method_colors = dict(zip(methods, colors))
        
        # 子图1: 严重事件数量
        ax = axes[0, 0]
        for method in methods:
            subset = self.severity_def_df[self.severity_def_df['method'] == method]
            ax.barh(range(len(subset)), subset['n_severe'], 
                   label=method, alpha=0.7, color=method_colors[method])
        ax.set_xlabel('严重事件数量', fontsize=12)
        ax.set_title('不同定义下的严重事件数量', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # 子图2: 严重事件百分比
        ax = axes[0, 1]
        labels = self.severity_def_df['column'].str.replace('severe_', '')
        colors_list = [method_colors[m] for m in self.severity_def_df['method']]
        ax.barh(range(len(self.severity_def_df)), self.severity_def_df['pct_severe'],
               color=colors_list, alpha=0.7)
        ax.set_yticks(range(len(self.severity_def_df)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('严重事件百分比 (%)', fontsize=12)
        ax.set_title('各定义覆盖的事件比例', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 子图3: 县数 vs 平均严重事件率（选一个定义）
        ax = axes[1, 0]
        # 使用中等标准
        rate_col = 'severe_combined_d24_p5_rate'
        if rate_col in self.county_results.columns:
            ax.scatter(self.county_results['n_events'], 
                      self.county_results[rate_col],
                      alpha=0.5, s=50)
            ax.set_xlabel('事件数量', fontsize=12)
            ax.set_ylabel('严重事件比例', fontsize=12)
            ax.set_title('县级: 事件数 vs 严重程度\n(标准: 24h & 5%)', 
                        fontsize=13, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(alpha=0.3)
        
        # 子图4: 前20个县的事件数
        ax = axes[1, 1]
        top20 = self.county_results.head(20)
        labels = [f"{row['county']}, {row['state']}" for _, row in top20.iterrows()]
        ax.barh(range(len(top20)), top20['n_events'], color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('事件数量', fontsize=12)
        ax.set_title('前20个县的停电事件数量', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_path / 'severity_overview.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存图表: {plot_file}")
        plt.close()
        
        # 图2: 热力图 - 组合条件下的严重程度
        combined_defs = [d for d in self.severity_definitions if d['method'] == 'combined']
        if combined_defs:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 创建数据矩阵
            durations = sorted(list(set([d['duration_threshold'] for d in combined_defs])))
            pcts = sorted(list(set([d['customer_pct_threshold'] for d in combined_defs])))
            
            matrix = np.zeros((len(durations), len(pcts)))
            for d in combined_defs:
                i = durations.index(d['duration_threshold'])
                j = pcts.index(d['customer_pct_threshold'])
                matrix[i, j] = d['pct_severe']
            
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                       xticklabels=[f'{p*100:.0f}%' for p in pcts],
                       yticklabels=[f'{d}h' for d in durations],
                       cbar_kws={'label': '严重事件百分比 (%)'})
            ax.set_xlabel('客户受影响百分比阈值', fontsize=12)
            ax.set_ylabel('停电时长阈值', fontsize=12)
            ax.set_title('组合条件下的严重事件分布热力图', fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            heatmap_file = output_path / 'severity_heatmap.png'
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ 保存热力图: {heatmap_file}")
            plt.close()
        
        return self
    
    def run_full_analysis(self, output_dir='./severity_results'):
        """运行完整分析"""
        self.load_data()
        self.define_severity_levels()
        self.aggregate_to_county()
        self.create_summary_report(output_dir)
        self.plot_results(output_dir)
        
        print("\n" + "="*80)
        print("✓ 分析完成！")
        print("="*80)
        print(f"\n输出文件保存在: {output_dir}/")
        print("  1. severity_definitions_summary.csv - 严重程度定义汇总")
        print("  2. county_severity_summary.csv - 县级汇总")
        print("  3. state_severity_summary.csv - 州级汇总")
        print("  4. severity_analysis_summary.txt - 分析报告")
        print("  5. severity_overview.png - 可视化图表")
        print("  6. severity_heatmap.png - 热力图")
        
        return self


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='县级停电严重程度分析'
    )
    parser.add_argument(
        'data_path',
        help='停电数据CSV文件路径'
    )
    parser.add_argument(
        '--output-dir',
        default='./severity_results',
        help='输出目录 (默认: ./severity_results)'
    )
    parser.add_argument(
        '--duration-thresholds',
        nargs='+',
        type=float,
        default=[4, 12, 24, 48],
        help='停电时长阈值（小时）'
    )
    parser.add_argument(
        '--pct-thresholds',
        nargs='+',
        type=float,
        default=[0.01, 0.02, 0.05, 0.10],
        help='客户百分比阈值'
    )
    parser.add_argument(
        '--abs-thresholds',
        nargs='+',
        type=int,
        default=[500, 1000, 5000, 10000],
        help='客户绝对数阈值'
    )
    
    args = parser.parse_args()
    
    # 运行分析
    analyzer = CountySeverityAnalyzer(args.data_path)
    analyzer.load_data()
    analyzer.define_severity_levels(
        duration_thresholds=args.duration_thresholds,
        customer_pct_thresholds=args.pct_thresholds,
        customer_abs_thresholds=args.abs_thresholds
    )
    analyzer.aggregate_to_county()
    analyzer.create_summary_report(args.output_dir)
    analyzer.plot_results(args.output_dir)
    
    print(f"\n✓ 分析完成！结果保存在: {args.output_dir}/")


if __name__ == '__main__':
    main()
