#!/usr/bin/env python3
"""
停电严重性采样器 - Phase 1快速原型
==========================================

假设：
- 你已经有 lambda[county] (停电频率)
- 现在需要：采样每次停电的 (duration, customers_pct)

输入：
- 历史停电数据 (full_pop_2022.csv)
- lambda[county] (你的频率模型)

输出：
- 完整的停电事件序列（时间 + 地点 + 严重性）
- 可用于EV充电网络仿真
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')


class OutageSeveritySampler:
    """
    停电严重性采样器
    
    功能：
    1. 从历史数据学习 P(duration, customers | county, state)
    2. 生成新的停电事件
    3. 支持小样本县的借力采样
    """
    
    def __init__(self, historical_data_path, min_samples=10):
        """
        初始化
        
        Parameters:
        -----------
        historical_data_path : str
            历史停电数据路径
        min_samples : int
            最小样本数阈值（少于此数的县从州级借力）
        """
        self.data_path = historical_data_path
        self.min_samples = min_samples
        self.df = None
        self.county_samplers = {}
        self.state_samplers = {}
        
        print("="*80)
        print("停电严重性采样器 - 初始化")
        print("="*80)
        
        self._load_and_prepare_data()
        self._build_samplers()
    
    def _load_and_prepare_data(self):
        """加载和准备历史数据"""
        print("\n[1/3] 加载历史数据...")
        
        self.df = pd.read_csv(self.data_path)
        
        # 数据清理
        self.df['duration_hr'] = self.df['duration_min'] / 60
        self.df['fips'] = self.df['fips'].astype(str).str.zfill(5)
        self.df = self.df.dropna(subset=['mean_customers', 'duration_min', 'POPESTIMATE2022'])
        self.df = self.df[(self.df['duration_hr'] > 0) & (self.df['mean_customers'] > 0)]
        self.df['customers_pct'] = (
            self.df['mean_customers'] / self.df['POPESTIMATE2022']
        ).clip(1e-6, 1)  # 避免0值
        
        # 对数变换（用于采样）
        self.df['log_duration'] = np.log(self.df['duration_hr'])
        self.df['log_customers_pct'] = np.log(self.df['customers_pct'])
        
        print(f"  总停电事件: {len(self.df):,}")
        print(f"  县数: {self.df['fips'].nunique()}")
        print(f"  州数: {self.df['state'].nunique()}")
        
        # 统计每个县的样本量
        county_counts = self.df.groupby('fips').size()
        print(f"\n  样本量统计:")
        print(f"    ≥{self.min_samples}个样本的县: {(county_counts >= self.min_samples).sum()}")
        print(f"    <{self.min_samples}个样本的县: {(county_counts < self.min_samples).sum()}")
    
    def _build_samplers(self):
        """构建县级和州级采样器"""
        print("\n[2/3] 构建采样器...")
        
        # 按县分组
        county_groups = self.df.groupby('fips')
        n_sufficient = 0
        n_insufficient = 0
        
        for fips, group in county_groups:
            if len(group) >= self.min_samples:
                # 样本充足：建立县级采样器
                self.county_samplers[fips] = {
                    'data': group[['duration_hr', 'customers_pct', 
                                   'log_duration', 'log_customers_pct']].copy(),
                    'state': group['state'].iloc[0],
                    'n_samples': len(group),
                    'method': 'county'
                }
                n_sufficient += 1
            else:
                # 样本不足：标记需要从州级借力
                self.county_samplers[fips] = {
                    'data': group[['duration_hr', 'customers_pct',
                                   'log_duration', 'log_customers_pct']].copy(),
                    'state': group['state'].iloc[0],
                    'n_samples': len(group),
                    'method': 'state'  # 将从州级采样
                }
                n_insufficient += 1
        
        print(f"  县级采样器: {n_sufficient} 个")
        print(f"  需要州级借力: {n_insufficient} 个")
        
        # 按州分组（为小样本县准备）
        state_groups = self.df.groupby('state')
        
        for state, group in state_groups:
            self.state_samplers[state] = {
                'data': group[['duration_hr', 'customers_pct',
                              'log_duration', 'log_customers_pct']].copy(),
                'n_samples': len(group)
            }
        
        print(f"  州级采样器: {len(self.state_samplers)} 个")
    
    def sample_outage(self, fips, method='auto'):
        """
        采样一次停电事件
        
        Parameters:
        -----------
        fips : str
            县FIPS代码
        method : str
            采样方法：
            - 'auto': 自动选择（推荐）
            - 'bootstrap': 直接重采样历史数据
            - 'kde': 核密度估计（更光滑）
        
        Returns:
        --------
        dict : {'duration_hr': float, 'customers_pct': float}
        """
        if fips not in self.county_samplers:
            raise ValueError(f"County {fips} not in historical data")
        
        sampler = self.county_samplers[fips]
        
        # 确定采样来源
        if sampler['method'] == 'county':
            # 从县级数据采样
            data = sampler['data']
        else:
            # 从州级数据采样
            state = sampler['state']
            data = self.state_samplers[state]['data']
        
        # 采样方法
        if method == 'auto' or method == 'bootstrap':
            # 方法1: Bootstrap（最简单，最真实）
            sample = data.sample(1).iloc[0]
            
        elif method == 'kde':
            # 方法2: KDE（更光滑，可以生成新值）
            # 在对数空间建模
            values = data[['log_duration', 'log_customers_pct']].values.T
            
            if len(values[0]) >= 2:  # KDE至少需要2个点
                kde = gaussian_kde(values)
                log_dur, log_cust = kde.resample(1).flatten()
                sample = {
                    'duration_hr': np.exp(log_dur),
                    'customers_pct': np.exp(log_cust)
                }
            else:
                # 样本太少，回退到bootstrap
                sample = data.sample(1).iloc[0]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'duration_hr': float(sample['duration_hr']),
            'customers_pct': float(sample['customers_pct']),
            'method_used': sampler['method']  # 'county' or 'state'
        }
    
    def sample_multiple_outages(self, fips, n_outages, method='auto'):
        """
        采样多次停电事件
        
        Parameters:
        -----------
        fips : str
            县FIPS代码
        n_outages : int
            停电次数
        method : str
            采样方法
        
        Returns:
        --------
        list of dict
        """
        return [self.sample_outage(fips, method) for _ in range(n_outages)]
    
    def get_county_statistics(self, fips):
        """获取县级历史统计"""
        if fips not in self.county_samplers:
            return None
        
        sampler = self.county_samplers[fips]
        data = sampler['data']
        
        return {
            'n_samples': sampler['n_samples'],
            'method': sampler['method'],
            'state': sampler['state'],
            'duration': {
                'mean': data['duration_hr'].mean(),
                'median': data['duration_hr'].median(),
                'std': data['duration_hr'].std(),
                'min': data['duration_hr'].min(),
                'max': data['duration_hr'].max(),
                'p95': data['duration_hr'].quantile(0.95)
            },
            'customers_pct': {
                'mean': data['customers_pct'].mean(),
                'median': data['customers_pct'].median(),
                'std': data['customers_pct'].std(),
                'min': data['customers_pct'].min(),
                'max': data['customers_pct'].max(),
                'p95': data['customers_pct'].quantile(0.95)
            }
        }
    
    def validate_sampler(self, fips, n_samples=1000):
        """
        验证采样器：生成样本并与历史数据对比
        
        Parameters:
        -----------
        fips : str
            县FIPS代码
        n_samples : int
            生成的样本数
        """
        print(f"\n{'='*80}")
        print(f"验证采样器 - {fips}")
        print(f"{'='*80}")
        
        # 获取历史统计
        stats = self.get_county_statistics(fips)
        if stats is None:
            print(f"县 {fips} 不在数据中")
            return
        
        print(f"\n历史数据: {stats['n_samples']} 个样本")
        print(f"采样方法: {stats['method']}")
        
        # 生成样本
        print(f"\n生成 {n_samples} 个模拟样本...")
        simulated = self.sample_multiple_outages(fips, n_samples)
        sim_durations = [s['duration_hr'] for s in simulated]
        sim_customers = [s['customers_pct'] for s in simulated]
        
        # 对比统计
        print(f"\nDuration (小时):")
        print(f"  历史: mean={stats['duration']['mean']:.2f}, "
              f"median={stats['duration']['median']:.2f}, "
              f"std={stats['duration']['std']:.2f}")
        print(f"  模拟: mean={np.mean(sim_durations):.2f}, "
              f"median={np.median(sim_durations):.2f}, "
              f"std={np.std(sim_durations):.2f}")
        
        print(f"\nCustomer %:")
        print(f"  历史: mean={stats['customers_pct']['mean']:.4f}, "
              f"median={stats['customers_pct']['median']:.4f}, "
              f"std={stats['customers_pct']['std']:.4f}")
        print(f"  模拟: mean={np.mean(sim_customers):.4f}, "
              f"median={np.median(sim_customers):.4f}, "
              f"std={np.std(sim_customers):.4f}")
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Duration对比
        ax = axes[0]
        hist_data = self.county_samplers[fips]['data']['duration_hr']
        ax.hist(hist_data, bins=30, alpha=0.5, label='Historical', 
               color='blue', density=True, edgecolor='black')
        ax.hist(sim_durations, bins=30, alpha=0.5, label='Simulated',
               color='red', density=True, edgecolor='black')
        ax.set_xlabel('Duration (hours)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Duration Distribution\n{fips}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Customer % 对比
        ax = axes[1]
        hist_data = self.county_samplers[fips]['data']['customers_pct']
        ax.hist(hist_data*100, bins=30, alpha=0.5, label='Historical',
               color='blue', density=True, edgecolor='black')
        ax.hist(np.array(sim_customers)*100, bins=30, alpha=0.5, label='Simulated',
               color='red', density=True, edgecolor='black')
        ax.set_xlabel('Customer Impact (%)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Customer Impact Distribution\n{fips}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'sampler_validation_{fips}.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ 可视化保存: sampler_validation_{fips}.png")
        
        return {
            'historical': stats,
            'simulated': {
                'duration_mean': np.mean(sim_durations),
                'duration_std': np.std(sim_durations),
                'customers_mean': np.mean(sim_customers),
                'customers_std': np.std(sim_customers)
            }
        }


class OutageSimulator:
    """
    完整的停电仿真器
    
    结合频率模型(lambda) + 严重性采样器
    """
    
    def __init__(self, severity_sampler, lambda_dict):
        """
        初始化
        
        Parameters:
        -----------
        severity_sampler : OutageSeveritySampler
            严重性采样器
        lambda_dict : dict
            县级停电频率 {fips: lambda_rate}
        """
        self.sampler = severity_sampler
        self.lambda_dict = lambda_dict
        
        print("\n[3/3] 停电仿真器就绪")
        print(f"  频率模型覆盖县数: {len(lambda_dict)}")
    
    def simulate_year(self, year=2024, random_seed=None):
        """
        模拟一年的停电事件
        
        Parameters:
        -----------
        year : int
            模拟年份
        random_seed : int
            随机种子
        
        Returns:
        --------
        pd.DataFrame : 停电事件列表
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        print(f"\n模拟 {year} 年停电事件...")
        
        all_outages = []
        
        for fips, lambda_rate in self.lambda_dict.items():
            # 生成停电次数
            n_outages = np.random.poisson(lambda_rate)
            
            if n_outages > 0:
                # 采样每次停电的严重性
                outages = self.sampler.sample_multiple_outages(fips, n_outages)
                
                for i, outage in enumerate(outages):
                    # 随机时间（均匀分布在一年中）
                    day_of_year = np.random.randint(1, 366)
                    
                    all_outages.append({
                        'fips': fips,
                        'year': year,
                        'day_of_year': day_of_year,
                        'outage_id': f"{fips}_{year}_{i}",
                        'duration_hr': outage['duration_hr'],
                        'customers_pct': outage['customers_pct'],
                        'sampling_method': outage['method_used']
                    })
        
        df_outages = pd.DataFrame(all_outages)
        
        print(f"  生成 {len(df_outages)} 次停电事件")
        print(f"  涉及 {df_outages['fips'].nunique()} 个县")
        
        return df_outages
    
    def simulate_multiple_years(self, n_years=10, start_year=2024):
        """
        模拟多年
        
        Parameters:
        -----------
        n_years : int
            模拟年数
        start_year : int
            起始年份
        
        Returns:
        --------
        pd.DataFrame
        """
        print(f"\n{'='*80}")
        print(f"多年仿真: {start_year}-{start_year+n_years-1}")
        print(f"{'='*80}")
        
        all_years = []
        
        for i in range(n_years):
            year = start_year + i
            df_year = self.simulate_year(year, random_seed=42+i)
            all_years.append(df_year)
        
        df_all = pd.concat(all_years, ignore_index=True)
        
        print(f"\n仿真总结:")
        print(f"  总停电事件: {len(df_all):,}")
        print(f"  平均每年: {len(df_all)/n_years:.1f}")
        print(f"  涉及县数: {df_all['fips'].nunique()}")
        
        return df_all
    
    def analyze_simulated_outages(self, df_outages, output_dir='./simulation_results'):
        """分析模拟的停电事件"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*80}")
        print("分析模拟结果")
        print(f"{'='*80}")
        
        # 统计
        print(f"\nDuration统计:")
        print(f"  均值: {df_outages['duration_hr'].mean():.2f} 小时")
        print(f"  中位数: {df_outages['duration_hr'].median():.2f} 小时")
        print(f"  P95: {df_outages['duration_hr'].quantile(0.95):.2f} 小时")
        print(f"  最大: {df_outages['duration_hr'].max():.2f} 小时")
        
        print(f"\nCustomer Impact统计:")
        print(f"  均值: {df_outages['customers_pct'].mean():.4f} ({df_outages['customers_pct'].mean()*100:.2f}%)")
        print(f"  中位数: {df_outages['customers_pct'].median():.4f}")
        print(f"  P95: {df_outages['customers_pct'].quantile(0.95):.4f}")
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Duration分布
        ax = axes[0, 0]
        ax.hist(df_outages['duration_hr'], bins=50, color='steelblue', 
               alpha=0.7, edgecolor='black')
        ax.set_xlabel('Duration (hours)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Simulated Outage Duration Distribution', 
                    fontsize=12, fontweight='bold')
        ax.axvline(df_outages['duration_hr'].median(), color='red', 
                  linestyle='--', label=f"Median: {df_outages['duration_hr'].median():.1f}h")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Customer Impact分布
        ax = axes[0, 1]
        ax.hist(df_outages['customers_pct']*100, bins=50, color='orange',
               alpha=0.7, edgecolor='black')
        ax.set_xlabel('Customer Impact (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Simulated Customer Impact Distribution',
                    fontsize=12, fontweight='bold')
        ax.axvline(df_outages['customers_pct'].median()*100, color='red',
                  linestyle='--', label=f"Median: {df_outages['customers_pct'].median()*100:.2f}%")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Duration vs Customer Impact
        ax = axes[1, 0]
        ax.scatter(df_outages['duration_hr'], df_outages['customers_pct']*100,
                  alpha=0.3, s=10)
        ax.set_xlabel('Duration (hours)', fontsize=11)
        ax.set_ylabel('Customer Impact (%)', fontsize=11)
        ax.set_title('Simulated: Duration vs Customer Impact',
                    fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # 4. 时间序列（如果是多年）
        ax = axes[1, 1]
        if 'year' in df_outages.columns:
            yearly_counts = df_outages.groupby('year').size()
            ax.bar(yearly_counts.index, yearly_counts.values, color='teal', alpha=0.7)
            ax.set_xlabel('Year', fontsize=11)
            ax.set_ylabel('Number of Outages', fontsize=11)
            ax.set_title('Simulated Outages by Year', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Single Year Simulation', 
                   ha='center', va='center', fontsize=14)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'simulation_summary.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ 可视化保存: {output_path / 'simulation_summary.png'}")
        
        # 保存数据
        df_outages.to_csv(output_path / 'simulated_outages.csv', index=False)
        print(f"✓ 数据保存: {output_path / 'simulated_outages.csv'}")
        
        return output_path


def example_usage():
    """使用示例"""
    print("\n" + "="*80)
    print("使用示例")
    print("="*80)
    
    print("""
# 步骤1: 初始化严重性采样器
sampler = OutageSeveritySampler('full_pop_2022.csv', min_samples=10)

# 步骤2: 准备频率模型
# 假设你已经有了lambda字典（从你的频率模型）
lambda_dict = {
    '01001': 2.5,  # Alabama, Autauga County: 平均每年2.5次停电
    '01003': 3.2,  # Alabama, Baldwin County: 平均每年3.2次停电
    # ... 你的所有县
}

# 步骤3: 初始化仿真器
simulator = OutageSimulator(sampler, lambda_dict)

# 步骤4: 运行仿真
df_outages = simulator.simulate_multiple_years(n_years=10, start_year=2024)

# 步骤5: 分析结果
simulator.analyze_simulated_outages(df_outages)

# 步骤6: 使用模拟的停电打击EV充电网络
for _, outage in df_outages.iterrows():
    county = outage['fips']
    duration = outage['duration_hr']
    impact_pct = outage['customers_pct']
    
    # 你的充电网络韧性分析代码
    # affected_chargers = network.get_chargers_in_county(county)
    # downtime = duration
    # unmet_demand = calculate_impact(affected_chargers, downtime)
    """)


if __name__ == '__main__':
    print("\n" + "="*80)
    print("停电严重性采样器 - 就绪")
    print("="*80)
    print("\n这是一个模块，请import后使用")
    print("查看 example_usage() 了解使用方法")
    print("\n或者参考文档: severity_modeling_for_simulation.txt")
