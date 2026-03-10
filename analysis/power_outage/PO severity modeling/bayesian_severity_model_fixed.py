#!/usr/bin/env python3
"""
县级停电严重程度 - 贝叶斯层级模型 (FIXED VERSION)
==========================================

FIX: Removed y_pred discrete variable from model to make it compatible with ADVI.
     Posterior predictive checks can be done after inference using pm.sample_posterior_predictive()

模型设计：
---------
1. 选择阈值：使用 Tier 2 (4h & 1%) 作为严重事件定义
   - 平衡了敏感性和特异性
   - 捕获约2%的事件，样本量充足

2. 层级结构：County → State → Global
   - 县级：每个县有自己的严重率 p_county[i]
   - 州级：县的严重率受州影响 (partial pooling)
   - 全局：整体先验

3. 协变量：
   - duration：平均停电时长（对数变换）
   - customers_pct：平均客户影响百分比（logit变换）

模型方程：
---------
y_i ~ Binomial(n_i, p_i)  # 县i的严重事件数

logit(p_i) = α_state[j[i]] + β₁·log(duration_i) + β₂·logit(customers_pct_i) + ε_i

α_state[j] ~ Normal(μ_global, σ_state)  # 州随机效应
μ_global ~ Normal(0, 2)                  # 全局截距
σ_state ~ HalfCauchy(1)                  # 州间变异

β₁, β₂ ~ Normal(0, 1)                    # 协变量效应
ε_i ~ Normal(0, σ_county)                # 县级残差
σ_county ~ HalfCauchy(1)                 # 县间变异

输出：
-----
1. 每个县的后验严重率 p_county (借力州和全局信息)
2. 州随机效应 α_state
3. 协变量效应 β₁, β₂
4. 收缩效果可视化
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.special import expit, logit
import warnings
warnings.filterwarnings('ignore')


class BayesianSeverityModel:
    """贝叶斯层级模型分析县级严重程度"""
    
    def __init__(self, data_path='full_pop_2022.csv', 
                 severity_threshold='tier2'):
        """
        初始化
        
        Parameters:
        -----------
        data_path : str
            数据文件路径
        severity_threshold : str
            严重程度阈值选择
            - 'tier1': 2h & 0.5%
            - 'tier2': 4h & 1% (推荐，默认)
            - 'tier3': 12h & 2%
            - 'tier4': 24h & 5%
        """
        self.data_path = data_path
        self.severity_threshold = severity_threshold
        self.df = None
        self.county_data = None
        self.model = None
        self.trace = None
        
        # 阈值定义
        self.thresholds = {
            'tier1': (2, 0.005, 'Tier 1: 2h & 0.5%'),
            'tier2': (4, 0.01, 'Tier 2: 4h & 1% (Default)'),
            'tier3': (12, 0.02, 'Tier 3: 12h & 2%'),
            'tier4': (24, 0.05, 'Tier 4: 24h & 5%'),
        }
        
    def load_and_prepare_data(self, min_events=10):
        """加载数据并定义严重程度"""
        print("="*80)
        print("贝叶斯层级模型 - 县级停电严重程度分析")
        print("="*80)
        
        # 获取选定阈值
        dur_thresh, pct_thresh, desc = self.thresholds[self.severity_threshold]
        print(f"\n[1/4] 使用阈值: {desc}")
        print(f"  时长 ≥ {dur_thresh} 小时")
        print(f"  客户影响 ≥ {pct_thresh*100}%")
        
        # 加载数据
        print(f"\n[2/4] 加载数据: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # 清理数据
        self.df['duration_hr'] = self.df['duration_min'] / 60
        self.df['fips'] = self.df['fips'].astype(str).str.zfill(5)
        self.df = self.df.dropna(subset=['mean_customers', 'duration_min', 'POPESTIMATE2022'])
        self.df = self.df[(self.df['duration_hr'] > 0) & (self.df['mean_customers'] > 0)]
        self.df['customers_pct'] = (
            self.df['mean_customers'] / self.df['POPESTIMATE2022']
        ).clip(0, 1)
        
        # 定义严重事件
        self.df['severe'] = (
            (self.df['duration_hr'] >= dur_thresh) & 
            (self.df['customers_pct'] >= pct_thresh)
        ).astype(int)
        
        n_severe = self.df['severe'].sum()
        pct_severe = (n_severe / len(self.df)) * 100
        print(f"  严重事件: {n_severe:,} / {len(self.df):,} ({pct_severe:.2f}%)")
        
        # 聚合到县级
        print(f"\n[3/4] 聚合到县级 (最少{min_events}个事件)")
        
        county_agg = self.df.groupby('fips').agg({
            'state': 'first',
            'county': 'first',
            'severe': ['sum', 'count'],  # n_severe, n_total
            'duration_hr': 'mean',
            'customers_pct': 'mean',
            'POPESTIMATE2022': 'first'
        }).reset_index()
        
        county_agg.columns = ['fips', 'state', 'county', 'n_severe', 'n_total',
                             'mean_duration_hr', 'mean_customers_pct', 'population']
        
        # 过滤低频县
        county_agg = county_agg[county_agg['n_total'] >= min_events].copy()
        
        # 计算原始比率
        county_agg['p_severe_raw'] = county_agg['n_severe'] / county_agg['n_total']
        
        # 对数变换（避免0和1的问题）
        county_agg['log_duration'] = np.log(county_agg['mean_duration_hr'] + 0.1)
        
        # Logit变换客户百分比（避免0和1）
        # 收缩到 [0.001, 0.999]
        pct_adj = county_agg['mean_customers_pct'].clip(0.001, 0.999)
        county_agg['logit_customers_pct'] = logit(pct_adj)
        
        # 标准化协变量（便于解释系数）
        county_agg['log_duration_std'] = (
            (county_agg['log_duration'] - county_agg['log_duration'].mean()) / 
            county_agg['log_duration'].std()
        )
        county_agg['logit_customers_pct_std'] = (
            (county_agg['logit_customers_pct'] - county_agg['logit_customers_pct'].mean()) / 
            county_agg['logit_customers_pct'].std()
        )
        
        # 创建州索引
        state_names = county_agg['state'].unique()
        self.state_mapping = {state: idx for idx, state in enumerate(state_names)}
        county_agg['state_idx'] = county_agg['state'].map(self.state_mapping)
        
        self.county_data = county_agg.reset_index(drop=True)
        
        print(f"  分析县数: {len(self.county_data)}")
        print(f"  分析州数: {len(state_names)}")
        print(f"  总事件数: {self.county_data['n_total'].sum():,}")
        print(f"  总严重事件: {self.county_data['n_severe'].sum():,}")
        
        # 数据摘要
        print(f"\n[4/4] 县级数据摘要:")
        print(f"  原始严重率 p_severe_raw:")
        print(f"    均值: {self.county_data['p_severe_raw'].mean():.4f}")
        print(f"    中位数: {self.county_data['p_severe_raw'].median():.4f}")
        print(f"    标准差: {self.county_data['p_severe_raw'].std():.4f}")
        print(f"  平均时长 (小时): {self.county_data['mean_duration_hr'].mean():.2f}")
        print(f"  平均客户%: {self.county_data['mean_customers_pct'].mean():.4f}")
        
        return self
    
    def build_hierarchical_model(self):
        """构建贝叶斯层级模型"""
        print("\n" + "="*80)
        print("构建贝叶斯层级模型")
        print("="*80)
        
        n_counties = len(self.county_data)
        n_states = len(self.state_mapping)
        
        print(f"\n模型结构:")
        print(f"  层级: County ({n_counties}) → State ({n_states}) → Global")
        print(f"  响应变量: n_severe ~ Binomial(n_total, p_county)")
        print(f"  协变量: log(duration), logit(customers_pct)")
        
        with pm.Model() as model:
            # 数据
            state_idx = self.county_data['state_idx'].values
            n_total = self.county_data['n_total'].values
            n_severe = self.county_data['n_severe'].values
            log_dur = self.county_data['log_duration_std'].values
            logit_cust = self.county_data['logit_customers_pct_std'].values
            
            # ============================================================
            # 层级结构
            # ============================================================
            
            # 全局截距
            mu_global = pm.Normal('mu_global', mu=0, sigma=2)
            
            # 州级随机效应
            sigma_state = pm.HalfCauchy('sigma_state', beta=1)
            alpha_state = pm.Normal('alpha_state', mu=mu_global, sigma=sigma_state, 
                                   shape=n_states)
            
            # 协变量效应（固定效应）
            beta_duration = pm.Normal('beta_duration', mu=0, sigma=1)
            beta_customers = pm.Normal('beta_customers', mu=0, sigma=1)
            
            # 县级残差
            sigma_county = pm.HalfCauchy('sigma_county', beta=1)
            epsilon_county = pm.Normal('epsilon_county', mu=0, sigma=sigma_county,
                                      shape=n_counties)
            
            # ============================================================
            # Logit模型
            # ============================================================
            logit_p = (
                alpha_state[state_idx] + 
                beta_duration * log_dur + 
                beta_customers * logit_cust + 
                epsilon_county
            )
            
            # 转换为概率
            p_county = pm.Deterministic('p_county', pm.math.invlogit(logit_p))
            
            # ============================================================
            # 似然函数
            # ============================================================
            y_obs = pm.Binomial('y_obs', n=n_total, p=p_county, observed=n_severe)
            
            # ============================================================
            # 后验预测检验
            # ============================================================
            # REMOVED: y_pred = pm.Binomial('y_pred', n=n_total, p=p_county, shape=n_counties)
            # FIX: This discrete variable is incompatible with ADVI
            # Use pm.sample_posterior_predictive() after inference instead
        
        self.model = model
        
        print(f"\n✓ 模型构建完成")
        print(f"\n模型参数:")
        print(f"  - mu_global: 全局截距")
        print(f"  - alpha_state[{n_states}]: 州随机效应")
        print(f"  - beta_duration: 时长效应")
        print(f"  - beta_customers: 客户%效应")
        print(f"  - sigma_state: 州间标准差")
        print(f"  - sigma_county: 县间标准差")
        print(f"  - p_county[{n_counties}]: 县级严重率")
        
        return self
    
    def sample_posterior(self, method='advi', draws=2000, tune=1000, chains=4, 
                        target_accept=0.95, n_advi=50000):
        """
        后验推断
        
        Parameters:
        -----------
        method : str
            'advi' - 变分推断（快速，~1-2分钟）⭐推荐
            'mcmc' - MCMC采样（慢，~10-15分钟，更准确）
        draws : int
            MCMC采样数（仅用于method='mcmc'）
        tune : int
            MCMC预热数（仅用于method='mcmc'）
        chains : int
            MCMC链数（仅用于method='mcmc'）
        n_advi : int
            ADVI迭代次数（仅用于method='advi'）
        """
        print("\n" + "="*80)
        
        if method == 'advi':
            print("变分推断 (ADVI) - 快速近似")
            print("="*80)
            print(f"\n参数:")
            print(f"  方法: ADVI (Automatic Differentiation Variational Inference)")
            print(f"  迭代次数: {n_advi:,}")
            print(f"  预计时间: 1-3分钟 ⚡")
            print(f"\n开始推断...")
            
            with self.model:
                # ADVI推断
                approx = pm.fit(n=n_advi, method='advi', random_seed=42)
                
                # 从近似分布中采样
                self.trace = approx.sample(draws=draws)
            
            print(f"✓ ADVI完成！")
            print(f"  ELBO (final): {approx.hist[-1]:.2f}")
            
            # 保存ELBO历史用于可视化
            self.elbo_history = approx.hist
            
        elif method == 'mcmc':
            print("MCMC采样 - 精确推断")
            print("="*80)
            print(f"\n参数:")
            print(f"  采样器: NUTS (No-U-Turn Sampler)")
            print(f"  采样数: {draws:,} (每条链)")
            print(f"  预热数: {tune:,} (每条链)")
            print(f"  链数: {chains}")
            print(f"  target_accept: {target_accept}")
            print(f"  预计时间: 10-20分钟 🐌")
            print(f"\n开始采样...")
            
            with self.model:
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    random_seed=42,
                    return_inferencedata=True
                )
            
            print(f"✓ MCMC完成！")
        
        else:
            raise ValueError(f"未知方法: {method}，请使用 'advi' 或 'mcmc'")
        
        return self
    
    def extract_results(self):
        """提取县级后验结果"""
        print("\n" + "="*80)
        print("提取后验结果")
        print("="*80)
        
        # 县级后验概率
        p_post = az.summary(self.trace, var_names=['p_county'])
        
        # 合并原始数据
        results = self.county_data[['fips', 'state', 'county', 'n_total', 'n_severe',
                                    'p_severe_raw', 'mean_duration_hr', 
                                    'mean_customers_pct', 'population']].copy()
        
        results['p_severe_mean'] = p_post['mean'].values
        results['p_severe_sd'] = p_post['sd'].values
        results['p_severe_lower'] = p_post['hdi_3%'].values
        results['p_severe_upper'] = p_post['hdi_97%'].values
        
        # 计算收缩量
        results['shrinkage'] = np.abs(results['p_severe_mean'] - results['p_severe_raw'])
        
        # 按后验均值排序
        results = results.sort_values('p_severe_mean', ascending=False).reset_index(drop=True)
        
        self.results = results
        
        # 全局参数
        global_vars = ['mu_global', 'beta_duration', 'beta_customers', 
                      'sigma_state', 'sigma_county']
        self.global_params = az.summary(self.trace, var_names=global_vars)
        
        print(f"\n✓ 结果提取完成")
        print(f"\n全局参数 (后验均值):")
        for var in global_vars:
            mean_val = self.global_params.loc[var, 'mean']
            print(f"  {var:20s}: {mean_val:7.4f}")
        
        print(f"\n前10个高风险县:")
        for i in range(min(10, len(self.results))):
            row = self.results.iloc[i]
            print(f"  {i+1:2d}. {row['county']}, {row['state']}: "
                  f"P(severe)={row['p_severe_mean']:.4f} "
                  f"[{row['p_severe_lower']:.4f}, {row['p_severe_upper']:.4f}]")
        
        return self
    
    def plot_diagnostics(self, output_dir='./bayesian_results'):
        """绘制诊断图"""
        print("\n" + "="*80)
        print("生成诊断图")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 1. ELBO收敛曲线 (仅ADVI)
        if hasattr(self, 'elbo_history'):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(self.elbo_history, color='steelblue', linewidth=1.5)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('ELBO', fontsize=12)
            ax.set_title('ADVI Convergence: Evidence Lower Bound (ELBO)', 
                        fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.axhline(y=self.elbo_history[-1], color='red', linestyle='--', 
                      alpha=0.5, label=f'Final ELBO: {self.elbo_history[-1]:.1f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_path / 'elbo_convergence.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ elbo_convergence.png")
        
        # 2. 轨迹图 (仅MCMC有意义，ADVI跳过)
        if not hasattr(self, 'elbo_history'):
            fig = az.plot_trace(
                self.trace, 
                var_names=['mu_global', 'beta_duration', 'beta_customers', 
                          'sigma_state', 'sigma_county'],
                compact=True,
                figsize=(12, 10)
            )
            plt.suptitle('MCMC Trace Plots', fontsize=16, fontweight='bold', y=1.002)
            plt.tight_layout()
            plt.savefig(output_path / 'trace_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ trace_plots.png")
        
        # 3. 后验分布
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        global_vars = ['mu_global', 'beta_duration', 'beta_customers', 
                      'sigma_state', 'sigma_county']
        
        for i, var in enumerate(global_vars):
            ax = axes[i]
            samples = self.trace.posterior[var].values.flatten()
            ax.hist(samples, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            
            mean_val = samples.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}')
            
            ax.set_xlabel(var, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Posterior: {var}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 删除多余子图
        if len(global_vars) < len(axes):
            for j in range(len(global_vars), len(axes)):
                fig.delaxes(axes[j])
        
        plt.suptitle('Posterior Distributions of Model Parameters', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'posterior_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ posterior_distributions.png")
        
        # 4. 州随机效应森林图
        state_effects = az.summary(self.trace, var_names=['alpha_state'])
        state_effects['state'] = list(self.state_mapping.keys())
        state_effects = state_effects.sort_values('mean')
        
        fig, ax = plt.subplots(figsize=(10, max(8, len(state_effects) * 0.3)))
        
        y_pos = np.arange(len(state_effects))
        ax.errorbar(state_effects['mean'], y_pos, 
                   xerr=[state_effects['mean'] - state_effects['hdi_3%'],
                         state_effects['hdi_97%'] - state_effects['mean']],
                   fmt='o', color='steelblue', ecolor='gray', capsize=3, markersize=5)
        
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(state_effects['state'], fontsize=9)
        ax.set_xlabel('State Random Effect (alpha_state)', fontsize=12)
        ax.set_title('State Random Effects with 94% HDI', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'state_random_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ state_random_effects.png")
        
        return self
    
    def plot_shrinkage_analysis(self, output_dir='./bayesian_results'):
        """绘制收缩效应分析"""
        print("\n" + "="*80)
        print("生成收缩效应图")
        print("="*80)
        
        output_path = Path(output_dir)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 子图1: 原始 vs 贝叶斯估计
        ax = axes[0, 0]
        ax.scatter(self.results['p_severe_raw'], self.results['p_severe_mean'],
                  alpha=0.5, s=50, c=self.results['n_total'], cmap='viridis')
        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='y=x (no shrinkage)')
        ax.set_xlabel('Raw Severity Rate', fontsize=11)
        ax.set_ylabel('Bayesian Posterior Mean', fontsize=11)
        ax.set_title('Shrinkage Effect: Raw vs Bayesian Estimates', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Sample Size (n_total)', fontsize=10)
        
        # 子图2: 收缩量 vs 样本量
        ax = axes[0, 1]
        ax.scatter(self.results['n_total'], self.results['shrinkage'],
                  alpha=0.5, s=50, color='coral')
        ax.set_xlabel('Sample Size (n_total)', fontsize=11)
        ax.set_ylabel('|Raw - Bayesian|', fontsize=11)
        ax.set_title('Shrinkage Magnitude vs Sample Size', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        
        # 子图3: 不确定性 vs 样本量
        ax = axes[1, 0]
        ax.scatter(self.results['n_total'], self.results['p_severe_sd'],
                  alpha=0.5, s=50, color='purple')
        ax.set_xlabel('Sample Size (n_total)', fontsize=11)
        ax.set_ylabel('Posterior SD', fontsize=11)
        ax.set_title('Uncertainty vs Sample Size', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # 子图4: 94% HDI宽度分布
        ax = axes[1, 1]
        hdi_width = self.results['p_severe_upper'] - self.results['p_severe_lower']
        ax.hist(hdi_width, bins=50, color='teal', alpha=0.7, edgecolor='black')
        ax.axvline(hdi_width.median(), color='red', linestyle='--', linewidth=2,
                  label=f'Median: {hdi_width.median():.4f}')
        ax.set_xlabel('94% HDI Width', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Credible Interval Width', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.suptitle('Bayesian Shrinkage Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'shrinkage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ shrinkage_analysis.png")
        
        return self
    
    def plot_covariate_effects(self, output_dir='./bayesian_results'):
        """绘制协变量效应"""
        print("\n" + "="*80)
        print("生成协变量效应图")
        print("="*80)
        
        output_path = Path(output_dir)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 子图1: Beta_duration 后验分布
        ax = axes[0, 0]
        beta_dur_samples = self.trace.posterior['beta_duration'].values.flatten()
        ax.hist(beta_dur_samples, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        mean_dur = beta_dur_samples.mean()
        ax.axvline(mean_dur, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_dur:.3f}')
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('beta_duration', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Effect of Log(Duration) on Severity', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 子图2: Beta_customers 后验分布
        ax = axes[0, 1]
        beta_cust_samples = self.trace.posterior['beta_customers'].values.flatten()
        ax.hist(beta_cust_samples, bins=50, color='orange', alpha=0.7, edgecolor='black')
        mean_cust = beta_cust_samples.mean()
        ax.axvline(mean_cust, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_cust:.3f}')
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('beta_customers', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Effect of Logit(Customer %) on Severity', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 子图3: Duration vs P(severe) - 实际数据 + 模型预测
        ax = axes[1, 0]
        ax.scatter(self.results['mean_duration_hr'], self.results['p_severe_mean'],
                  alpha=0.5, s=50, label='Counties')
        ax.set_xlabel('Mean Duration (hours)', fontsize=11)
        ax.set_ylabel('P(severe) - Bayesian', fontsize=11)
        ax.set_title('Duration vs Severity (County-Level)', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 子图4: Customer % vs P(severe)
        ax = axes[1, 1]
        ax.scatter(self.results['mean_customers_pct']*100, self.results['p_severe_mean'],
                  alpha=0.5, s=50, color='orange', label='Counties')
        ax.set_xlabel('Mean Customer Impact (%)', fontsize=11)
        ax.set_ylabel('P(severe) - Bayesian', fontsize=11)
        ax.set_title('Customer Impact vs Severity (County-Level)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'covariate_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ covariate_effects.png")
        
        return self
    
    def save_results(self, output_dir='./bayesian_results'):
        """保存结果"""
        print("\n" + "="*80)
        print("保存结果")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 1. 县级后验结果
        file1 = output_path / 'county_posterior_severity.csv'
        self.results.to_csv(file1, index=False)
        print(f"  ✓ {file1}")
        
        # 2. 全局参数
        file2 = output_path / 'global_parameters.csv'
        self.global_params.to_csv(file2)
        print(f"  ✓ {file2}")
        
        # 3. 州随机效应
        state_effects = az.summary(self.trace, var_names=['alpha_state'])
        state_effects['state'] = list(self.state_mapping.keys())
        file3 = output_path / 'state_random_effects.csv'
        state_effects.to_csv(file3, index=False)
        print(f"  ✓ {file3}")
        
        # 4. 文本报告
        dur_thresh, pct_thresh, desc = self.thresholds[self.severity_threshold]
        
        file4 = output_path / 'BAYESIAN_ANALYSIS_REPORT.txt'
        with open(file4, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("贝叶斯层级模型 - 县级停电严重程度分析\n")
            f.write("="*80 + "\n\n")
            
            f.write("模型设置\n")
            f.write("-"*80 + "\n")
            f.write(f"严重程度定义: {desc}\n")
            f.write(f"  - 时长阈值: ≥{dur_thresh}小时\n")
            f.write(f"  - 客户影响阈值: ≥{pct_thresh*100}%\n\n")
            
            f.write("层级结构: County → State → Global\n\n")
            
            f.write("模型参数后验估计\n")
            f.write("-"*80 + "\n")
            for param in ['mu_global', 'beta_duration', 'beta_customers', 
                         'sigma_state', 'sigma_county']:
                row = self.global_params.loc[param]
                f.write(f"{param:20s}: {row['mean']:7.4f} ")
                f.write(f"[{row['hdi_3%']:7.4f}, {row['hdi_97%']:7.4f}] ")
                f.write(f"(SD: {row['sd']:6.4f})\n")
            
            f.write("\n\n解释\n")
            f.write("-"*80 + "\n")
            beta_dur = self.global_params.loc['beta_duration', 'mean']
            beta_cust = self.global_params.loc['beta_customers', 'mean']
            
            f.write(f"1. beta_duration = {beta_dur:.4f}\n")
            if beta_dur > 0:
                f.write(f"   停电时长越长，严重程度越高（正相关）\n")
            else:
                f.write(f"   停电时长与严重程度负相关（异常，需检查）\n")
            
            f.write(f"\n2. beta_customers = {beta_cust:.4f}\n")
            if beta_cust > 0:
                f.write(f"   受影响客户比例越大，严重程度越高（正相关）\n")
            else:
                f.write(f"   客户影响与严重程度负相关（异常，需检查）\n")
            
            sigma_state = self.global_params.loc['sigma_state', 'mean']
            sigma_county = self.global_params.loc['sigma_county', 'mean']
            f.write(f"\n3. sigma_state = {sigma_state:.4f}\n")
            f.write(f"   州之间的变异程度\n")
            f.write(f"\n4. sigma_county = {sigma_county:.4f}\n")
            f.write(f"   县之间的残差变异程度\n")
            
            f.write("\n\n前20个高风险县\n")
            f.write("-"*80 + "\n")
            for i, row in self.results.head(20).iterrows():
                f.write(f"{i+1:2d}. {row['county']:25s}, {row['state']:15s}\n")
                f.write(f"    P(severe): {row['p_severe_mean']:.4f} ")
                f.write(f"[{row['p_severe_lower']:.4f}, {row['p_severe_upper']:.4f}]\n")
                f.write(f"    基于 {row['n_total']} 个事件, ")
                f.write(f"原始率: {row['p_severe_raw']:.4f}\n\n")
        
        print(f"  ✓ {file4}")
        
        return output_path
    
    def run_full_analysis(self, output_dir='./bayesian_results',
                         min_events=10, method='advi', 
                         draws=2000, tune=1000, n_advi=50000):
        """
        运行完整分析流程
        
        Parameters:
        -----------
        method : str
            'advi' - 变分推断（推荐，快速）⚡
            'mcmc' - MCMC采样（精确，慢）
        """
        
        # 步骤1: 加载数据
        self.load_and_prepare_data(min_events=min_events)
        
        # 步骤2: 构建模型
        self.build_hierarchical_model()
        
        # 步骤3: 推断
        if method == 'advi':
            self.sample_posterior(method='advi', n_advi=n_advi)
        else:
            self.sample_posterior(method='mcmc', draws=draws, tune=tune)
        
        # 步骤4: 提取结果
        self.extract_results()
        
        # 步骤5: 可视化
        self.plot_diagnostics(output_dir)
        self.plot_shrinkage_analysis(output_dir)
        self.plot_covariate_effects(output_dir)
        
        # 步骤6: 保存结果
        self.save_results(output_dir)
        
        print("\n" + "="*80)
        print("✅ 贝叶斯分析完成！")
        print("="*80)
        
        if method == 'advi':
            print(f"\n⚡ 使用了ADVI快速推断 (~100倍速度提升)")
            print(f"   优点: 速度快，适合原型开发和探索")
            print(f"   限制: 近似推断，可能低估不确定性")
            print(f"   建议: 最终发表用MCMC验证关键结果")
        
        print(f"\n输出文件 (在 {output_dir}/):")
        print("  数据:")
        print("    1. county_posterior_severity.csv - 县级后验结果")
        print("    2. global_parameters.csv - 全局参数后验")
        print("    3. state_random_effects.csv - 州随机效应")
        print("    4. BAYESIAN_ANALYSIS_REPORT.txt - 分析报告")
        print("  可视化:")
        if method == 'advi':
            print("    5. elbo_convergence.png - ADVI收敛曲线")
        print("    6. trace_plots.png - MCMC轨迹图 (ADVI跳过)")
        print("    7. posterior_distributions.png - 后验分布")
        print("    8. state_random_effects.png - 州随机效应森林图")
        print("    9. shrinkage_analysis.png - 收缩效应分析")
        print("    10. covariate_effects.png - 协变量效应")
        
        return self


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='贝叶斯层级模型 - 县级停电严重程度分析'
    )
    parser.add_argument('data_path', help='数据文件路径')
    parser.add_argument('--threshold', default='tier2',
                       choices=['tier1', 'tier2', 'tier3', 'tier4'],
                       help='严重程度阈值 (默认: tier2 = 4h & 1%%)')
    parser.add_argument('--method', default='advi',
                       choices=['advi', 'mcmc'],
                       help='推断方法: advi(快速⚡) 或 mcmc(精确🐌) (默认: advi)')
    parser.add_argument('--min-events', type=int, default=10,
                       help='最小事件数 (默认: 10)')
    parser.add_argument('--n-advi', type=int, default=50000,
                       help='ADVI迭代次数 (仅用于method=advi, 默认: 50000)')
    parser.add_argument('--draws', type=int, default=2000,
                       help='MCMC采样数 (仅用于method=mcmc, 默认: 2000)')
    parser.add_argument('--tune', type=int, default=1000,
                       help='MCMC预热数 (仅用于method=mcmc, 默认: 1000)')
    parser.add_argument('--output-dir', default='./bayesian_results',
                       help='输出目录 (默认: ./bayesian_results)')
    
    args = parser.parse_args()
    
    # 运行分析
    model = BayesianSeverityModel(args.data_path, args.threshold)
    model.run_full_analysis(
        output_dir=args.output_dir,
        min_events=args.min_events,
        method=args.method,
        draws=args.draws,
        tune=args.tune,
        n_advi=args.n_advi
    )


if __name__ == '__main__':
    main()
