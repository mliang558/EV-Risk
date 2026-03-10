#!/usr/bin/env python3
"""
县级停电严重程度 - 贝叶斯层级模型
==========================================

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
            y_pred = pm.Binomial('y_pred', n=n_total, p=p_county, shape=n_counties)
        
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
            ADVI迭代次数（仅用于method='advi'），默认50000
        """
        
        if method == 'advi':
            print("\n" + "="*80)
            print("变分推断 (ADVI) - 快速近似")
            print("="*80)
            
            print(f"\n参数:")
            print(f"  方法: ADVI (Automatic Differentiation Variational Inference)")
            print(f"  迭代次数: {n_advi:,}")
            print(f"  预计时间: 1-3分钟 ⚡")
            
            print(f"\n开始推断...")
            
            with self.model:
                # ADVI近似
                approx = pm.fit(n=n_advi, method='advi', random_seed=42)
                
                # 从近似分布中采样
                self.trace = approx.sample(draws=2000)
                
                # 转换为InferenceData格式
                self.trace = az.from_pymc3(trace=self.trace)
            
            print(f"\n✓ 推断完成")
            
            # 显示ELBO收敛曲线
            print(f"\n收敛信息:")
            print(f"  最终ELBO: {approx.hist[-1]:,.2f}")
            print(f"  (ELBO越大越好，检查是否平稳)")
            
            # 保存ELBO历史
            self.elbo_hist = approx.hist
            
            # 基本统计
            print(f"\n后验统计:")
            summary = az.summary(self.trace, 
                                var_names=['mu_global', 'beta_duration', 'beta_customers',
                                          'sigma_state', 'sigma_county'])
            print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])
            
            print(f"\n⚡ ADVI优点: 速度快 (~100倍于MCMC)")
            print(f"   ADVI限制: 近似推断，可能低估不确定性")
            print(f"   建议: 快速原型用ADVI，最终分析用MCMC")
        
        else:  # method == 'mcmc'
            print("\n" + "="*80)
            print("MCMC采样 - 精确推断")
            print("="*80)
            
            print(f"\n参数:")
            print(f"  Draws: {draws}")
            print(f"  Tune: {tune}")
            print(f"  Chains: {chains}")
            print(f"  Target accept: {target_accept}")
            print(f"  预计时间: 10-20分钟 🐌")
            
            print(f"\n开始采样...")
            
            with self.model:
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    target_accept=target_accept,
                    return_inferencedata=True,
                    random_seed=42
                )
            
            print(f"\n✓ 采样完成")
            
            # 诊断
            print(f"\n收敛诊断:")
            summary = az.summary(self.trace, 
                                var_names=['mu_global', 'beta_duration', 'beta_customers',
                                          'sigma_state', 'sigma_county'])
            print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk']])
            
            # 检查r_hat
            max_rhat = summary['r_hat'].max()
            if max_rhat > 1.01:
                print(f"\n⚠️  警告: 最大R-hat = {max_rhat:.4f} > 1.01")
                print(f"   建议增加采样数或调整target_accept")
            else:
                print(f"\n✓ 收敛良好 (最大R-hat = {max_rhat:.4f})")
        
        return self
    
    def extract_results(self):
        """提取后验结果"""
        print("\n" + "="*80)
        print("提取后验结果")
        print("="*80)
        
        # 提取县级后验
        p_county_samples = self.trace.posterior['p_county'].values
        p_county_samples = p_county_samples.reshape(-1, len(self.county_data))
        
        results = self.county_data[['fips', 'state', 'county', 'n_total', 'n_severe',
                                    'p_severe_raw', 'mean_duration_hr', 
                                    'mean_customers_pct']].copy()
        
        # 后验统计
        results['p_severe_mean'] = p_county_samples.mean(axis=0)
        results['p_severe_sd'] = p_county_samples.std(axis=0)
        results['p_severe_lower'] = np.percentile(p_county_samples, 2.5, axis=0)
        results['p_severe_upper'] = np.percentile(p_county_samples, 97.5, axis=0)
        
        # 计算收缩效果
        results['shrinkage'] = np.abs(
            results['p_severe_mean'] - results['p_severe_raw']
        ) / (results['p_severe_raw'] + 1e-6)
        
        # 排序
        results = results.sort_values('p_severe_mean', ascending=False).reset_index(drop=True)
        
        print(f"\n前10个高风险县:")
        print(results[['county', 'state', 'n_total', 'p_severe_raw', 
                      'p_severe_mean', 'p_severe_lower', 'p_severe_upper']].head(10))
        
        # 提取全局参数
        global_params = az.summary(
            self.trace, 
            var_names=['mu_global', 'beta_duration', 'beta_customers',
                      'sigma_state', 'sigma_county']
        )
        
        print(f"\n全局参数后验:")
        print(global_params[['mean', 'sd', 'hdi_3%', 'hdi_97%']])
        
        self.results = results
        self.global_params = global_params
        
        return results
    
    def plot_diagnostics(self, output_dir='./bayesian_results'):
        """绘制诊断图"""
        print("\n" + "="*80)
        print("生成诊断图")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 如果是ADVI，绘制ELBO收敛曲线
        if hasattr(self, 'elbo_hist'):
            print("  绘制ELBO收敛曲线...")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.elbo_hist, linewidth=2, color='steelblue')
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('ELBO', fontsize=11)
            ax.set_title('ADVI Convergence (ELBO over iterations)', 
                        fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            
            # 标注最后10%的平均值
            last_10pct = int(len(self.elbo_hist) * 0.9)
            mean_last = np.mean(self.elbo_hist[last_10pct:])
            ax.axhline(mean_last, color='red', linestyle='--', linewidth=2,
                      label=f'Mean (last 10%): {mean_last:,.1f}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_path / 'elbo_convergence.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    ✓ elbo_convergence.png")
        
        # 图1: Trace plots (如果是MCMC才有多条链)
        print("  绘制trace plots...")
        try:
            axes = az.plot_trace(
                self.trace,
                var_names=['mu_global', 'beta_duration', 'beta_customers',
                          'sigma_state', 'sigma_county'],
                figsize=(14, 10)
            )
            plt.tight_layout()
            plt.savefig(output_path / 'trace_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    ✓ trace_plots.png")
        except:
            print(f"    ⚠️  trace_plots跳过 (ADVI无链概念)")
        
        # 图2: Posterior distributions
        print("  绘制posterior distributions...")
        axes = az.plot_posterior(
            self.trace,
            var_names=['mu_global', 'beta_duration', 'beta_customers',
                      'sigma_state', 'sigma_county'],
            figsize=(14, 10),
            hdi_prob=0.95
        )
        plt.tight_layout()
        plt.savefig(output_path / 'posterior_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ posterior_distributions.png")
        
        # 图3: Forest plot - 州随机效应
        print("  绘制state random effects...")
        fig, ax = plt.subplots(figsize=(10, max(8, len(self.state_mapping) * 0.3)))
        
        alpha_state_summary = az.summary(self.trace, var_names=['alpha_state'])
        state_names = list(self.state_mapping.keys())
        
        y_pos = np.arange(len(state_names))
        ax.errorbar(
            alpha_state_summary['mean'], y_pos,
            xerr=[alpha_state_summary['mean'] - alpha_state_summary['hdi_3%'],
                  alpha_state_summary['hdi_97%'] - alpha_state_summary['mean']],
            fmt='o', capsize=5, capthick=2
        )
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(state_names, fontsize=8)
        ax.set_xlabel('State Random Effect (α_state)', fontsize=11)
        ax.set_title('State Random Effects with 95% HDI', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_path / 'state_random_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ state_random_effects.png")
        
        return self
    
    def plot_shrinkage_analysis(self, output_dir='./bayesian_results'):
        """绘制收缩效应分析"""
        print("\n" + "="*80)
        print("生成收缩效应分析图")
        print("="*80)
        
        output_path = Path(output_dir)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1: Raw vs Bayesian
        ax = axes[0, 0]
        scatter = ax.scatter(
            self.results['p_severe_raw'],
            self.results['p_severe_mean'],
            s=self.results['n_total']/10,
            alpha=0.6,
            c=self.results['n_total'],
            cmap='viridis'
        )
        max_val = max(self.results['p_severe_raw'].max(),
                     self.results['p_severe_mean'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel('Raw P(severe)', fontsize=11)
        ax.set_ylabel('Bayesian P(severe)', fontsize=11)
        ax.set_title('Shrinkage Effect\n(Size = # events)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='# Events')
        
        # 子图2: Shrinkage vs Sample Size
        ax = axes[0, 1]
        ax.scatter(self.results['n_total'], self.results['shrinkage'], alpha=0.6)
        ax.set_xlabel('Number of Events', fontsize=11)
        ax.set_ylabel('Shrinkage (|Bayes - Raw| / Raw)', fontsize=11)
        ax.set_title('Shrinkage vs Sample Size', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # 子图3: Uncertainty vs Sample Size
        ax = axes[1, 0]
        ci_width = self.results['p_severe_upper'] - self.results['p_severe_lower']
        ax.scatter(self.results['n_total'], ci_width, alpha=0.6, color='coral')
        ax.set_xlabel('Number of Events', fontsize=11)
        ax.set_ylabel('95% Credible Interval Width', fontsize=11)
        ax.set_title('Uncertainty vs Sample Size', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        
        # 子图4: Posterior distributions for selected counties
        ax = axes[1, 1]
        
        # 选择3个县: 高风险、中风险、低风险
        high_risk = self.results.iloc[0]
        mid_risk = self.results.iloc[len(self.results)//2]
        low_risk = self.results.iloc[-1]
        
        p_samples = self.trace.posterior['p_county'].values.reshape(-1, len(self.results))
        
        for county, color, label in [
            (0, 'red', f"High: {high_risk['county']}, {high_risk['state']}"),
            (len(self.results)//2, 'orange', f"Mid: {mid_risk['county']}, {mid_risk['state']}"),
            (len(self.results)-1, 'green', f"Low: {low_risk['county']}, {low_risk['state']}")
        ]:
            ax.hist(p_samples[:, county], bins=50, alpha=0.5, 
                   label=label, color=color, density=True)
        
        ax.set_xlabel('P(severe)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Posterior Distributions (Examples)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        
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
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 提取协变量系数样本
        beta_dur_samples = self.trace.posterior['beta_duration'].values.flatten()
        beta_cust_samples = self.trace.posterior['beta_customers'].values.flatten()
        
        # 子图1: Duration效应后验分布
        ax = axes[0, 0]
        ax.hist(beta_dur_samples, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        mean_dur = beta_dur_samples.mean()
        ax.axvline(mean_dur, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_dur:.3f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('β_duration', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Effect of Log(Duration) on Severity', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # 子图2: Customer %效应后验分布
        ax = axes[0, 1]
        ax.hist(beta_cust_samples, bins=50, edgecolor='black', alpha=0.7, color='coral')
        mean_cust = beta_cust_samples.mean()
        ax.axvline(mean_cust, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_cust:.3f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('β_customers', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Effect of Logit(Customer %) on Severity', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
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
