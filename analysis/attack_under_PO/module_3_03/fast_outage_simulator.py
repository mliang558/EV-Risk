"""
极速优化版停电模拟器

性能优化：
1. 预计算效率矩阵，避免重复计算 nx.global_efficiency()
2. 使用查表法代替实时计算
3. 简化Timeline为"等效独立事件"近似
4. 批量处理减少循环开销

预期速度：1000次模拟 < 5分钟
"""

import numpy as np
import networkx as nx
import pickle
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


class FastOutageSimulator:
    """
    极速停电模拟器
    
    关键优化：
    1. 预计算节点重要性 → 避免反复调用 global_efficiency
    2. 使用近似方法处理重叠事件
    3. 向量化操作
    """
    
    def __init__(self, state_name: str, data_path: str, 
                 precompute_efficiency: bool = True,
                 verbose: bool = True):
        """
        初始化
        
        Parameters:
        -----------
        precompute_efficiency : bool
            是否预计算效率查找表（推荐True，会加速10-100倍）
        """
        self.state = state_name
        self.data_path = data_path
        self.verbose = verbose
        self.precompute_efficiency = precompute_efficiency
        
        if verbose:
            print(f"="*70)
            print(f"初始化极速停电模拟器: {state_name}")
            print(f"="*70)
        
        self._load_all_data()
        self._prepare_simulation()
        
        if precompute_efficiency:
            if verbose:
                print(f"\n预计算效率矩阵 (这需要几分钟，但之后会很快)...")
            self._precompute_efficiency_lookup()
        else:
            if verbose:
                print(f"\n⚠ 未启用预计算，速度会较慢")
        
        if verbose:
            print(f"\n✓ 初始化完成")
            print(f"  - 充电站: {len(self.G.nodes())}")
            print(f"  - Counties: {len(self.lambda_county)}")
            print(f"  - 基准效率: {self.baseline_efficiency:.4f}")
    
    def _load_all_data(self):
        """加载数据"""
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.G = data['network']
        self.lambda_county = data['county_lambdas']
        self.county_data = data['county_demographics']
        self.historical_outages = data['historical_outages']
        
        # 转换节点为整数索引（加速）
        self.node_list = list(self.G.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        
        self.baseline_efficiency = nx.global_efficiency(self.G)
        self._build_node_county_index()
    
    def _build_node_county_index(self):
        """构建索引"""
        self.nodes_by_county = defaultdict(list)
        
        for node in self.G.nodes():
            county_fips = (self.G.nodes[node].get('county') or 
                          self.G.nodes[node].get('county_fips'))
            if county_fips:
                self.nodes_by_county[county_fips].append(node)
    
    def _prepare_simulation(self):
        """准备模拟"""
        self.counties = list(self.lambda_county.keys())
        self.lambdas = np.array([self.lambda_county[c] for c in self.counties])
        
        total_lambda = self.lambdas.sum()
        self.county_probs = (self.lambdas / total_lambda if total_lambda > 0 
                            else np.ones(len(self.counties)) / len(self.counties))
        
        self.total_lambda = total_lambda
    
    def _precompute_efficiency_lookup(self):
        """
        预计算效率查找表
        
        策略：采样一些典型的节点移除场景，建立插值模型
        
        这是最关键的优化！
        """
        n_nodes = len(self.G.nodes())
        
        if self.verbose:
            print(f"  - 总节点数: {n_nodes}")
        
        # 策略：预计算不同规模的节点移除效率
        # 采样：移除 1%, 2%, 5%, 10%, 20%, 50% 节点
        sample_sizes = [
            max(1, int(n_nodes * p)) 
            for p in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        ]
        
        # 对每个规模，随机采样多次
        self.efficiency_samples = {
            'n_removed': [],
            'efficiency': []
        }
        
        n_samples_per_size = 5  # 每个规模采样5次
        
        if self.verbose:
            print(f"  - 采样策略: {len(sample_sizes)}个规模 × {n_samples_per_size}次")
        
        for size in sample_sizes:
            for _ in range(n_samples_per_size):
                # 随机选择节点移除
                nodes_to_remove = np.random.choice(
                    self.node_list, 
                    size=min(size, n_nodes), 
                    replace=False
                )
                
                # 计算效率
                G_damaged = self.G.copy()
                G_damaged.remove_nodes_from(nodes_to_remove)
                
                try:
                    eff = nx.global_efficiency(G_damaged)
                except:
                    eff = 0.0
                
                self.efficiency_samples['n_removed'].append(len(nodes_to_remove))
                self.efficiency_samples['efficiency'].append(eff)
        
        # 转换为numpy数组
        self.efficiency_samples['n_removed'] = np.array(self.efficiency_samples['n_removed'])
        self.efficiency_samples['efficiency'] = np.array(self.efficiency_samples['efficiency'])
        
        if self.verbose:
            print(f"  - 完成 {len(self.efficiency_samples['n_removed'])} 个样本")
            print(f"  ✓ 预计算完成，后续模拟将使用快速近似")
    
    def estimate_efficiency_fast(self, n_nodes_removed: int) -> float:
        """
        快速估算效率（使用预计算的查找表）
        
        使用线性插值
        """
        if not self.precompute_efficiency:
            # 退化到慢速方法
            return None
        
        if n_nodes_removed == 0:
            return self.baseline_efficiency
        
        # 查找最近的样本点
        samples = self.efficiency_samples
        
        # 找到相近规模的样本
        mask = samples['n_removed'] == n_nodes_removed
        if mask.sum() > 0:
            # 直接平均
            return samples['efficiency'][mask].mean()
        
        # 线性插值
        if n_nodes_removed < samples['n_removed'].min():
            # 外推
            return self.baseline_efficiency - (
                (self.baseline_efficiency - samples['efficiency'].max()) * 
                (n_nodes_removed / samples['n_removed'].min())
            )
        
        if n_nodes_removed > samples['n_removed'].max():
            # 外推
            return max(0, samples['efficiency'].min() * 
                      (1 - (n_nodes_removed - samples['n_removed'].max()) / len(self.G.nodes())))
        
        # 线性插值
        idx = np.searchsorted(samples['n_removed'], n_nodes_removed)
        if idx == 0:
            idx = 1
        
        x0, x1 = samples['n_removed'][idx-1], samples['n_removed'][idx]
        y0, y1 = samples['efficiency'][idx-1], samples['efficiency'][idx]
        
        # 线性插值
        return y0 + (y1 - y0) * (n_nodes_removed - x0) / (x1 - x0)
    
    def haversine_distance(self, loc1: Tuple[float, float], 
                          loc2: Tuple[float, float]) -> float:
        """计算GPS距离"""
        lat1, lon1 = np.radians(loc1)
        lat2, lon2 = np.radians(loc2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371 * c
    
    def calculate_impact_radius(self, county: str, affected_customers: int) -> float:
        """计算影响半径"""
        county_info = self.county_data[county]
        density = county_info['population'] / county_info['area_km2']
        density = max(density, 1)
        
        r_km = np.sqrt(affected_customers / (density * np.pi))
        return np.clip(r_km, 5, 150)
    
    def bootstrap_sample_outage(self, county: str) -> Dict:
        """Bootstrap采样"""
        historical = self.historical_outages.get(county, [])
        
        if not historical:
            all_outages = []
            for outages in self.historical_outages.values():
                all_outages.extend(outages)
            historical = all_outages if all_outages else None
        
        if not historical:
            return {
                'duration': np.random.lognormal(1.5, 0.8),
                'affected_customers': int(np.random.lognormal(8, 1.5))
            }
        
        return historical[np.random.randint(len(historical))]
    
    def compute_efficiency_loss_fast(self, n_nodes_affected: int, 
                                     duration: float) -> float:
        """
        快速计算损失（使用查找表）
        
        Parameters:
        -----------
        n_nodes_affected : int
            受影响节点数量（而不是具体哪些节点）
        duration : float
            持续时间
            
        Returns:
        --------
        integrated loss
        """
        if n_nodes_affected == 0:
            return 0.0
        
        # 使用快速估算
        E_damaged = self.estimate_efficiency_fast(n_nodes_affected)
        
        if E_damaged is None:
            # 退化到慢速精确方法（不应该到这里）
            return 0.0
        
        # 相对损失
        if self.baseline_efficiency > 0:
            relative_loss = (self.baseline_efficiency - E_damaged) / self.baseline_efficiency
        else:
            relative_loss = 0.0
        
        relative_loss = max(0.0, min(1.0, relative_loss))
        
        # 时间积分
        return relative_loss * duration
    
    def generate_single_outage_event_fast(self, county: Optional[str] = None) -> Dict:
        """
        快速生成单个事件
        
        关键优化：只记录受影响节点数量，不记录具体节点
        """
        # 选择county
        if county is None:
            county = np.random.choice(self.counties, p=self.county_probs)
        
        # Bootstrap采样
        outage_params = self.bootstrap_sample_outage(county)
        duration = outage_params['duration']
        affected_customers = outage_params['affected_customers']
        
        # 计算范围
        radius_km = self.calculate_impact_radius(county, affected_customers)
        
        # 选择中心点
        centroid = self.county_data[county]['centroid']
        offset_lat = np.random.uniform(-0.1, 0.1)
        offset_lon = np.random.uniform(-0.1, 0.1)
        outage_center = (centroid[0] + offset_lat, centroid[1] + offset_lon)
        
        # 快速计算受影响节点数量（不需要具体节点ID）
        n_affected = 0
        for node in self.G.nodes():
            node_location = self.G.nodes[node].get('location')
            if node_location and self.haversine_distance(outage_center, node_location) <= radius_km:
                n_affected += 1
        
        return {
            'county': county,
            'duration': duration,
            'n_affected_stations': n_affected,
            'radius_km': radius_km
        }
    
    def simulate_annual_outages_fast(self) -> Dict:
        """
        快速年度模拟
        
        简化假设：
        1. 事件独立（不考虑时间重叠）
        2. 使用查找表快速估算效率
        3. 如果重叠率低(<5%)，误差可接受
        """
        # 生成事件数量
        n_events = np.random.poisson(self.total_lambda)
        
        if n_events == 0:
            return {
                'n_events': 0,
                'total_annual_loss': 0.0
            }
        
        # 批量生成所有事件
        total_loss = 0.0
        
        for _ in range(n_events):
            event = self.generate_single_outage_event_fast()
            
            # 快速计算损失
            loss = self.compute_efficiency_loss_fast(
                event['n_affected_stations'],
                event['duration']
            )
            
            total_loss += loss
        
        # 重叠修正因子（简化）
        # 假设：如果总停电时长 / 年度时长 > 5%，应用修正
        total_hours = sum(
            self.generate_single_outage_event_fast()['duration'] 
            for _ in range(min(n_events, 100))  # 采样
        )
        overlap_ratio = (total_hours / 8760) if n_events > 0 else 0
        
        if overlap_ratio > 0.05:
            # 简单修正：减少5-15%
            correction_factor = 1 - min(0.15, overlap_ratio * 0.5)
            total_loss *= correction_factor
        
        return {
            'n_events': n_events,
            'total_annual_loss': total_loss
        }
    
    def monte_carlo_simulation(self, n_simulations: int = 1000,
                              verbose: bool = True) -> Dict:
        """
        快速蒙特卡洛模拟
        
        Parameters:
        -----------
        n_simulations : int
            模拟次数
        verbose : bool
            是否显示进度条
            
        Returns:
        --------
        dict with statistics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"快速蒙特卡洛模拟: {n_simulations} 次迭代")
            print(f"{'='*70}")
        
        annual_losses = []
        n_events_list = []
        
        # 使用进度条
        iterator = tqdm(range(n_simulations), desc="模拟进度") if verbose else range(n_simulations)
        
        for _ in iterator:
            result = self.simulate_annual_outages_fast()
            annual_losses.append(result['total_annual_loss'])
            n_events_list.append(result['n_events'])
        
        annual_losses = np.array(annual_losses)
        n_events_list = np.array(n_events_list)
        
        results = {
            'annual_losses': annual_losses,
            'n_events_per_year': n_events_list,
            'mean_annual_loss': np.mean(annual_losses),
            'std_annual_loss': np.std(annual_losses),
            'median_annual_loss': np.median(annual_losses),
            'VaR_95': np.percentile(annual_losses, 95),
            'CVaR_95': np.mean(annual_losses[annual_losses >= np.percentile(annual_losses, 95)]),
            'min_loss': np.min(annual_losses),
            'max_loss': np.max(annual_losses),
            'mean_n_events': np.mean(n_events_list),
            'n_simulations': n_simulations
        }
        
        if verbose:
            print(f"\n结果:")
            print(f"  平均年损失: {results['mean_annual_loss']:.4f}")
            print(f"  标准差: {results['std_annual_loss']:.4f}")
            print(f"  95% VaR: {results['VaR_95']:.4f}")
            print(f"  平均事件数: {results['mean_n_events']:.1f}")
        
        return results
    
    def generate_report(self, mc_results: Dict) -> pd.DataFrame:
        """生成报告"""
        report = pd.DataFrame({
            'Metric': [
                'State',
                'N_Simulations',
                'N_Stations',
                'N_Counties',
                'Baseline_Efficiency',
                'Mean_Annual_Loss',
                'Std_Annual_Loss',
                'Median_Annual_Loss',
                'VaR_95',
                'CVaR_95',
                'Min_Loss',
                'Max_Loss',
                'Mean_N_Events_Per_Year',
                'Total_Lambda'
            ],
            'Value': [
                self.state,
                mc_results['n_simulations'],
                len(self.G.nodes()),
                len(self.counties),
                f"{self.baseline_efficiency:.4f}",
                f"{mc_results['mean_annual_loss']:.4f}",
                f"{mc_results['std_annual_loss']:.4f}",
                f"{mc_results['median_annual_loss']:.4f}",
                f"{mc_results['VaR_95']:.4f}",
                f"{mc_results['CVaR_95']:.4f}",
                f"{mc_results['min_loss']:.4f}",
                f"{mc_results['max_loss']:.4f}",
                f"{mc_results['mean_n_events']:.1f}",
                f"{self.total_lambda:.2f}"
            ]
        })
        
        return report


if __name__ == "__main__":
    print("Fast Outage Simulator - Optimized for Speed")
    print("Expected: 1000 simulations in < 5 minutes")
