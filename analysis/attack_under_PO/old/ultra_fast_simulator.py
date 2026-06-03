"""
超级极速停电模拟器 V2

核心思想：用度中心性近似代替global_efficiency
- 避免调用 nx.global_efficiency() (太慢！)
- 使用节点度数的简单加权平均作为网络"效率"的代理
- 预期：1000次模拟 < 2分钟

理论依据：
- 网络效率主要反映连通性
- 节点度数是连通性的良好代理指标
- 移除高度节点 → 连通性下降 → "效率"下降

适用场景：
- 快速探索和测试
- 大规模参数扫描
- 初步结果验证

警告：
- 这是近似方法，精度损失~10-20%
- 适合相对比较，不适合精确数值
"""

import numpy as np
import networkx as nx
import pickle
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.spatial import cKDTree
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


class UltraFastSimulator:
    """
    超级极速模拟器
    
    关键创新：
    1. 用度中心性代替global_efficiency
    2. 预计算所有节点的度数
    3. 效率 ≈ 平均度数 / 最大度数
    """
    
    def __init__(self, state_name: str, data_path: str, verbose: bool = True):
        self.state = state_name
        self.data_path = data_path
        self.verbose = verbose
        
        if verbose:
            print(f"="*70)
            print(f"初始化超级极速模拟器: {state_name}")
            print(f"="*70)
        
        self._load_all_data()
        self._prepare_simulation()
        self._build_spatial_index()
        self._precompute_degree_metrics()
        
        if verbose:
            print(f"\n✓ 初始化完成")
            print(f"  - 充电站: {len(self.G.nodes())}")
            print(f"  - Counties: {len(self.lambda_county)}")
            print(f"  - 基准效率(度代理): {self.baseline_efficiency_proxy:.4f}")
    
    def _load_all_data(self):
        """加载数据"""
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.G = data['network']
        self.lambda_county = data['county_lambdas']
        self.county_data = data['county_demographics']
        self.historical_outages = data['historical_outages']
        
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
        """准备模拟参数"""
        self.counties = list(self.lambda_county.keys())
        self.lambdas = np.array([self.lambda_county[c] for c in self.counties])
        
        total_lambda = self.lambdas.sum()
        self.county_probs = (self.lambdas / total_lambda if total_lambda > 0 
                            else np.ones(len(self.counties)) / len(self.counties))
        
        self.total_lambda = total_lambda
    
    def _build_spatial_index(self):
        """构建空间索引"""
        if self.verbose:
            print(f"\n构建空间索引...")
        
        self.node_list = []
        self.node_coords = []
        
        for node in self.G.nodes():
            location = self.G.nodes[node].get('location')
            if location:
                self.node_list.append(node)
                self.node_coords.append(location)
        
        self.node_coords = np.array(self.node_coords)
        self.spatial_tree = cKDTree(self.node_coords)
        
        if self.verbose:
            print(f"  ✓ 空间索引: {len(self.node_list)} 节点")
    
    def _precompute_degree_metrics(self):
        """
        预计算度中心性指标
        
        这是核心优化！用度数代替global_efficiency
        """
        if self.verbose:
            print(f"\n预计算度中心性指标...")
        
        # 计算所有节点的度数
        self.node_degrees = dict(self.G.degree())
        
        # 转换为数组（按node_list顺序）
        self.degree_array = np.array([
            self.node_degrees.get(node, 0) 
            for node in self.node_list
        ])
        
        # 计算基准"效率"（度的归一化平均）
        self.max_degree = max(self.node_degrees.values()) if self.node_degrees else 1
        self.baseline_efficiency_proxy = np.mean(self.degree_array) / self.max_degree
        
        # 预计算每个节点的"重要性"（度数 / 最大度数）
        self.node_importance = {
            node: deg / self.max_degree 
            for node, deg in self.node_degrees.items()
        }
        
        if self.verbose:
            print(f"  ✓ 度中心性计算完成")
            print(f"    平均度数: {np.mean(self.degree_array):.2f}")
            print(f"    最大度数: {self.max_degree}")
            print(f"    基准效率代理: {self.baseline_efficiency_proxy:.4f}")
    
    def haversine_distance_vectorized(self, loc1: Tuple[float, float], 
                                     locs2: np.ndarray) -> np.ndarray:
        """向量化距离计算"""
        lat1, lon1 = np.radians(loc1)
        lats2 = np.radians(locs2[:, 0])
        lons2 = np.radians(locs2[:, 1])
        
        dlat = lats2 - lat1
        dlon = lons2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371 * c
    
    def find_affected_stations_fast(self, outage_center: Tuple[float, float], 
                                    radius_km: float) -> List:
        """使用KD树快速找出受影响节点"""
        radius_deg = radius_km / 111.0
        indices = self.spatial_tree.query_ball_point(outage_center, radius_deg)
        
        if not indices:
            return []
        
        candidate_coords = self.node_coords[indices]
        distances = self.haversine_distance_vectorized(outage_center, candidate_coords)
        
        mask = distances <= radius_km
        affected_indices = np.array(indices)[mask]
        
        return [self.node_list[i] for i in affected_indices]
    
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
    
    def select_outage_center(self, county: str) -> Tuple[float, float]:
        """选择停电中心点"""
        centroid = self.county_data[county]['centroid']
        offset_lat = np.random.uniform(-0.1, 0.1)
        offset_lon = np.random.uniform(-0.1, 0.1)
        return (centroid[0] + offset_lat, centroid[1] + offset_lon)
    
    def compute_efficiency_loss_ultra_fast(self, affected_nodes: List, 
                                          duration: float) -> float:
        """
        超快效率损失计算（使用度数代理）
        
        核心思想：
        - 效率 ≈ 剩余节点的平均度数
        - 不需要计算最短路径！
        
        Parameters:
        -----------
        affected_nodes : list
            受影响节点
        duration : float
            持续时间
            
        Returns:
        --------
        integrated loss
        """
        if not affected_nodes:
            return 0.0
        
        # 方法1: 简单版 - 基于受影响节点的度数占比
        affected_degree_sum = sum(self.node_degrees.get(node, 0) for node in affected_nodes)
        total_degree_sum = sum(self.node_degrees.values())
        
        if total_degree_sum == 0:
            return 0.0
        
        # 相对损失 ≈ 受影响节点的度数占比
        relative_loss = affected_degree_sum / total_degree_sum
        
        # 调整：考虑连接性的非线性影响
        # 经验公式：loss = (affected_degree_ratio)^0.8
        relative_loss = relative_loss ** 0.8
        
        relative_loss = max(0.0, min(1.0, relative_loss))
        
        # 时间积分
        return relative_loss * duration
    
    def simulate_single_outage_event(self, county: Optional[str] = None) -> Dict:
        """模拟单个事件（超快版本）"""
        # 选择county
        if county is None:
            county = np.random.choice(self.counties, p=self.county_probs)
        
        # Bootstrap采样
        outage_params = self.bootstrap_sample_outage(county)
        duration = outage_params['duration']
        affected_customers = outage_params['affected_customers']
        
        # 计算半径
        radius_km = self.calculate_impact_radius(county, affected_customers)
        
        # 选择中心点
        outage_center = self.select_outage_center(county)
        
        # 找出受影响节点（使用KD树）
        affected_nodes = self.find_affected_stations_fast(outage_center, radius_km)
        
        # 超快损失计算（使用度数代理）
        efficiency_loss = self.compute_efficiency_loss_ultra_fast(affected_nodes, duration)
        
        return {
            'county': county,
            'duration': duration,
            'affected_customers': affected_customers,
            'radius_km': radius_km,
            'affected_stations': len(affected_nodes),
            'efficiency_loss': efficiency_loss
        }
    
    def simulate_annual_outages(self) -> Dict:
        """年度模拟"""
        n_events = np.random.poisson(self.total_lambda)
        
        if n_events == 0:
            return {
                'n_events': 0,
                'total_annual_loss': 0.0
            }
        
        total_loss = 0.0
        
        for _ in range(n_events):
            event = self.simulate_single_outage_event()
            total_loss += event['efficiency_loss']
        
        return {
            'n_events': n_events,
            'total_annual_loss': total_loss
        }
    
    def monte_carlo_simulation(self, n_simulations: int = 1000,
                              verbose: bool = True) -> Dict:
        """蒙特卡洛模拟"""
        if verbose:
            print(f"\n{'='*70}")
            print(f"超级极速蒙特卡洛模拟: {n_simulations} 次")
            print(f"{'='*70}")
        
        annual_losses = []
        n_events_list = []
        
        iterator = tqdm(range(n_simulations), desc="模拟进度") if verbose else range(n_simulations)
        
        for _ in iterator:
            result = self.simulate_annual_outages()
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
                'Baseline_Efficiency_Proxy',
                'Mean_Annual_Loss',
                'Std_Annual_Loss',
                'VaR_95',
                'CVaR_95',
                'Mean_N_Events',
                'Method'
            ],
            'Value': [
                self.state,
                mc_results['n_simulations'],
                len(self.G.nodes()),
                len(self.counties),
                f"{self.baseline_efficiency_proxy:.4f}",
                f"{mc_results['mean_annual_loss']:.4f}",
                f"{mc_results['std_annual_loss']:.4f}",
                f"{mc_results['VaR_95']:.4f}",
                f"{mc_results['CVaR_95']:.4f}",
                f"{mc_results['mean_n_events']:.1f}",
                'Degree-based approximation'
            ]
        })
        
        return report


if __name__ == "__main__":
    print("Ultra Fast Simulator V2")
    print("Uses degree centrality as efficiency proxy")
    print("Expected: 1000 simulations in < 2 minutes")
