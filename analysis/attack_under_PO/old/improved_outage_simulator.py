"""
改进版停电模拟器 - 正确处理同时发生的多个事件

关键改进：
1. 确认损失计算中已乘以duration
2. 正确处理同时发生的多个停电事件
3. 考虑事件时间重叠
"""

import numpy as np
import networkx as nx
import pickle
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class ImprovedOutageSimulator:
    """
    改进版停电模拟器
    
    关键改进：
    1. 正确处理同时发生的多个事件（节点集合并）
    2. 考虑事件时间重叠
    3. 支持年度时间线模拟
    """
    
    def __init__(self, state_name: str, data_path: str, verbose: bool = True):
        self.state = state_name
        self.data_path = data_path
        self.verbose = verbose
        
        if verbose:
            print(f"="*70)
            print(f"初始化改进版停电模拟器: {state_name}")
            print(f"="*70)
        
        self._load_all_data()
        self._prepare_simulation()
        
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
        
        self.baseline_efficiency = nx.global_efficiency(self.G)
        self._build_node_county_index()
    
    def _build_node_county_index(self):
        """构建节点-county索引"""
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
        
        if self.verbose:
            print(f"\n总年停电频率: {self.total_lambda:.2f} events/year")
    
    def haversine_distance(self, loc1: Tuple[float, float], 
                          loc2: Tuple[float, float]) -> float:
        """计算GPS距离 (km)"""
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
    
    def compute_efficiency_loss(self, affected_nodes: Set, duration: float) -> float:
        """
        计算效率损失
        
        ⚠️ 重要: 损失 = relative_loss × duration
        
        Parameters:
        -----------
        affected_nodes : set
            受影响节点集合
        duration : float
            停电持续时间 (小时)
            
        Returns:
        --------
        integrated_loss : float
            时间积分损失 (单位: dimensionless × hours)
        """
        if not affected_nodes:
            return 0.0
        
        # 移除受影响节点
        G_damaged = self.G.copy()
        G_damaged.remove_nodes_from(affected_nodes)
        
        # 计算受损效率
        try:
            E_damaged = nx.global_efficiency(G_damaged)
        except:
            E_damaged = 0.0
        
        # 相对效率损失 (0到1之间)
        if self.baseline_efficiency > 0:
            relative_loss = (self.baseline_efficiency - E_damaged) / self.baseline_efficiency
        else:
            relative_loss = 0.0
        
        relative_loss = max(0.0, min(1.0, relative_loss))
        
        # ✅ 关键: 乘以duration (恒定损失模型)
        integrated_loss = relative_loss * duration
        
        return integrated_loss
    
    def generate_single_outage_event(self, county: Optional[str] = None, 
                                     start_time: float = 0.0) -> Dict:
        """
        生成单个停电事件（不立即计算损失）
        
        Returns:
        --------
        dict with:
        - county
        - start_time (hours from year start)
        - duration (hours)
        - end_time (hours)
        - affected_nodes (set)
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
        
        # 找出受影响节点
        affected_nodes = set()
        for node in self.G.nodes():
            node_location = self.G.nodes[node].get('location')
            if node_location and self.haversine_distance(outage_center, node_location) <= radius_km:
                affected_nodes.add(node)
        
        # 计算结束时间
        end_time = start_time + duration
        
        return {
            'county': county,
            'start_time': start_time,
            'duration': duration,
            'end_time': end_time,
            'affected_customers': affected_customers,
            'radius_km': radius_km,
            'outage_center': outage_center,
            'affected_nodes': affected_nodes,
            'n_affected_stations': len(affected_nodes)
        }
    
    def compute_timeline_loss(self, events: List[Dict], verbose: bool = False) -> Dict:
        """
        计算考虑时间重叠的总损失
        
        正确处理方法:
        1. 将年度时间离散化为小时
        2. 对每小时，找出所有活跃事件
        3. 合并所有活跃事件的受影响节点
        4. 计算该小时的网络效率损失
        5. 累加所有小时的损失
        
        Parameters:
        -----------
        events : list of dict
            事件列表，每个事件包含start_time, end_time, affected_nodes
        verbose : bool
            是否输出详细信息
            
        Returns:
        --------
        dict with:
        - total_loss
        - max_simultaneous_events
        - timeline (optional, for debugging)
        """
        if not events:
            return {
                'total_loss': 0.0,
                'max_simultaneous_events': 0,
                'n_events': 0
            }
        
        # 找出最晚结束时间
        max_time = max(event['end_time'] for event in events)
        
        # 离散化时间 (每小时)
        hours = int(np.ceil(max_time)) + 1
        
        if verbose:
            print(f"\n计算年度损失:")
            print(f"  总事件数: {len(events)}")
            print(f"  时间跨度: 0 到 {max_time:.1f} 小时")
        
        # 对每小时计算
        total_loss = 0.0
        max_simultaneous = 0
        
        for hour in range(hours):
            # 找出该小时活跃的所有事件
            active_events = [
                event for event in events
                if event['start_time'] <= hour < event['end_time']
            ]
            
            n_active = len(active_events)
            max_simultaneous = max(max_simultaneous, n_active)
            
            if n_active == 0:
                continue
            
            # 合并所有活跃事件的受影响节点
            combined_affected_nodes = set()
            for event in active_events:
                combined_affected_nodes.update(event['affected_nodes'])
            
            # 计算该小时的损失 (duration = 1 hour)
            hourly_loss = self.compute_efficiency_loss(combined_affected_nodes, duration=1.0)
            total_loss += hourly_loss
            
            if verbose and n_active > 1:
                print(f"  Hour {hour}: {n_active} 同时事件, {len(combined_affected_nodes)} 节点受影响, 损失={hourly_loss:.4f}")
        
        if verbose:
            print(f"\n  总损失: {total_loss:.4f}")
            print(f"  最多同时事件: {max_simultaneous}")
        
        return {
            'total_loss': total_loss,
            'max_simultaneous_events': max_simultaneous,
            'n_events': len(events)
        }
    
    def simulate_annual_outages_timeline(self, verbose: bool = False) -> Dict:
        """
        模拟一年的停电 - 使用时间线方法（正确处理重叠）
        
        Returns:
        --------
        dict with annual results
        """
        # 生成事件数量
        n_events = np.random.poisson(self.total_lambda)
        
        if n_events == 0:
            return {
                'n_events': 0,
                'total_annual_loss': 0.0,
                'max_simultaneous_events': 0,
                'event_log': []
            }
        
        # 生成事件时间 (Poisson过程，均匀分布在8760小时内)
        event_times = np.sort(np.random.uniform(0, 8760, n_events))
        
        # 生成每个事件
        events = []
        for start_time in event_times:
            event = self.generate_single_outage_event(start_time=start_time)
            events.append(event)
        
        # 使用时间线方法计算总损失
        timeline_result = self.compute_timeline_loss(events, verbose=verbose)
        
        return {
            'n_events': n_events,
            'total_annual_loss': timeline_result['total_loss'],
            'max_simultaneous_events': timeline_result['max_simultaneous_events'],
            'event_log': events
        }
    
    def simulate_annual_outages_simple(self, verbose: bool = False) -> Dict:
        """
        简化方法 - 假设事件不重叠（原始方法）
        
        ⚠️ 注意: 这个方法假设事件独立，会高估总损失
        
        仅用于对比
        """
        n_events = np.random.poisson(self.total_lambda)
        
        total_loss = 0.0
        events = []
        
        for _ in range(n_events):
            county = np.random.choice(self.counties, p=self.county_probs)
            outage_params = self.bootstrap_sample_outage(county)
            duration = outage_params['duration']
            
            # 简化: 直接生成受影响节点
            event = self.generate_single_outage_event(county=county)
            
            # ❌ 错误: 每个事件独立计算损失
            event_loss = self.compute_efficiency_loss(event['affected_nodes'], duration)
            total_loss += event_loss
            
            events.append(event)
        
        return {
            'n_events': n_events,
            'total_annual_loss': total_loss,
            'event_log': events
        }
    
    def monte_carlo_simulation(self, n_simulations: int = 1000,
                              method: str = 'timeline',
                              verbose: bool = False) -> Dict:
        """
        蒙特卡洛模拟
        
        Parameters:
        -----------
        n_simulations : int
            模拟次数
        method : str
            'timeline': 使用时间线方法（正确）
            'simple': 假设事件独立（对比用）
        verbose : bool
            是否显示进度
            
        Returns:
        --------
        dict with statistics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"蒙特卡洛模拟")
            print(f"  方法: {method}")
            print(f"  迭代: {n_simulations}")
            print(f"{'='*70}")
        
        annual_losses = []
        n_events_list = []
        max_simultaneous_list = []
        
        for i in range(n_simulations):
            if method == 'timeline':
                result = self.simulate_annual_outages_timeline(verbose=False)
                max_simultaneous_list.append(result['max_simultaneous_events'])
            else:
                result = self.simulate_annual_outages_simple(verbose=False)
            
            annual_losses.append(result['total_annual_loss'])
            n_events_list.append(result['n_events'])
            
            if verbose and (i+1) % 100 == 0:
                print(f"  完成 {i+1}/{n_simulations}")
        
        annual_losses = np.array(annual_losses)
        n_events_list = np.array(n_events_list)
        
        results = {
            'method': method,
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
        
        if method == 'timeline' and max_simultaneous_list:
            results['mean_max_simultaneous'] = np.mean(max_simultaneous_list)
            results['max_simultaneous_observed'] = np.max(max_simultaneous_list)
        
        if verbose:
            print(f"\n结果:")
            print(f"  平均年损失: {results['mean_annual_loss']:.4f}")
            print(f"  95% VaR: {results['VaR_95']:.4f}")
            print(f"  平均事件数: {results['mean_n_events']:.1f}")
            if method == 'timeline':
                print(f"  平均最多同时事件: {results['mean_max_simultaneous']:.1f}")
                print(f"  观察到的最多同时事件: {results['max_simultaneous_observed']}")
        
        return results
    
    def compare_methods(self, n_simulations: int = 1000) -> pd.DataFrame:
        """
        对比两种方法的差异
        
        Returns:
        --------
        pd.DataFrame with comparison
        """
        print(f"\n{'='*70}")
        print(f"对比分析: Timeline方法 vs Simple方法")
        print(f"{'='*70}")
        
        # Timeline方法 (正确)
        print(f"\n运行Timeline方法 (考虑事件重叠)...")
        timeline_results = self.monte_carlo_simulation(
            n_simulations=n_simulations,
            method='timeline',
            verbose=True
        )
        
        # Simple方法 (可能高估)
        print(f"\n运行Simple方法 (假设事件独立)...")
        simple_results = self.monte_carlo_simulation(
            n_simulations=n_simulations,
            method='simple',
            verbose=True
        )
        
        # 对比
        comparison = pd.DataFrame({
            'Metric': [
                'Method',
                'Mean_Annual_Loss',
                'Std_Annual_Loss',
                'VaR_95',
                'CVaR_95',
                'Mean_N_Events',
                'Mean_Max_Simultaneous'
            ],
            'Timeline': [
                'Timeline (Correct)',
                f"{timeline_results['mean_annual_loss']:.4f}",
                f"{timeline_results['std_annual_loss']:.4f}",
                f"{timeline_results['VaR_95']:.4f}",
                f"{timeline_results['CVaR_95']:.4f}",
                f"{timeline_results['mean_n_events']:.1f}",
                f"{timeline_results.get('mean_max_simultaneous', 0):.1f}"
            ],
            'Simple': [
                'Simple (May overestimate)',
                f"{simple_results['mean_annual_loss']:.4f}",
                f"{simple_results['std_annual_loss']:.4f}",
                f"{simple_results['VaR_95']:.4f}",
                f"{simple_results['CVaR_95']:.4f}",
                f"{simple_results['mean_n_events']:.1f}",
                'N/A'
            ]
        })
        
        # 计算差异
        overestimate_pct = (
            (simple_results['mean_annual_loss'] - timeline_results['mean_annual_loss']) /
            timeline_results['mean_annual_loss'] * 100
        )
        
        print(f"\n{'='*70}")
        print(f"对比结果")
        print(f"{'='*70}")
        print(comparison.to_string(index=False))
        
        print(f"\n分析:")
        print(f"  Simple方法高估损失: {overestimate_pct:.1f}%")
        print(f"  原因: 未考虑事件时间重叠")
        print(f"  平均最多{timeline_results.get('mean_max_simultaneous', 0):.1f}个事件同时发生")
        
        return comparison


if __name__ == "__main__":
    print("Improved Outage Simulator")
    print("Correctly handles overlapping events")
