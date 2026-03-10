"""
Module 3: Realistic Power Outage Impact Simulation
模拟真实停电事件对充电站网络的影响

改进版本 - 理清逻辑和流程

核心逻辑:
1. 选择County (基于lambda概率权重)
2. Bootstrap采样该County的历史停电数据 (duration + affected_customers)
3. 根据人口密度计算影响半径
4. 在County内随机选择停电中心点
5. 找到半径内所有充电站 → Black out
6. 计算网络性能损失
"""

import numpy as np
import networkx as nx
import pickle
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')


class RealisticOutageSimulator:
    """
    真实停电影响模拟器
    
    模拟流程：
    1. 根据每个county的lambda（年停电频率）选择发生停电的county
    2. 从该county的历史数据bootstrap采样duration和affected_customers
    3. 根据affected_customers和人口密度计算影响半径
    4. 在county内随机选择停电中心点
    5. 找出半径内的所有充电站
    6. 计算网络效率损失
    """
    
    def __init__(self, state_name: str, data_path: str, verbose: bool = True):
        """
        初始化模拟器
        
        Parameters:
        -----------
        state_name : str
            州名 (e.g., 'California', 'Texas', 'Oregon')
        data_path : str
            数据pickle文件路径 (包含network, lambdas, historical_outages, demographics)
        verbose : bool
            是否输出详细信息
        """
        self.state = state_name
        self.data_path = data_path
        self.verbose = verbose
        
        # 加载数据
        if verbose:
            print(f"="*70)
            print(f"初始化 {state_name} 停电模拟器")
            print(f"="*70)
        
        self._load_all_data()
        self._validate_data()
        self._prepare_county_selection()
        
        if verbose:
            print(f"\n✓ 初始化完成")
            print(f"  - 充电站: {len(self.G.nodes())}")
            print(f"  - Counties: {len(self.lambda_county)}")
            print(f"  - 基准网络效率: {self.baseline_efficiency:.4f}")
    
    def _load_all_data(self):
        """从pickle文件加载所有数据"""
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        # 1. 充电站网络
        self.G = data['network']
        
        # 2. County停电频率 (lambda, events/year)
        # 格式: {'01001': 2.5, '01003': 1.8, ...}
        self.lambda_county = data['county_lambdas']
        
        # 3. County人口统计
        # 格式: {'01001': {'population': 100000, 'area_km2': 500, 'centroid': (lat, lon)}}
        self.county_data = data['county_demographics']
        
        # 4. 历史停电数据 (用于bootstrap)
        # 格式: {'01001': [{'duration': 3.5, 'affected_customers': 5000}, ...]}
        self.historical_outages = data['historical_outages']
        
        # 5. 计算基准网络效率
        self.baseline_efficiency = nx.global_efficiency(self.G)
        
        # 6. 为每个节点添加county信息索引
        self._build_node_county_index()
    
    def _build_node_county_index(self):
        """构建节点-county索引，加速查询"""
        self.nodes_by_county = defaultdict(list)
        
        for node in self.G.nodes():
            node_data = self.G.nodes[node]
            
            # 查找county信息
            county_fips = node_data.get('county') or node_data.get('county_fips')
            
            if county_fips:
                self.nodes_by_county[county_fips].append(node)
        
        if self.verbose:
            print(f"\n节点-County索引:")
            for county, nodes in list(self.nodes_by_county.items())[:3]:
                print(f"  {county}: {len(nodes)} 节点")
    
    def _validate_data(self):
        """验证数据完整性"""
        errors = []
        warnings_list = []
        
        # 检查节点是否有location
        nodes_without_location = 0
        for node in self.G.nodes():
            if 'location' not in self.G.nodes[node]:
                nodes_without_location += 1
        
        if nodes_without_location > 0:
            warnings_list.append(f"{nodes_without_location} 节点缺少location属性")
        
        # 检查节点是否有county
        nodes_without_county = 0
        for node in self.G.nodes():
            node_data = self.G.nodes[node]
            if not (node_data.get('county') or node_data.get('county_fips')):
                nodes_without_county += 1
        
        if nodes_without_county > 0:
            warnings_list.append(f"{nodes_without_county} 节点缺少county属性")
        
        # 检查数据一致性
        lambda_counties = set(self.lambda_county.keys())
        demo_counties = set(self.county_data.keys())
        hist_counties = set(self.historical_outages.keys())
        
        common = lambda_counties & demo_counties & hist_counties
        
        if len(common) < len(lambda_counties):
            missing = len(lambda_counties) - len(common)
            warnings_list.append(f"{missing} counties缺少完整数据")
        
        if self.verbose:
            if errors:
                print(f"\n❌ 数据错误:")
                for e in errors:
                    print(f"  - {e}")
            
            if warnings_list:
                print(f"\n⚠ 数据警告:")
                for w in warnings_list:
                    print(f"  - {w}")
            
            if not errors and not warnings_list:
                print(f"\n✓ 数据验证通过")
    
    def _prepare_county_selection(self):
        """准备county选择的概率分布"""
        # 提取所有counties和对应的lambda
        self.counties = list(self.lambda_county.keys())
        self.lambdas = np.array([self.lambda_county[c] for c in self.counties])
        
        # 计算选择概率 (正比于lambda)
        total_lambda = self.lambdas.sum()
        self.county_probs = self.lambdas / total_lambda if total_lambda > 0 else np.ones(len(self.counties)) / len(self.counties)
        
        # 总停电率
        self.total_lambda = total_lambda
        
        if self.verbose:
            print(f"\nCounty停电统计:")
            print(f"  总年停电频率: {self.total_lambda:.2f} events/year")
            print(f"  平均每county: {self.total_lambda/len(self.counties):.2f} events/year")
            
            # 显示top 5高风险counties
            top_indices = np.argsort(self.lambdas)[-5:][::-1]
            print(f"\n  Top 5 高风险counties:")
            for idx in top_indices:
                county = self.counties[idx]
                lam = self.lambdas[idx]
                print(f"    {county}: {lam:.2f} events/year")
    
    def haversine_distance(self, loc1: Tuple[float, float], 
                          loc2: Tuple[float, float]) -> float:
        """
        计算两个GPS坐标之间的距离 (km)
        
        Parameters:
        -----------
        loc1, loc2 : tuple of (lat, lon)
        
        Returns:
        --------
        distance in kilometers
        """
        lat1, lon1 = np.radians(loc1)
        lat2, lon2 = np.radians(loc2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371 * c  # 地球半径 = 6371 km
    
    def calculate_impact_radius(self, county: str, 
                               affected_customers: int) -> float:
        """
        根据受影响客户数量计算影响半径
        
        逻辑: 假设停电区域是圆形
        affected_customers = population_density × π × r²
        => r = sqrt(affected_customers / (population_density × π))
        
        Parameters:
        -----------
        county : str
            停电发生的county FIPS
        affected_customers : int
            受影响客户数量
            
        Returns:
        --------
        radius in kilometers
        """
        county_info = self.county_data[county]
        
        # 人口密度 (people per km²)
        population_density = county_info['population'] / county_info['area_km2']
        
        # 防止除零
        if population_density < 1:
            population_density = 1
        
        # 计算半径 (km)
        r_km = np.sqrt(affected_customers / (population_density * np.pi))
        
        # 合理范围: 最小5km (局部停电), 最大150km (大规模停电)
        r_km = np.clip(r_km, 5, 150)
        
        return r_km
    
    def select_outage_center(self, county: str) -> Tuple[float, float]:
        """
        在county内随机选择停电中心点
        
        为简化，使用county centroid附近的随机点
        
        Parameters:
        -----------
        county : str
            County FIPS
            
        Returns:
        --------
        (lat, lon) tuple
        """
        centroid = self.county_data[county]['centroid']
        
        # 在centroid周围随机偏移 (±0.1度 ≈ ±11km)
        offset_lat = np.random.uniform(-0.1, 0.1)
        offset_lon = np.random.uniform(-0.1, 0.1)
        
        outage_center = (
            centroid[0] + offset_lat,
            centroid[1] + offset_lon
        )
        
        return outage_center
    
    def identify_affected_nodes(self, outage_center: Tuple[float, float], 
                               radius_km: float) -> List:
        """
        找出半径内的所有充电站节点
        
        Parameters:
        -----------
        outage_center : tuple
            停电中心点 (lat, lon)
        radius_km : float
            影响半径 (km)
            
        Returns:
        --------
        list of affected node IDs
        """
        affected = []
        
        for node in self.G.nodes():
            node_location = self.G.nodes[node].get('location')
            
            if node_location is None:
                continue
            
            # 计算距离
            dist_km = self.haversine_distance(outage_center, node_location)
            
            if dist_km <= radius_km:
                affected.append(node)
        
        return affected
    
    def bootstrap_sample_outage(self, county: str) -> Dict:
        """
        从历史数据bootstrap采样停电事件
        
        Parameters:
        -----------
        county : str
            County FIPS
            
        Returns:
        --------
        dict with keys: 'duration' (hours), 'affected_customers' (count)
        """
        historical = self.historical_outages.get(county, [])
        
        if len(historical) == 0:
            # 如果该county没有历史数据，使用全州平均
            all_outages = []
            for outages in self.historical_outages.values():
                all_outages.extend(outages)
            historical = all_outages
        
        if len(historical) == 0:
            # 仍然没有数据，使用默认值
            return {
                'duration': np.random.lognormal(1.5, 0.8),  # median ~4.5 hours
                'affected_customers': int(np.random.lognormal(8, 1.5))  # median ~3000
            }
        
        # Bootstrap采样
        sampled = historical[np.random.randint(len(historical))]
        return sampled
    
    def compute_efficiency_loss(self, affected_nodes: List, 
                               duration: float,
                               recovery_model: str = 'constant') -> float:
        """
        计算时间积分的网络效率损失
        
        Parameters:
        -----------
        affected_nodes : list
            受影响节点列表
        duration : float
            停电持续时间 (hours)
        recovery_model : str
            恢复模型: 'constant' (恒定损失) 或 'exponential' (指数恢复)
            
        Returns:
        --------
        temporal integrated efficiency loss (dimensionless, unit: hour)
        """
        if len(affected_nodes) == 0:
            return 0.0
        
        # 创建受损网络
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
        
        if recovery_model == 'constant':
            # 假设整个持续时间内损失恒定
            integrated_loss = relative_loss * duration
            
        elif recovery_model == 'exponential':
            # 假设指数恢复: L(t) = L₀ × exp(-t/τ)
            # τ = duration / 2 (半衰期为总持续时间的一半)
            tau = duration / 2
            
            # 时间积分: ∫₀^T L₀ × exp(-t/τ) dt = L₀ × τ × (1 - exp(-T/τ))
            integrated_loss = relative_loss * tau * (1 - np.exp(-duration / tau))
        
        else:
            raise ValueError(f"Unknown recovery model: {recovery_model}")
        
        return integrated_loss
    
    def simulate_single_outage_event(self, county: Optional[str] = None) -> Dict:
        """
        模拟单个停电事件
        
        核心流程:
        1. 选择县 (如果未指定)
        2. Bootstrap采样duration和affected_customers
        3. 计算影响半径
        4. 随机选择停电中心
        5. 找出受影响充电站
        6. 计算效率损失
        
        Parameters:
        -----------
        county : str, optional
            指定county FIPS，如果None则根据lambda概率选择
            
        Returns:
        --------
        dict with event details
        """
        # Step 1: 选择county
        if county is None:
            county = np.random.choice(self.counties, p=self.county_probs)
        
        # Step 2: Bootstrap采样停电参数
        outage_params = self.bootstrap_sample_outage(county)
        duration = outage_params['duration']
        affected_customers = outage_params['affected_customers']
        
        # Step 3: 计算影响半径
        radius_km = self.calculate_impact_radius(county, affected_customers)
        
        # Step 4: 选择停电中心点
        outage_center = self.select_outage_center(county)
        
        # Step 5: 找出受影响节点
        affected_nodes = self.identify_affected_nodes(outage_center, radius_km)
        
        # Step 6: 计算效率损失
        efficiency_loss = self.compute_efficiency_loss(
            affected_nodes, 
            duration, 
            recovery_model='constant'
        )
        
        # 返回事件详情
        event = {
            'county': county,
            'duration': duration,
            'affected_customers': affected_customers,
            'radius_km': radius_km,
            'outage_center': outage_center,
            'affected_stations': len(affected_nodes),
            'affected_node_ids': affected_nodes,
            'efficiency_loss': efficiency_loss
        }
        
        return event
    
    def simulate_annual_outages(self, recovery_model: str = 'constant', 
                               verbose: bool = False) -> Dict:
        """
        模拟一年的停电事件
        
        基于Poisson过程:
        1. 根据总lambda生成年停电次数
        2. 对每个事件，根据lambda概率选择county
        3. 模拟每个事件
        4. 累积总损失
        
        Parameters:
        -----------
        recovery_model : str
            恢复模型
        verbose : bool
            是否输出详细信息
            
        Returns:
        --------
        dict with annual summary
        """
        # 生成年停电次数 (Poisson分布)
        n_events = np.random.poisson(self.total_lambda)
        
        if verbose:
            print(f"\n模拟年停电: {n_events} 事件")
        
        # 模拟每个事件
        total_loss = 0.0
        event_log = []
        
        for i in range(n_events):
            event = self.simulate_single_outage_event()
            total_loss += event['efficiency_loss']
            event_log.append(event)
            
            if verbose and (i+1) % 10 == 0:
                print(f"  完成 {i+1}/{n_events} 事件")
        
        avg_loss_per_event = total_loss / n_events if n_events > 0 else 0.0
        
        return {
            'n_events': n_events,
            'total_annual_loss': total_loss,
            'average_loss_per_event': avg_loss_per_event,
            'event_log': event_log
        }
    
    def monte_carlo_simulation(self, n_simulations: int = 1000,
                              recovery_model: str = 'constant',
                              verbose: bool = False) -> Dict:
        """
        蒙特卡洛模拟
        
        运行多次年度模拟以获得统计分布
        
        Parameters:
        -----------
        n_simulations : int
            模拟次数
        recovery_model : str
            恢复模型
        verbose : bool
            是否显示进度
            
        Returns:
        --------
        dict with statistics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"开始蒙特卡洛模拟: {n_simulations} 次迭代")
            print(f"{'='*70}")
        
        annual_losses = []
        n_events_list = []
        
        for i in range(n_simulations):
            result = self.simulate_annual_outages(recovery_model, verbose=False)
            annual_losses.append(result['total_annual_loss'])
            n_events_list.append(result['n_events'])
            
            if verbose and (i+1) % 100 == 0:
                print(f"  完成 {i+1}/{n_simulations} 模拟")
        
        annual_losses = np.array(annual_losses)
        n_events_list = np.array(n_events_list)
        
        # 计算统计量
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
            'n_simulations': n_simulations,
            'recovery_model': recovery_model
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"模拟完成")
            print(f"{'='*70}")
            print(f"\n结果统计:")
            print(f"  平均年损失: {results['mean_annual_loss']:.4f}")
            print(f"  标准差: {results['std_annual_loss']:.4f}")
            print(f"  95% VaR: {results['VaR_95']:.4f}")
            print(f"  95% CVaR: {results['CVaR_95']:.4f}")
            print(f"  平均年事件数: {results['mean_n_events']:.1f}")
        
        return results
    
    def generate_report(self, mc_results: Dict) -> pd.DataFrame:
        """
        生成统计报告
        
        Parameters:
        -----------
        mc_results : dict
            蒙特卡洛模拟结果
            
        Returns:
        --------
        pandas DataFrame
        """
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
                'Total_Lambda',
                'Recovery_Model'
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
                f"{self.total_lambda:.2f}",
                mc_results['recovery_model']
            ]
        })
        
        return report


def run_multi_state_comparison(state_configs: Dict[str, str],
                               n_simulations: int = 1000,
                               recovery_model: str = 'constant') -> pd.DataFrame:
    """
    多州对比分析
    
    Parameters:
    -----------
    state_configs : dict
        州名到数据路径的映射
        Format: {'California': 'data/ca_complete.pkl', 'Texas': 'data/tx_complete.pkl'}
    n_simulations : int
        每个州的模拟次数
    recovery_model : str
        恢复模型
        
    Returns:
    --------
    pd.DataFrame with comparison
    """
    results_list = []
    
    for state_name, data_path in state_configs.items():
        print(f"\n处理 {state_name}...")
        
        simulator = RealisticOutageSimulator(state_name, data_path, verbose=False)
        mc_results = simulator.monte_carlo_simulation(
            n_simulations=n_simulations,
            recovery_model=recovery_model,
            verbose=True
        )
        
        results_list.append({
            'State': state_name,
            'N_Stations': len(simulator.G.nodes()),
            'N_Counties': len(simulator.counties),
            'Total_Lambda': simulator.total_lambda,
            'Mean_Annual_Loss': mc_results['mean_annual_loss'],
            'Std_Annual_Loss': mc_results['std_annual_loss'],
            'VaR_95': mc_results['VaR_95'],
            'CVaR_95': mc_results['CVaR_95'],
            'Mean_N_Events': mc_results['mean_n_events'],
            'Loss_Per_Station': mc_results['mean_annual_loss'] / len(simulator.G.nodes()),
            'Baseline_Efficiency': simulator.baseline_efficiency
        })
    
    comparison_df = pd.DataFrame(results_list)
    
    # 归一化指标
    comparison_df['Normalized_Loss'] = (
        (comparison_df['Mean_Annual_Loss'] - comparison_df['Mean_Annual_Loss'].min()) /
        (comparison_df['Mean_Annual_Loss'].max() - comparison_df['Mean_Annual_Loss'].min())
    )
    
    # 综合脆弱性评分
    comparison_df['Composite_Normalized_Loss'] = (
        0.4 * comparison_df['Normalized_Loss'] +
        0.3 * (comparison_df['Loss_Per_Station'] / comparison_df['Loss_Per_Station'].max()) +
        0.3 * (comparison_df['VaR_95'] / comparison_df['VaR_95'].max())
    )
    
    # 排名
    comparison_df['Vulnerability_Rank'] = comparison_df['Composite_Normalized_Loss'].rank(
        ascending=False
    ).astype(int)
    
    # 排序
    comparison_df = comparison_df.sort_values('Vulnerability_Rank')
    
    return comparison_df


if __name__ == "__main__":
    print("Realistic Outage Simulator Module")
    print("Import this module and use RealisticOutageSimulator class")
