"""
Module 3: Realistic Power Outage Impact Simulation
模拟真实停电事件对充电站网络的影响

Author: Your Name
Date: 2024
"""

import numpy as np
import networkx as nx
import pickle
import pandas as pd
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')


class RealisticOutageSimulator:
    """
    Realistic Power Outage Impact Simulator
    
    This class simulates the impact of realistic power outage events on 
    electric vehicle charging station networks, incorporating:
    - Bayesian-estimated county-level outage frequencies
    - Bootstrap-sampled outage severity from historical data
    - Spatial propagation based on population density
    - Temporal integration of network efficiency loss
    """
    
    def __init__(self, state_name: str, data_path: str):
        """
        Initialize the simulator
        
        Parameters:
        -----------
        state_name : str
            State name (e.g., 'California', 'Texas')
        data_path : str
            Path to pickle file containing network data
        """
        self.state = state_name
        self.data_path = data_path
        
        # Load all data
        print(f"Loading data for {state_name}...")
        self._load_all_data()
        print(f"Data loaded: {len(self.G.nodes())} charging stations")
        
    def _load_all_data(self):
        """Load all necessary data from pickle file"""
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Charging station network
        self.G = data['network']  # NetworkX graph
        
        # County-level outage rates (Bayesian estimated lambda)
        # Format: {'County1': 2.5, 'County2': 1.8, ...}  # events per year
        self.lambda_county = data['county_lambdas']
        
        # County demographics
        # Format: {'County1': {'population': 100000, 'area_km2': 500, 'centroid': (lat, lon)}}
        self.county_data = data['county_demographics']
        
        # Bootstrap sampler (historical outage data)
        # Format: {'County1': [{'duration': 3.5, 'affected_customers': 5000}, ...]}
        self.historical_outages = data['historical_outages']
        
        # State-level summary
        self.state_info = {
            'total_population': sum(c['population'] for c in self.county_data.values()),
            'total_area_km2': sum(c['area_km2'] for c in self.county_data.values()),
            'n_counties': len(self.county_data)
        }
        
        # Calculate baseline network efficiency
        self.baseline_efficiency = nx.global_efficiency(self.G)
        print(f"Baseline network efficiency: {self.baseline_efficiency:.4f}")
    
    def haversine_distance(self, loc1: Tuple[float, float], 
                          loc2: Tuple[float, float]) -> float:
        """
        Calculate distance between two GPS coordinates (km)
        
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
        
        return 6371 * c  # Earth radius = 6371 km
    
    def calculate_impact_radius(self, county: str, 
                               affected_customers: int) -> float:
        """
        Calculate impact radius based on number of affected customers
        
        Logic: Assume circular outage area
        affected_customers = population_density × π × r²
        => r = sqrt(affected_customers / (population_density × π))
        
        Parameters:
        -----------
        county : str
            County where outage occurs
        affected_customers : int
            Number of affected customers
            
        Returns:
        --------
        radius in kilometers
        """
        county_info = self.county_data[county]
        
        # Population density (people per km²)
        population_density = county_info['population'] / county_info['area_km2']
        
        # Prevent division by zero
        if population_density < 1:
            population_density = 1
        
        # Calculate radius (km)
        r_km = np.sqrt(affected_customers / (population_density * np.pi))
        
        # Reasonable bounds: min 5km (local outage), max 150km (large-scale outage)
        r_km = np.clip(r_km, 5, 150)
        
        return r_km
    
    def identify_affected_nodes(self, county: str, radius_km: float) -> List:
        """
        Identify all charging station nodes within radius r from county center
        
        Parameters:
        -----------
        county : str
            County where outage occurs
        radius_km : float
            Impact radius (km)
            
        Returns:
        --------
        list of affected node IDs
        """
        county_center = self.county_data[county]['centroid']  # (lat, lon)
        
        affected = []
        for node in self.G.nodes():
            # Assume nodes have 'location' attribute: (lat, lon)
            node_location = self.G.nodes[node].get('location')
            
            if node_location is None:
                continue
            
            # Calculate distance
            dist_km = self.haversine_distance(county_center, node_location)
            
            if dist_km <= radius_km:
                affected.append(node)
        
        return affected
    
    def bootstrap_sample_outage(self, county: str) -> Dict:
        """
        Bootstrap sample an outage event from historical data
        
        Parameters:
        -----------
        county : str
            County name
            
        Returns:
        --------
        dict with keys: 'duration' (hours), 'affected_customers' (count)
        """
        historical = self.historical_outages.get(county, [])
        
        if len(historical) == 0:
            # If no historical data, use state average
            all_outages = []
            for outages in self.historical_outages.values():
                all_outages.extend(outages)
            historical = all_outages
        
        if len(historical) == 0:
            # Still no data, use default values
            return {
                'duration': np.random.lognormal(1.5, 0.8),  # ~4.5 hours median
                'affected_customers': int(np.random.lognormal(8, 1.5))  # ~3000 customers
            }
        
        # Bootstrap sample
        sampled = historical[np.random.randint(len(historical))]
        return sampled
    
    def compute_efficiency_loss(self, affected_nodes: List, 
                               duration: float,
                               recovery_model: str = 'constant') -> float:
        """
        Calculate temporal integrated network efficiency loss
        
        Parameters:
        -----------
        affected_nodes : list
            List of affected nodes
        duration : float
            Outage duration (hours)
        recovery_model : str
            Recovery model: 'constant' (constant loss) or 'exponential' (exponential recovery)
            
        Returns:
        --------
        temporal integrated efficiency loss (dimensionless, unit: hour)
        """
        if len(affected_nodes) == 0:
            return 0.0
        
        # Create damaged network
        G_damaged = self.G.copy()
        G_damaged.remove_nodes_from(affected_nodes)
        
        # Calculate damaged efficiency
        try:
            E_damaged = nx.global_efficiency(G_damaged)
        except:
            E_damaged = 0.0
        
        # Relative efficiency loss (between 0 and 1)
        relative_loss = (self.baseline_efficiency - E_damaged) / self.baseline_efficiency
        
        if recovery_model == 'constant':
            # Assume constant loss throughout duration
            integrated_loss = relative_loss * duration
            
        elif recovery_model == 'exponential':
            # Assume exponential recovery: L(t) = L₀ × exp(-t/τ)
            # τ = duration / 2 (half-life is half of total duration)
            tau = duration / 2
            
            # Integral: ∫₀ᵀ L₀×exp(-t/τ) dt = L₀×τ×(1 - exp(-T/τ))
            integrated_loss = relative_loss * tau * (1 - np.exp(-duration / tau))
        
        else:
            raise ValueError(f"Unknown recovery model: {recovery_model}")
        
        return integrated_loss
    
    def simulate_single_outage_event(self, county: str, 
                                     recovery_model: str = 'constant') -> Dict:
        """
        Simulate a single outage event
        
        Parameters:
        -----------
        county : str
            County where outage occurs
        recovery_model : str
            Recovery model
            
        Returns:
        --------
        dict containing event details and loss
        """
        # Step 1: Bootstrap sample outage characteristics
        outage = self.bootstrap_sample_outage(county)
        duration = outage['duration']
        affected_customers = outage['affected_customers']
        
        # Step 2: Calculate impact radius
        radius_km = self.calculate_impact_radius(county, affected_customers)
        
        # Step 3: Identify affected nodes
        affected_nodes = self.identify_affected_nodes(county, radius_km)
        
        # Step 4: Calculate efficiency loss
        loss = self.compute_efficiency_loss(
            affected_nodes, 
            duration, 
            recovery_model
        )
        
        return {
            'county': county,
            'duration': duration,
            'affected_customers': affected_customers,
            'radius_km': radius_km,
            'affected_stations': len(affected_nodes),
            'efficiency_loss': loss
        }
    
    def simulate_annual_outages(self, time_horizon: int = 365,
                               recovery_model: str = 'constant',
                               verbose: bool = False) -> Dict:
        """
        Simulate all outage events within one year
        
        Parameters:
        -----------
        time_horizon : int
            Simulation duration (days)
        recovery_model : str
            Recovery model
        verbose : bool
            Whether to print detailed information
            
        Returns:
        --------
        dict containing annual statistics
        """
        total_loss = 0.0
        event_log = []
        
        # For each county, generate outage events based on Poisson process
        for county, lambda_annual in self.lambda_county.items():
            # Generate number of outages for this county in one year
            n_events = np.random.poisson(lambda_annual)
            
            if verbose and n_events > 0:
                print(f"  {county}: {n_events} outage(s)")
            
            # Simulate each outage
            for _ in range(n_events):
                event = self.simulate_single_outage_event(county, recovery_model)
                event_log.append(event)
                total_loss += event['efficiency_loss']
        
        return {
            'total_annual_loss': total_loss,
            'n_events': len(event_log),
            'event_log': event_log,
            'average_loss_per_event': total_loss / max(len(event_log), 1)
        }
    
    def monte_carlo_simulation(self, n_simulations: int = 1000,
                              time_horizon: int = 365,
                              recovery_model: str = 'constant',
                              verbose: bool = True) -> Dict:
        """
        Monte Carlo simulation: Run multiple annual outage scenarios
        
        Parameters:
        -----------
        n_simulations : int
            Number of simulations
        time_horizon : int
            Duration of each simulation (days)
        recovery_model : str
            Recovery model
        verbose : bool
            Whether to display progress
            
        Returns:
        --------
        dict containing simulation results
        """
        print(f"\nRunning {n_simulations} Monte Carlo simulations for {self.state}...")
        
        annual_losses = []
        n_events_list = []
        
        for i in range(n_simulations):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_simulations} simulations")
            
            result = self.simulate_annual_outages(
                time_horizon=time_horizon,
                recovery_model=recovery_model,
                verbose=False
            )
            
            annual_losses.append(result['total_annual_loss'])
            n_events_list.append(result['n_events'])
        
        annual_losses = np.array(annual_losses)
        n_events_list = np.array(n_events_list)
        
        return {
            'mean_annual_loss': np.mean(annual_losses),
            'std_annual_loss': np.std(annual_losses),
            'median_annual_loss': np.median(annual_losses),
            'loss_distribution': annual_losses,
            'VaR_95': np.percentile(annual_losses, 95),
            'CVaR_95': np.mean(annual_losses[annual_losses > np.percentile(annual_losses, 95)]),
            'mean_n_events': np.mean(n_events_list),
            'total_expected_events': np.sum(list(self.lambda_county.values()))
        }
    
    def normalize_loss(self, loss_value: float) -> Dict:
        """
        Normalize loss to make results comparable across different states
        
        Parameters:
        -----------
        loss_value : float
            Raw loss value
            
        Returns:
        --------
        dict of normalized metrics
        """
        n_stations = len(self.G.nodes())
        
        normalized = {
            # 1. Normalize by network size
            'loss_per_station': loss_value / n_stations if n_stations > 0 else 0,
            
            # 2. Normalize by population (per 100k people)
            'loss_per_100k_population': (loss_value * 1e5) / self.state_info['total_population'],
            
            # 3. Normalize by area (per 1000 km²)
            'loss_per_1000km2': (loss_value * 1000) / self.state_info['total_area_km2'],
            
            # 4. Composite normalization
            'composite_normalized': (loss_value * 1e5 / self.state_info['total_population']) * 
                                   (1000 / self.state_info['total_area_km2'])
        }
        
        return normalized
    
    def generate_report(self, mc_results: Dict) -> pd.DataFrame:
        """
        Generate simulation report
        
        Parameters:
        -----------
        mc_results : dict
            Monte Carlo simulation results
            
        Returns:
        --------
        pandas DataFrame with summary statistics
        """
        mean_loss = mc_results['mean_annual_loss']
        normalized = self.normalize_loss(mean_loss)
        
        report = {
            'State': self.state,
            'N_Stations': len(self.G.nodes()),
            'N_Counties': self.state_info['n_counties'],
            'Total_Population': self.state_info['total_population'],
            'Area_km2': self.state_info['total_area_km2'],
            'Baseline_Efficiency': self.baseline_efficiency,
            'Mean_Annual_Loss': mean_loss,
            'Std_Annual_Loss': mc_results['std_annual_loss'],
            'Median_Annual_Loss': mc_results['median_annual_loss'],
            'VaR_95': mc_results['VaR_95'],
            'CVaR_95': mc_results['CVaR_95'],
            'Mean_N_Events_Per_Year': mc_results['mean_n_events'],
            'Loss_Per_Station': normalized['loss_per_station'],
            'Loss_Per_100k_Pop': normalized['loss_per_100k_population'],
            'Composite_Normalized_Loss': normalized['composite_normalized']
        }
        
        return pd.DataFrame([report])


def run_multi_state_comparison(state_configs: Dict[str, str],
                               n_simulations: int = 1000,
                               recovery_model: str = 'constant') -> pd.DataFrame:
    """
    Compare power outage resilience across multiple states
    
    Parameters:
    -----------
    state_configs : dict
        {state_name: pickle_file_path} mapping
    n_simulations : int
        Number of simulations per state
    recovery_model : str
        Recovery model
        
    Returns:
    --------
    comparison DataFrame
    """
    all_reports = []
    
    for state_name, data_path in state_configs.items():
        print(f"\n{'='*60}")
        print(f"Processing {state_name}")
        print(f"{'='*60}")
        
        # Initialize simulator
        simulator = RealisticOutageSimulator(state_name, data_path)
        
        # Run Monte Carlo simulation
        mc_results = simulator.monte_carlo_simulation(
            n_simulations=n_simulations,
            recovery_model=recovery_model
        )
        
        # Generate report
        report = simulator.generate_report(mc_results)
        all_reports.append(report)
        
        # Print summary
        print(f"\nResults for {state_name}:")
        print(f"  Mean Annual Loss: {mc_results['mean_annual_loss']:.4f}")
        print(f"  95% VaR: {mc_results['VaR_95']:.4f}")
        print(f"  Expected Events/Year: {mc_results['mean_n_events']:.1f}")
    
    # Merge all reports
    comparison_df = pd.concat(all_reports, ignore_index=True)
    
    # Add ranking
    comparison_df['Vulnerability_Rank'] = comparison_df['Composite_Normalized_Loss'].rank(
        ascending=False
    )
    
    return comparison_df


if __name__ == "__main__":
    print("Realistic Outage Simulator Module")
    print("="*60)
    print("This module simulates power outage impacts on EV charging networks.")
    print("\nUsage:")
    print("  1. Prepare your data using data_preparation.py")
    print("  2. Run: simulator = RealisticOutageSimulator('State', 'data.pkl')")
    print("  3. Run: results = simulator.monte_carlo_simulation(n_simulations=1000)")
    print("  4. Generate report: report = simulator.generate_report(results)")
    print("="*60)
