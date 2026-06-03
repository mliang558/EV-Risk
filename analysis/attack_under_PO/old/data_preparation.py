"""
Data Preparation Tool
将现有数据转换成RealisticOutageSimulator需要的格式

This script helps convert your existing data into the format required by
the RealisticOutageSimulator.
"""

import pickle
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, Any


def convert_data_to_required_format(
    network_pickle_path: str,
    bayesian_lambda_csv: str,
    historical_outages_csv: str,
    county_demographics_csv: str,
    output_path: str
) -> Dict:
    """
    Convert existing data to required format
    
    Parameters:
    -----------
    network_pickle_path : str
        Path to your charging station network pickle file
    bayesian_lambda_csv : str
        CSV file with county and lambda values
        Columns: ['county', 'lambda_annual']
    historical_outages_csv : str
        CSV file with historical outage records
        Columns: ['county', 'duration_hours', 'affected_customers']
    county_demographics_csv : str
        CSV file with county demographics
        Columns: ['county', 'population', 'area_km2', 'centroid_lat', 'centroid_lon']
    output_path : str
        Output pickle file path
        
    Returns:
    --------
    Prepared state data dictionary
    """
    
    print("="*60)
    print("Data Conversion Tool")
    print("="*60)
    print("\nLoading your existing data...")
    
    # 1. Load network
    print(f"\n1. Loading network from: {network_pickle_path}")
    with open(network_pickle_path, 'rb') as f:
        existing_data = pickle.load(f)
    
    # Assume network is in NetworkX format
    # Adjust key name if needed ('network', 'G', 'graph', etc.)
    if 'network' in existing_data:
        G = existing_data['network']
    elif 'G' in existing_data:
        G = existing_data['G']
    elif 'graph' in existing_data:
        G = existing_data['graph']
    else:
        # If pickle file contains just the graph
        G = existing_data
    
    print(f"   ✓ Network loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")

    # 转换 lat/lon 到 location
    for node in G.nodes():
        if 'lat' in G.nodes[node] and 'lon' in G.nodes[node]:
            G.nodes[node]['location'] = (G.nodes[node]['lat'], G.nodes[node]['lon'])
    
    # Validate network has location data
    _validate_network_locations(G)
    
    # 2. Load Bayesian lambda
    print(f"\n2. Loading Bayesian lambda from: {bayesian_lambda_csv}")
    lambda_df = pd.read_csv(bayesian_lambda_csv)
    county_lambdas = dict(zip(lambda_df['fips'], lambda_df['lambda_mean']))
    print(f"   ✓ Lambda data loaded for {len(county_lambdas)} counties")
    print(f"   ✓ Lambda range: [{min(county_lambdas.values()):.2f}, {max(county_lambdas.values()):.2f}]")
    
    # 3. Load historical outage data
    print(f"\n3. Loading historical outages from: {historical_outages_csv}")
    outages_df = pd.read_csv(historical_outages_csv)
    historical_outages = {}
    
    for county in outages_df['county'].unique():
        county_outages = outages_df[outages_df['county'] == county]
        historical_outages[county] = [
            {
                'duration': pd.to_timedelta(row['duration']).total_seconds() / 3600,
                'affected_customers': int(row['mean_customers'])
            }
            for _, row in county_outages.iterrows()
        ]
    
    total_outage_records = sum(len(v) for v in historical_outages.values())
    print(f"   ✓ Historical outage data loaded for {len(historical_outages)} counties")
    print(f"   ✓ Total outage records: {total_outage_records}")
    
    # 4. Load county demographics
    print(f"\n4. Loading county demographics from: {county_demographics_csv}")
    demographics_df = pd.read_csv(county_demographics_csv, encoding='latin-1')
    county_demographics = {}
    demographics_df['fips'] = demographics_df['STATE'].astype(str).str.zfill(2) + demographics_df['COUNTY'].astype(str).str.zfill(3)

    for _, row in demographics_df.iterrows():
        county_demographics[row['fips']] = {
            'population': int(row['POPESTIMATE2022']),
            'area_km2': float(row['area_km2']),
            'centroid': (float(row['centroid_lat']), float(row['centroid_lon']))
        }
    
    print(f"   ✓ Demographics loaded for {len(county_demographics)} counties")
    
    # 5. Validate data consistency
    print("\n5. Validating data consistency...")
    _validate_data_consistency(county_lambdas, historical_outages, county_demographics)
    def normalize_fips(fips):
    """统一FIPS格式为5位字符串"""
    if pd.isna(fips):
        return None
    if isinstance(fips, int):
        return str(fips).zfill(5)
    elif isinstance(fips, str):
        return fips.strip().zfill(5)
    else:
        return str(fips).zfill(5)

    # 在验证之前先标准化FIPS
    print("\n6. Normalizing FIPS codes...")
    if 'county_lambda' in data:
        data['county_lambda'] = {normalize_fips(k): v for k, v in data['county_lambda'].items()}
        print(f"   ✓ Lambda: {len(data['county_lambda'])} counties")

    if 'county_outages' in data:
        data['county_outages'] = {normalize_fips(k): v for k, v in data['county_outages'].items()}
        print(f"   ✓ Outages: {len(data['county_outages'])} counties")

    if 'county_demographics' in data:
        data['county_demographics'] = {normalize_fips(k): v for k, v in data['county_demographics'].items()}
        print(f"   ✓ Demographics: {len(data['county_demographics'])} counties")

    # 然后再验证
    print("\n7. Validating data consistency...")

    # 6. Assemble final data structure
    print("\n6. Assembling final data structure...")
    state_data = {
        'network': G,
        'county_lambdas': county_lambdas,
        'historical_outages': historical_outages,
        'county_demographics': county_demographics
    }
    
    # 7. Save
    print(f"\n7. Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(state_data, f)
    
    print("\n" + "="*60)
    print("✓ Data conversion complete!")
    print("="*60)
    
    # Print summary
    _print_data_summary(state_data)
    
    return state_data


def _validate_network_locations(G: nx.Graph):
    """Validate that network nodes have location attributes"""
    
    missing_locations = []
    invalid_locations = []
    
    for node in G.nodes():
        loc = G.nodes[node].get('location')
        
        if loc is None:
            missing_locations.append(node)
        elif not isinstance(loc, (tuple, list)) or len(loc) != 2:
            invalid_locations.append(node)
    
    if missing_locations:
        print(f"   ⚠ Warning: {len(missing_locations)} nodes missing 'location' attribute")
        print(f"      Example nodes: {missing_locations[:5]}")
        print("      → You may need to add location data to these nodes")
    
    if invalid_locations:
        print(f"   ⚠ Warning: {len(invalid_locations)} nodes have invalid location format")
        print(f"      Example nodes: {invalid_locations[:5]}")
        print("      → Locations should be tuples: (latitude, longitude)")
    
    if not missing_locations and not invalid_locations:
        print("   ✓ All nodes have valid location data")


def _validate_data_consistency(county_lambdas: Dict, 
                               historical_outages: Dict,
                               county_demographics: Dict):
    """Validate consistency across data sources"""
    
    lambda_counties = set(county_lambdas.keys())
    outage_counties = set(historical_outages.keys())
    demo_counties = set(county_demographics.keys())
    
    # Find mismatches
    missing_in_outages = lambda_counties - outage_counties
    missing_in_demographics = lambda_counties - demo_counties
    
    if missing_in_outages:
        print(f"   ⚠ Warning: {len(missing_in_outages)} counties in lambda but not in outage data")
        print(f"      Example: {list(missing_in_outages)[:3]}")
    
    if missing_in_demographics:
        print(f"   ⚠ Warning: {len(missing_in_demographics)} counties in lambda but not in demographics")
        print(f"      Example: {list(missing_in_demographics)[:3]}")
    
    if not missing_in_outages and not missing_in_demographics:
        print("   ✓ All data sources are consistent")


def _print_data_summary(state_data: Dict):
    """Print summary of prepared data"""
    
    G = state_data['network']
    lambdas = state_data['county_lambdas']
    outages = state_data['historical_outages']
    demographics = state_data['county_demographics']
    
    total_pop = sum(d['population'] for d in demographics.values())
    total_area = sum(d['area_km2'] for d in demographics.values())
    total_outage_records = sum(len(v) for v in outages.values())
    
    print("\nData Summary:")
    print(f"  Network:")
    print(f"    - Nodes (charging stations): {len(G.nodes())}")
    print(f"    - Edges: {len(G.edges())}")
    print(f"  Counties: {len(lambdas)}")
    print(f"  Total Population: {total_pop:,}")
    print(f"  Total Area: {total_area:,.1f} km²")
    print(f"  Total Outage Records: {total_outage_records}")
    print(f"  Expected Annual Outages: {sum(lambdas.values()):.1f}")


def create_sample_data(output_path: str = 'sample_state_data.pkl'):
    """
    Create sample data for testing
    
    This generates synthetic data that demonstrates the required format.
    Replace with your actual data.
    """
    
    print("Creating sample data for demonstration...")
    
    # Create sample network
    G = nx.Graph()
    
    # Sample charging stations in California Bay Area
    stations = [
        {'id': 0, 'location': (37.7749, -122.4194), 'county': 'San Francisco', 'name': 'SF Downtown'},
        {'id': 1, 'location': (37.8044, -122.2712), 'county': 'Alameda', 'name': 'Oakland'},
        {'id': 2, 'location': (37.3382, -121.8863), 'county': 'Santa Clara', 'name': 'San Jose'},
        {'id': 3, 'location': (37.5485, -121.9886), 'county': 'Alameda', 'name': 'Fremont'},
        {'id': 4, 'location': (37.9577, -122.3477), 'county': 'Contra Costa', 'name': 'Richmond'},
    ]
    
    for station in stations:
        G.add_node(station['id'], 
                  location=station['location'],
                  county=station['county'],
                  name=station['name'])
    
    # Add edges
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (2, 3)])
    
    # Sample county demographics
    county_demographics = {
        'San Francisco': {
            'population': 873965,
            'area_km2': 121.4,
            'centroid': (37.7749, -122.4194)
        },
        'Alameda': {
            'population': 1671329,
            'area_km2': 1910,
            'centroid': (37.6017, -121.7195)
        },
        'Santa Clara': {
            'population': 1936259,
            'area_km2': 3343,
            'centroid': (37.3541, -121.9552)
        },
        'Contra Costa': {
            'population': 1153526,
            'area_km2': 1857,
            'centroid': (37.9161, -121.9510)
        }
    }
    
    # Sample Bayesian lambda (events per year)
    county_lambdas = {
        'San Francisco': 2.5,
        'Alameda': 3.2,
        'Santa Clara': 1.8,
        'Contra Costa': 2.1
    }
    
    # Sample historical outages
    historical_outages = {
        'San Francisco': [
            {'duration': 3.5, 'affected_customers': 5000},
            {'duration': 2.1, 'affected_customers': 2000},
            {'duration': 6.8, 'affected_customers': 12000},
            {'duration': 1.2, 'affected_customers': 800},
        ],
        'Alameda': [
            {'duration': 4.2, 'affected_customers': 8000},
            {'duration': 1.5, 'affected_customers': 1500},
            {'duration': 5.5, 'affected_customers': 10000},
        ],
        'Santa Clara': [
            {'duration': 2.8, 'affected_customers': 4000},
            {'duration': 3.9, 'affected_customers': 6000},
        ],
        'Contra Costa': [
            {'duration': 3.1, 'affected_customers': 5500},
            {'duration': 4.7, 'affected_customers': 7500},
        ]
    }
    
    # Assemble data
    state_data = {
        'network': G,
        'county_demographics': county_demographics,
        'county_lambdas': county_lambdas,
        'historical_outages': historical_outages
    }
    
    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(state_data, f)
    
    print(f"✓ Sample data created: {output_path}")
    print("\nYou can use this sample data to test the simulator:")
    print(f"  from realistic_outage_simulator import RealisticOutageSimulator")
    print(f"  simulator = RealisticOutageSimulator('California', '{output_path}')")
    print(f"  results = simulator.monte_carlo_simulation(n_simulations=100)")
    
    return state_data


if __name__ == "__main__":
    print("Data Preparation Tool")
    print("="*60)
    print("\nOptions:")
    print("  1. Create sample data for testing")
    print("  2. Convert your actual data")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        create_sample_data('sample_state_data.pkl')
        
    elif choice == "2":
        print("\nPlease provide paths to your data files:")
        network_path = input("  Network pickle path: ").strip()
        lambda_csv = input("  Bayesian lambda CSV path: ").strip()
        outages_csv = input("  Historical outages CSV path: ").strip()
        demographics_csv = input("  County demographics CSV path: ").strip()
        output_path = input("  Output pickle path: ").strip()
        
        convert_data_to_required_format(
            network_pickle_path=network_path,
            bayesian_lambda_csv=lambda_csv,
            historical_outages_csv=outages_csv,
            county_demographics_csv=demographics_csv,
            output_path=output_path
        )
    
    else:
        print("Invalid choice. Exiting.")
