"""
Batch State Data Preparation Tool
批量为所有州的网络添加lambda、停电、人口统计数据

为每个州的network pickle文件添加该州的:
- Bayesian lambda (停电频率)
- Historical outages (历史停电记录)
- County demographics (县人口统计)

输出格式适用于 RealisticOutageSimulator
"""

import pickle
import pandas as pd
import networkx as nx
import os
import glob
from typing import Dict, Any, Set, Optional
from pathlib import Path


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


# 州名到FIPS代码的映射
STATE_NAME_TO_FIPS = {
    'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05',
    'California': '06', 'Colorado': '08', 'Connecticut': '09', 'Delaware': '10',
    'District of Columbia': '11', 'Florida': '12', 'Georgia': '13',
    'Hawaii': '15', 'Idaho': '16', 'Illinois': '17', 'Indiana': '18',
    'Iowa': '19', 'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22',
    'Maine': '23', 'Maryland': '24', 'Massachusetts': '25', 'Michigan': '26',
    'Minnesota': '27', 'Mississippi': '28', 'Missouri': '29', 'Montana': '30',
    'Nebraska': '31', 'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34',
    'New Mexico': '35', 'New York': '36', 'North Carolina': '37', 'North Dakota': '38',
    'Ohio': '39', 'Oklahoma': '40', 'Oregon': '41', 'Pennsylvania': '42',
    'Rhode Island': '44', 'South Carolina': '45', 'South Dakota': '46',
    'Tennessee': '47', 'Texas': '48', 'Utah': '49', 'Vermont': '50',
    'Virginia': '51', 'Washington': '53', 'West Virginia': '54',
    'Wisconsin': '55', 'Wyoming': '56'
}

# 州缩写到FIPS代码的映射
STATE_ABBR_TO_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08',
    'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12', 'GA': '13', 'HI': '15',
    'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21',
    'LA': '22', 'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27',
    'MS': '28', 'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
    'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46',
    'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53',
    'WV': '54', 'WI': '55', 'WY': '56'
}

# FIPS到州名的映射
FIPS_TO_STATE = {v: k for k, v in STATE_NAME_TO_FIPS.items()}


def extract_state_from_filename(filename: str) -> Optional[str]:
    """
    从文件名中提取州FIPS代码
    
    支持的格式:
    - AL_graph.pkl -> '01'
    - California_network.pkl -> '06'
    - state_06.pkl -> '06'
    """
    basename = os.path.basename(filename)
    name_part = basename.split('.')[0]  # 去掉扩展名
    
    # 尝试匹配州缩写 (AL, CA, etc.)
    parts = name_part.split('_')
    for part in parts:
        part_upper = part.upper()
        if part_upper in STATE_ABBR_TO_FIPS:
            return STATE_ABBR_TO_FIPS[part_upper]
    
    # 尝试匹配州全名
    for state_name, fips in STATE_NAME_TO_FIPS.items():
        if state_name.lower() in name_part.lower():
            return fips
    
    # 尝试直接提取FIPS代码
    for part in parts:
        if part.isdigit() and len(part) == 2:
            if part in FIPS_TO_STATE:
                return part
    
    return None


def load_shared_data(
    lambda_csv: str,
    outages_csv: str,
    demographics_csv: str
) -> tuple:
    """
    加载所有共享数据（一次性加载，避免重复读取）
    
    Returns:
    --------
    tuple: (lambda_df, outages_df, demographics_df)
    """
    print("\n" + "="*70)
    print("Loading shared data files...")
    print("="*70)
    
    # Load lambda data
    print(f"\n1. Loading Bayesian lambda: {lambda_csv}")
    lambda_df = pd.read_csv(lambda_csv)
    if 'fips' in lambda_df.columns:
        lambda_df['fips'] = lambda_df['fips'].apply(normalize_fips)
    print(f"   ✓ Loaded {len(lambda_df)} counties")
    
    # Load outages data
    print(f"\n2. Loading historical outages: {outages_csv}")
    outages_df = pd.read_csv(outages_csv)
    if 'county' in outages_df.columns:
        outages_df['county'] = outages_df['county'].apply(normalize_fips)
    print(f"   ✓ Loaded {len(outages_df)} outage records")
    
    # Load demographics data
    print(f"\n3. Loading county demographics: {demographics_csv}")
    demographics_df = pd.read_csv(demographics_csv, encoding='latin-1')
    
    # 生成FIPS如果不存在
    if 'fips' not in demographics_df.columns:
        if 'STATE' in demographics_df.columns and 'COUNTY' in demographics_df.columns:
            demographics_df['fips'] = (
                demographics_df['STATE'].astype(str).str.zfill(2) + 
                demographics_df['COUNTY'].astype(str).str.zfill(3)
            )
    
    demographics_df['fips'] = demographics_df['fips'].apply(normalize_fips)
    print(f"   ✓ Loaded {len(demographics_df)} counties")
    
    return lambda_df, outages_df, demographics_df


def process_single_state(
    network_file: str,
    lambda_df: pd.DataFrame,
    outages_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    output_dir: str,
    state_fips: Optional[str] = None
) -> Dict:
    """
    处理单个州的网络数据
    
    Parameters:
    -----------
    network_file : str
        州网络pickle文件路径
    lambda_df : pd.DataFrame
        Lambda数据DataFrame (所有州)
    outages_df : pd.DataFrame
        停电数据DataFrame (所有州)
    demographics_df : pd.DataFrame
        人口统计DataFrame (所有州)
    output_dir : str
        输出目录
    state_fips : Optional[str]
        州FIPS代码，如果None则自动检测
        
    Returns:
    --------
    Dict: 处理后的state_data，如果失败返回None
    """
    
    filename = os.path.basename(network_file)
    print(f"\n{'='*70}")
    print(f"Processing: {filename}")
    print(f"{'='*70}")
    
    # 1. 确定州FIPS
    if state_fips is None:
        state_fips = extract_state_from_filename(filename)
    
    if state_fips is None:
        print(f"   ❌ Cannot determine state from filename: {filename}")
        print(f"      Please rename file to include state abbreviation (e.g., CA_graph.pkl)")
        return None
    
    state_name = FIPS_TO_STATE.get(state_fips, f"State {state_fips}")
    print(f"   State: {state_name} (FIPS: {state_fips})")
    
    # 2. 加载网络
    try:
        print(f"   → Loading network...")
        with open(network_file, 'rb') as f:
            network_data = pickle.load(f)
        
        # 提取图结构
        if isinstance(network_data, nx.Graph):
            G = network_data
        elif isinstance(network_data, dict):
            if 'network' in network_data:
                G = network_data['network']
            elif 'G' in network_data:
                G = network_data['G']
            elif 'graph' in network_data:
                G = network_data['graph']
            else:
                print(f"   ❌ Cannot find graph in pickle file")
                return None
        else:
            print(f"   ❌ Unknown pickle format")
            return None
        
        print(f"   ✓ Network: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        # 转换 lat/lon 到 location
        for node in G.nodes():
            if 'lat' in G.nodes[node] and 'lon' in G.nodes[node]:
                G.nodes[node]['location'] = (G.nodes[node]['lat'], G.nodes[node]['lon'])
        
    except Exception as e:
        print(f"   ❌ Failed to load network: {str(e)}")
        return None
    
    # 3. 过滤Lambda数据
    print(f"   → Filtering lambda data for state {state_fips}...")
    state_lambda_df = lambda_df[lambda_df['fips'].str.startswith(state_fips)]
    county_lambdas = dict(zip(state_lambda_df['fips'], state_lambda_df['lambda_mean']))
    print(f"   ✓ Lambda: {len(county_lambdas)} counties")
    
    # 4. 过滤停电数据
    print(f"   → Filtering outage data for state {state_fips}...")
    state_outages_df = outages_df[outages_df['county'].str.startswith(state_fips)]
    
    historical_outages = {}
    for county in state_outages_df['county'].unique():
        county_outages = state_outages_df[state_outages_df['county'] == county]
        historical_outages[county] = [
            {
                'duration': pd.to_timedelta(row['duration']).total_seconds() / 3600,
                'affected_customers': int(row['mean_customers'])
            }
            for _, row in county_outages.iterrows()
        ]
    
    total_records = sum(len(v) for v in historical_outages.values())
    print(f"   ✓ Outages: {len(historical_outages)} counties, {total_records} records")
    
    # 5. 过滤人口统计数据
    print(f"   → Filtering demographics for state {state_fips}...")
    state_demo_df = demographics_df[demographics_df['fips'].str.startswith(state_fips)]
    
    county_demographics = {}
    skipped_missing_area = 0
    
    for _, row in state_demo_df.iterrows():
        # 检查必需字段
        if pd.isna(row['area_km2']) or pd.isna(row['centroid_lat']) or pd.isna(row['centroid_lon']):
            skipped_missing_area += 1
            continue
        
        county_demographics[row['fips']] = {
            'population': int(row['POPESTIMATE2022']),
            'area_km2': float(row['area_km2']),
            'centroid': (float(row['centroid_lat']), float(row['centroid_lon']))
        }
    
    print(f"   ✓ Demographics: {len(county_demographics)} counties")
    if skipped_missing_area > 0:
        print(f"   ⚠ Skipped {skipped_missing_area} counties with missing area/centroid data")
    
    # 6. 验证数据一致性
    lambda_counties = set(county_lambdas.keys())
    outage_counties = set(historical_outages.keys())
    demo_counties = set(county_demographics.keys())
    common = lambda_counties & outage_counties & demo_counties
    
    print(f"   → Data consistency: {len(common)} counties in all datasets")
    
    if len(common) == 0:
        print(f"   ⚠ WARNING: No common counties across all datasets!")
    
    # 7. 组装数据
    state_data = {
        'network': G,
        'county_lambdas': county_lambdas,
        'historical_outages': historical_outages,
        'county_demographics': county_demographics,
        'state_fips': state_fips,
        'state_name': state_name
    }
    
    # 8. 保存
    output_filename = f"{state_name.replace(' ', '_')}_complete.pkl"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"   → Saving to: {output_filename}")
    with open(output_path, 'wb') as f:
        pickle.dump(state_data, f)
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"   ✓ Saved ({file_size:.2f} MB)")
    
    # 9. 打印摘要
    total_pop = sum(d['population'] for d in county_demographics.values())
    print(f"\n   Summary:")
    print(f"   - Charging stations: {len(G.nodes())}")
    print(f"   - Counties: {len(county_lambdas)}")
    print(f"   - Population: {total_pop:,}")
    print(f"   - Expected annual outages: {sum(county_lambdas.values()):.1f}")
    
    return state_data


def batch_process_states(
    network_dir: str,
    lambda_csv: str,
    outages_csv: str,
    demographics_csv: str,
    output_dir: str,
    file_pattern: str = "*_graph.pkl",
    exclude_states: list = None
):
    """
    批量处理所有州的网络
    
    Parameters:
    -----------
    network_dir : str
        包含所有州网络pickle文件的目录
    lambda_csv : str
        Lambda CSV文件路径
    outages_csv : str
        停电CSV文件路径
    demographics_csv : str
        人口统计CSV文件路径
    output_dir : str
        输出目录
    file_pattern : str
        网络文件匹配模式 (例如: "*_graph.pkl", "*.pkl")
    exclude_states : list
        要排除的州列表 (FIPS代码或缩写)，例如: ['09', 'CT', 'Connecticut']
    """
    
    print("="*70)
    print("BATCH STATE DATA PREPARATION")
    print("="*70)
    print(f"\nInput directory: {network_dir}")
    print(f"Output directory: {output_dir}")
    print(f"File pattern: {file_pattern}")
    
    # 标准化排除列表
    exclude_fips = set()
    if exclude_states:
        print(f"\nExcluding states:")
        for state in exclude_states:
            # 转换为FIPS代码
            if state.upper() in STATE_ABBR_TO_FIPS:
                fips = STATE_ABBR_TO_FIPS[state.upper()]
                exclude_fips.add(fips)
                state_name = FIPS_TO_STATE.get(fips, state)
                print(f"  - {state_name} (FIPS: {fips})")
            elif state in STATE_NAME_TO_FIPS.values():
                exclude_fips.add(state)
                state_name = FIPS_TO_STATE.get(state, state)
                print(f"  - {state_name} (FIPS: {state})")
            else:
                # 尝试匹配州名
                for name, fips in STATE_NAME_TO_FIPS.items():
                    if name.lower() == state.lower():
                        exclude_fips.add(fips)
                        print(f"  - {name} (FIPS: {fips})")
                        break
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory ready")
    
    # 查找所有网络文件
    search_pattern = os.path.join(network_dir, file_pattern)
    network_files = glob.glob(search_pattern)
    
    if not network_files:
        print(f"\n❌ No network files found matching pattern: {search_pattern}")
        print(f"   Please check the directory and file pattern")
        return
    
    # 过滤掉排除的州
    if exclude_fips:
        filtered_files = []
        skipped_files = []
        
        for f in network_files:
            state_fips = extract_state_from_filename(f)
            if state_fips in exclude_fips:
                skipped_files.append(os.path.basename(f))
            else:
                filtered_files.append(f)
        
        if skipped_files:
            print(f"\nSkipped {len(skipped_files)} excluded state files:")
            for f in skipped_files:
                print(f"  - {f}")
        
        network_files = filtered_files
    
    print(f"\nFound {len(network_files)} network files to process:")
    for f in network_files:
        print(f"  - {os.path.basename(f)}")
    
    if not network_files:
        print("\n❌ No files left to process after exclusions")
        return
    
    # 加载共享数据（一次性）
    lambda_df, outages_df, demographics_df = load_shared_data(
        lambda_csv, outages_csv, demographics_csv
    )
    
    # 处理每个州
    print("\n" + "="*70)
    print("PROCESSING STATES")
    print("="*70)
    
    results = {
        'success': [],
        'failed': []
    }
    
    for i, network_file in enumerate(network_files, 1):
        print(f"\n[{i}/{len(network_files)}]", end=" ")
        
        try:
            result = process_single_state(
                network_file=network_file,
                lambda_df=lambda_df,
                outages_df=outages_df,
                demographics_df=demographics_df,
                output_dir=output_dir
            )
            
            if result:
                results['success'].append(os.path.basename(network_file))
            else:
                results['failed'].append(os.path.basename(network_file))
        
        except Exception as e:
            print(f"   ❌ Unexpected error: {str(e)}")
            results['failed'].append(os.path.basename(network_file))
    
    # 最终报告
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    
    print(f"\n✓ Successfully processed: {len(results['success'])} states")
    for state in results['success']:
        print(f"  ✓ {state}")
    
    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])} states")
        for state in results['failed']:
            print(f"  ❌ {state}")
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   All complete state data files are saved there")
    print(f"\nYou can now use these files for simulation:")
    print(f"  from realistic_outage_simulator import RealisticOutageSimulator")
    print(f"  simulator = RealisticOutageSimulator('California', 'California_complete.pkl')")
    print(f"  results = simulator.monte_carlo_simulation(n_simulations=100)")


if __name__ == "__main__":
    import sys
    
    print("Batch State Data Preparation Tool")
    print("="*70)
    
    if len(sys.argv) >= 5:
        # 命令行模式
        network_dir = sys.argv[1]
        lambda_csv = sys.argv[2]
        outages_csv = sys.argv[3]
        demographics_csv = sys.argv[4]
        output_dir = sys.argv[5] if len(sys.argv) > 5 else os.path.join(network_dir, 'complete_data')
        file_pattern = sys.argv[6] if len(sys.argv) > 6 else "*_graph.pkl"
        exclude_states = sys.argv[7].split(',') if len(sys.argv) > 7 else None
    else:
        # 交互模式
        print("\nPlease provide paths:")
        network_dir = input("  Network directory (contains all state pkl files): ").strip()
        lambda_csv = input("  Bayesian lambda CSV: ").strip()
        outages_csv = input("  Historical outages CSV: ").strip()
        demographics_csv = input("  County demographics CSV: ").strip()
        
        default_output = os.path.join(network_dir, 'complete_data')
        output_dir = input(f"  Output directory [{default_output}]: ").strip() or default_output
        
        file_pattern = input("  File pattern [*_graph.pkl]: ").strip() or "*_graph.pkl"
        
        print("\nExclude states (optional):")
        print("  Enter state codes separated by commas (e.g., CT,RI,DE)")
        print("  Or leave empty to process all states")
        exclude_input = input("  Exclude: ").strip()
        exclude_states = [s.strip() for s in exclude_input.split(',')] if exclude_input else None
    
    batch_process_states(
        network_dir=network_dir,
        lambda_csv=lambda_csv,
        outages_csv=outages_csv,
        demographics_csv=demographics_csv,
        output_dir=output_dir,
        file_pattern=file_pattern,
        exclude_states=exclude_states
    )
