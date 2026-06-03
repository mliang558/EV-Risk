"""
一键为所有州网络添加县FIPS信息
使用空间连接(point-in-polygon)匹配充电站和县边界

坐标系统会自动对齐，无需手动处理
"""

import pickle
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import os
import glob
from pathlib import Path


def add_county_batch(
    network_dir: str,
    county_shapefile: str,
    backup: bool = True
):
    """
    批量为所有州网络添加县信息
    
    Parameters:
    -----------
    network_dir : str
        网络文件目录
    county_shapefile : str
        县边界shapefile路径
    backup : bool
        是否备份原文件
    """
    
    print("="*70)
    print("一键添加县信息到所有州网络")
    print("="*70)
    
    # 1. 查找所有网络文件
    print(f"\n步骤1: 查找网络文件")
    print(f"目录: {network_dir}")
    
    search_patterns = ["*_graph.pkl", "*_network.pkl", "*.pkl"]
    network_files = []
    
    for pattern in search_patterns:
        files = glob.glob(os.path.join(network_dir, pattern))
        network_files.extend(files)
    
    # 去重
    network_files = list(set(network_files))
    
    if not network_files:
        print(f"❌ 未找到网络文件")
        return
    
    print(f"✓ 找到 {len(network_files)} 个文件:")
    for f in network_files[:10]:
        print(f"  - {os.path.basename(f)}")
    if len(network_files) > 10:
        print(f"  ... 还有 {len(network_files)-10} 个文件")
    
    # 2. 加载县边界shapefile（只加载一次）
    print(f"\n步骤2: 加载县边界shapefile")
    print(f"文件: {county_shapefile}")
    
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    
    try:
        counties_gdf = gpd.read_file(county_shapefile)
        print(f"✓ 加载了 {len(counties_gdf)} 个县")
        print(f"✓ 坐标系: {counties_gdf.crs}")
    except Exception as e:
        print(f"❌ 加载shapefile失败: {str(e)}")
        return
    
    # 确定FIPS列
    fips_col = None
    for col in ['GEOID', 'FIPS', 'fips', 'geoid']:
        if col in counties_gdf.columns:
            fips_col = col
            break
    
    if fips_col is None:
        if 'STATEFP' in counties_gdf.columns and 'COUNTYFP' in counties_gdf.columns:
            print("  → 从STATEFP和COUNTYFP生成FIPS")
            counties_gdf['FIPS'] = (
                counties_gdf['STATEFP'].astype(str).str.zfill(2) + 
                counties_gdf['COUNTYFP'].astype(str).str.zfill(3)
            )
            fips_col = 'FIPS'
        else:
            print(f"❌ 找不到FIPS列")
            print(f"可用列: {list(counties_gdf.columns)}")
            return
    
    print(f"✓ FIPS列: {fips_col}")
    
    # 只保留需要的列以提高性能
    counties_gdf = counties_gdf[[fips_col, 'geometry']].copy()
    
    # 确保FIPS是5位字符串
    counties_gdf[fips_col] = counties_gdf[fips_col].astype(str).str.zfill(5)
    
    # 3. 处理每个网络文件
    print(f"\n步骤3: 批量处理网络文件")
    print("="*70)
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, network_file in enumerate(network_files, 1):
        filename = os.path.basename(network_file)
        print(f"\n[{i}/{len(network_files)}] {filename}")
        
        try:
            # 加载网络
            with open(network_file, 'rb') as f:
                data = pickle.load(f)
            
            # 提取图
            if isinstance(data, nx.Graph):
                G = data
                save_as_dict = False
            elif isinstance(data, dict):
                save_as_dict = True
                if 'network' in data:
                    G = data['network']
                elif 'G' in data:
                    G = data['G']
                elif 'graph' in data:
                    G = data['graph']
                else:
                    print(f"  ❌ 找不到图结构")
                    failed_count += 1
                    continue
            else:
                print(f"  ❌ 未知格式")
                failed_count += 1
                continue
            
            print(f"  → 加载: {len(G.nodes())} 节点, {len(G.edges())} 边")
            
            # 检查是否已有县信息
            sample_node = list(G.nodes())[0]
            if 'county' in G.nodes[sample_node] or 'county_fips' in G.nodes[sample_node]:
                print(f"  ⊙ 已有县信息，跳过")
                skipped_count += 1
                continue
            
            # 提取节点坐标
            node_data = []
            coord_found = False
            
            for node in G.nodes():
                node_attrs = G.nodes[node]
                lat, lon = None, None
                
                # 尝试不同的坐标格式
                if 'location' in node_attrs:
                    loc = node_attrs['location']
                    if isinstance(loc, (tuple, list)) and len(loc) == 2:
                        lat, lon = loc
                        coord_found = True
                elif 'lat' in node_attrs and 'lon' in node_attrs:
                    lat = node_attrs['lat']
                    lon = node_attrs['lon']
                    coord_found = True
                elif 'latitude' in node_attrs and 'longitude' in node_attrs:
                    lat = node_attrs['latitude']
                    lon = node_attrs['longitude']
                    coord_found = True
                
                if lat is not None and lon is not None:
                    node_data.append({
                        'node_id': node,
                        'geometry': Point(lon, lat)
                    })
            
            if not coord_found:
                print(f"  ❌ 找不到坐标信息")
                print(f"     可用属性: {list(G.nodes[sample_node].keys())}")
                failed_count += 1
                continue
            
            if not node_data:
                print(f"  ❌ 没有有效坐标的节点")
                failed_count += 1
                continue
            
            # 创建GeoDataFrame
            nodes_gdf = gpd.GeoDataFrame(node_data, crs='EPSG:4326')
            print(f"  → {len(nodes_gdf)} 节点有坐标")
            
            # 坐标系对齐
            if counties_gdf.crs != nodes_gdf.crs:
                print(f"  → 对齐坐标系: {counties_gdf.crs} → {nodes_gdf.crs}")
                counties_aligned = counties_gdf.to_crs(nodes_gdf.crs)
            else:
                counties_aligned = counties_gdf
            
            # 空间连接
            print(f"  → 执行空间连接...")
            joined = gpd.sjoin(
                nodes_gdf,
                counties_aligned,
                how='left',
                predicate='within'
            )
            
            # 统计
            matched = joined[fips_col].notna().sum()
            match_rate = matched / len(joined) * 100
            
            print(f"  → 匹配: {matched}/{len(joined)} ({match_rate:.1f}%)")
            
            # 添加到节点
            added = 0
            for _, row in joined.iterrows():
                node_id = row['node_id']
                county_fips = row[fips_col]
                
                if pd.notna(county_fips):
                    G.nodes[node_id]['county'] = county_fips
                    G.nodes[node_id]['county_fips'] = county_fips
                    added += 1
            
            print(f"  → 添加县信息: {added} 节点")
            
            # 备份原文件
            if backup:
                backup_file = network_file + '.backup'
                if not os.path.exists(backup_file):
                    import shutil
                    shutil.copy2(network_file, backup_file)
                    print(f"  → 备份: {os.path.basename(backup_file)}")
            
            # 保存
            if save_as_dict:
                with open(network_file, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(network_file, 'wb') as f:
                    pickle.dump(G, f)
            
            print(f"  ✓ 保存成功")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            failed_count += 1
    
    # 最终报告
    print("\n" + "="*70)
    print("处理完成")
    print("="*70)
    print(f"\n✓ 成功: {success_count}")
    print(f"⊙ 跳过: {skipped_count} (已有县信息)")
    print(f"❌ 失败: {failed_count}")
    
    if success_count > 0:
        print(f"\n下一步:")
        print(f"运行批处理脚本添加lambda、停电、人口数据:")
        print(f"  python batch_state_preparation_final.py")


if __name__ == "__main__":
    print("="*70)
    print("一键为网络添加县信息")
    print("="*70)
    
    # 默认路径
    default_network_dir = r"C:\Users\maple\Desktop\EV_Project\codes\module 1\new\results_fixed\network_structures"
    default_shapefile = r"C:\Users\maple\Desktop\EV_Project\tl_2021_us_county\tl_2021_us_county.shp"
    
    print(f"\n请输入路径（直接回车使用默认值）:")
    
    network_dir = input(f"  网络目录 [{default_network_dir}]: ").strip()
    if not network_dir:
        network_dir = default_network_dir
    
    shapefile = input(f"  县边界shapefile [{default_shapefile}]: ").strip()
    if not shapefile:
        shapefile = default_shapefile
    
    backup_input = input(f"  备份原文件? (y/n) [y]: ").strip().lower()
    backup = backup_input != 'n'
    
    print(f"\n确认:")
    print(f"  网络目录: {network_dir}")
    print(f"  Shapefile: {shapefile}")
    print(f"  备份: {'是' if backup else '否'}")
    
    confirm = input(f"\n开始处理? (y/n): ").strip().lower()
    
    if confirm == 'y':
        add_county_batch(network_dir, shapefile, backup)
    else:
        print("取消")
