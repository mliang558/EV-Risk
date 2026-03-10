"""
合并County Shapefile和人口数据
将面积、中心坐标从shapefile合并到人口统计CSV
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os

def merge_shapefile_with_demographics(
    shapefile_path: str,
    demographics_csv_path: str,
    output_csv_path: str
):
    """
    合并shapefile和人口数据
    
    Parameters:
    -----------
    shapefile_path : str
        County shapefile路径 (e.g., 'tl_2021_us_county.shp')
    demographics_csv_path : str
        人口统计CSV路径 (e.g., 'co-est2024-alldata.csv')
    output_csv_path : str
        输出CSV路径
    """
    
    print("="*70)
    print("合并Shapefile和人口数据")
    print("="*70)
    
    # 0. 设置自动修复环境变量
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    
    # 1. 读取shapefile
    print(f"\n1. 读取shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    print(f"   ✓ 加载了 {len(gdf)} 个县")
    print(f"   原始CRS: {gdf.crs}")
    
    # 检查shapefile的关键列
    print(f"   列名: {list(gdf.columns)}")
    
    # 通常shapefile有STATEFP和COUNTYFP列
    # 生成FIPS（5位：2位州码 + 3位县码）
    if 'STATEFP' in gdf.columns and 'COUNTYFP' in gdf.columns:
        gdf['fips'] = gdf['STATEFP'].astype(str).str.zfill(2) + \
                      gdf['COUNTYFP'].astype(str).str.zfill(3)
    elif 'GEOID' in gdf.columns:
        gdf['fips'] = gdf['GEOID'].astype(str).str.zfill(5)
    else:
        print("   ⚠ 警告: 未找到FIPS相关列，请检查shapefile")
        return
    
    print(f"   ✓ 生成FIPS码")
    
    # 2. 转换到等面积投影计算面积
    print(f"\n2. 计算面积（转换到EPSG:5070 Albers等面积投影）")
    gdf_albers = gdf.to_crs('EPSG:5070')
    gdf['area_m2'] = gdf_albers.geometry.area
    gdf['area_km2'] = gdf['area_m2'] / 1e6
    print(f"   ✓ 面积计算完成")
    print(f"   面积范围: {gdf['area_km2'].min():.1f} - {gdf['area_km2'].max():.1f} km²")
    
    # 3. 计算中心点（WGS84坐标）
    print(f"\n3. 计算县中心坐标")
    gdf_albers['centroid_geom'] = gdf_albers.geometry.centroid
    centroids_wgs84 = gdf_albers['centroid_geom'].to_crs('EPSG:4326')
    gdf['centroid_lat'] = centroids_wgs84.y
    gdf['centroid_lon'] = centroids_wgs84.x
    print(f"   ✓ 中心坐标计算完成")
    
    # 4. 读取人口数据
    print(f"\n4. 读取人口数据: {demographics_csv_path}")
    
    # 尝试不同编码
    for encoding in ['latin-1', 'ISO-8859-1', 'cp1252', 'utf-8']:
        try:
            demographics_df = pd.read_csv(demographics_csv_path, encoding=encoding)
            print(f"   ✓ 使用编码 {encoding} 成功读取")
            break
        except UnicodeDecodeError:
            continue
    else:
        print("   ❌ 无法读取CSV，尝试所有编码都失败")
        return
    
    print(f"   ✓ 加载了 {len(demographics_df)} 行数据")
    print(f"   列名: {list(demographics_df.columns)[:10]}...")  # 显示前10列
    
    # 生成FIPS（如果需要）
    if 'fips' not in demographics_df.columns:
        if 'STATE' in demographics_df.columns and 'COUNTY' in demographics_df.columns:
            demographics_df['fips'] = \
                demographics_df['STATE'].astype(str).str.zfill(2) + \
                demographics_df['COUNTY'].astype(str).str.zfill(3)
            print(f"   ✓ 从STATE和COUNTY生成FIPS")
        else:
            print("   ⚠ 警告: demographics数据缺少FIPS相关列")
            return
    
    # 只保留county级别数据（SUMLEV=050）
    if 'SUMLEV' in demographics_df.columns:
        demographics_df = demographics_df[demographics_df['SUMLEV'] == 50].copy()
        print(f"   ✓ 筛选county级别数据: {len(demographics_df)} 行")
    
    # 5. 合并数据
    print(f"\n5. 合并数据（基于FIPS）")
    
    # 从shapefile提取需要的列
    geo_data = gdf[['fips', 'area_km2', 'centroid_lat', 'centroid_lon']].copy()
    
    # 合并
    merged_df = demographics_df.merge(
        geo_data,
        on='fips',
        how='left'
    )
    
    print(f"   ✓ 合并完成: {len(merged_df)} 行")
    
    # 检查缺失值
    missing_area = merged_df['area_km2'].isna().sum()
    missing_lat = merged_df['centroid_lat'].isna().sum()
    
    if missing_area > 0 or missing_lat > 0:
        print(f"   ⚠ 缺失值:")
        print(f"     area_km2: {missing_area} 个")
        print(f"     centroid: {missing_lat} 个")
        print(f"   → 这些county在shapefile中未找到匹配")
    else:
        print(f"   ✓ 所有county都成功匹配")
    
    # 6. 保存结果
    print(f"\n6. 保存到: {output_csv_path}")
    
    # 如果输出路径是目录，创建默认文件名
    if os.path.isdir(output_csv_path):
        output_csv_path = os.path.join(output_csv_path, 'merged_county_data.csv')
        print(f"   → 输出路径是目录，使用文件名: merged_county_data.csv")
        print(f"   → 完整路径: {output_csv_path}")
    
    merged_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    file_size = os.path.getsize(output_csv_path) / 1024 / 1024
    print(f"   ✓ 保存成功 ({file_size:.1f} MB)")
    
    # 7. 验证结果
    print(f"\n7. 验证结果")
    print(f"   总行数: {len(merged_df)}")
    print(f"   包含列: fips, POPESTIMATE2022, area_km2, centroid_lat, centroid_lon")
    
    # 显示示例
    print(f"\n示例数据（前3行）:")
    sample_cols = ['fips', 'STNAME', 'CTYNAME', 'POPESTIMATE2022', 'area_km2', 
                   'centroid_lat', 'centroid_lon']
    available_cols = [c for c in sample_cols if c in merged_df.columns]
    print(merged_df[available_cols].head(3).to_string(index=False))
    
    print("\n" + "="*70)
    print("✓ 合并完成！")
    print(f"✓ 输出文件: {output_csv_path}")
    print("="*70)
    
    return merged_df


if __name__ == "__main__":
    import sys
    
    print("County Shapefile + Demographics 合并工具")
    print("="*70)
    
    # 交互式输入
    if len(sys.argv) == 1:
        print("\n请提供文件路径:")
        shapefile_path = input("  Shapefile路径 (.shp): ").strip()
        demographics_csv = input("  人口CSV路径: ").strip()
        output_csv = input("  输出CSV路径 (文件或目录): ").strip()
    elif len(sys.argv) == 4:
        shapefile_path = sys.argv[1]
        demographics_csv = sys.argv[2]
        output_csv = sys.argv[3]
    else:
        print("\n用法:")
        print("  交互式: python merge_county_data.py")
        print("  命令行: python merge_county_data.py <shapefile> <demographics_csv> <output_csv>")
        print("\n示例:")
        print("  python merge_county_data.py tl_2021_us_county.shp co-est2024-alldata.csv merged_output.csv")
        sys.exit(1)
    
    # 运行合并
    merge_shapefile_with_demographics(
        shapefile_path=shapefile_path,
        demographics_csv_path=demographics_csv,
        output_csv_path=output_csv
    )
