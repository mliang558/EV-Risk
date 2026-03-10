#!/usr/bin/env python3
"""
完整仿真示例 - 频率模型 + 严重性采样器
==========================================

假设你已经有：
1. lambda[county] - 停电频率（从你的模型）
2. full_pop_2022.csv - 历史停电数据

这个脚本展示如何结合它们进行完整仿真
"""

import pandas as pd
import numpy as np
from outage_severity_sampler import OutageSeveritySampler, OutageSimulator


def load_frequency_model(frequency_file=None):
    """
    加载你的频率模型
    
    如果你的频率模型是从之前的分析得到的，可以：
    1. 从CSV加载
    2. 从贝叶斯模型的后验提取
    3. 直接计算历史频率
    """
    if frequency_file is not None:
        # 选项1: 从你保存的文件加载
        df_lambda = pd.read_csv(frequency_file)
        lambda_dict = dict(zip(df_lambda['fips'], df_lambda['lambda']))
        
    else:
        # 选项2: 从历史数据直接计算
        print("从历史数据计算频率...")
        df = pd.read_csv('full_pop_2022.csv')
        df['fips'] = df['fips'].astype(str).str.zfill(5)
        
        # 假设数据是2022一年的
        n_years = 1
        
        # 计算每个县的停电次数
        county_counts = df.groupby('fips').size()
        
        # lambda = 次数 / 年数
        lambda_dict = (county_counts / n_years).to_dict()
        
        print(f"  计算了 {len(lambda_dict)} 个县的频率")
        print(f"  平均频率: {np.mean(list(lambda_dict.values())):.2f} 次/年")
        print(f"  中位数频率: {np.median(list(lambda_dict.values())):.2f} 次/年")
    
    return lambda_dict


def main():
    """主流程"""
    print("="*80)
    print("完整停电仿真 - Demo")
    print("="*80)
    
    # ========================================
    # 步骤1: 初始化严重性采样器
    # ========================================
    print("\n" + "="*80)
    print("步骤1: 初始化严重性采样器")
    print("="*80)
    
    sampler = OutageSeveritySampler(
        historical_data_path='full_pop_2022.csv',
        min_samples=10  # 少于10个样本的县从州级借力
    )
    
    # ========================================
    # 步骤2: 加载频率模型
    # ========================================
    print("\n" + "="*80)
    print("步骤2: 加载频率模型")
    print("="*80)
    
    # 如果你有保存的频率模型：
    # lambda_dict = load_frequency_model('county_lambda.csv')
    
    # 否则从历史数据计算：
    lambda_dict = load_frequency_model()
    
    # ========================================
    # 步骤3: 初始化仿真器
    # ========================================
    print("\n" + "="*80)
    print("步骤3: 初始化仿真器")
    print("="*80)
    
    simulator = OutageSimulator(sampler, lambda_dict)
    
    # ========================================
    # 步骤4: 运行仿真
    # ========================================
    print("\n" + "="*80)
    print("步骤4: 运行仿真")
    print("="*80)
    
    # 选项A: 仿真单年
    print("\n选项A: 仿真单年...")
    df_single = simulator.simulate_year(year=2024, random_seed=42)
    
    print(f"\n单年仿真结果预览:")
    print(df_single.head(10))
    
    # 选项B: 仿真多年
    print("\n\n选项B: 仿真多年...")
    df_multi = simulator.simulate_multiple_years(
        n_years=10, 
        start_year=2024
    )
    
    print(f"\n多年仿真结果预览:")
    print(df_multi.head(10))
    
    # ========================================
    # 步骤5: 分析仿真结果
    # ========================================
    print("\n" + "="*80)
    print("步骤5: 分析仿真结果")
    print("="*80)
    
    output_path = simulator.analyze_simulated_outages(
        df_multi,
        output_dir='./simulation_results'
    )
    
    # ========================================
    # 步骤6: 验证采样器（可选）
    # ========================================
    print("\n" + "="*80)
    print("步骤6: 验证采样器（示例）")
    print("="*80)
    
    # 挑几个县验证
    print("\n选择几个典型县进行验证...")
    
    # 找一个样本量大的县
    county_samples = df_multi['fips'].value_counts()
    if len(county_samples) > 0:
        test_county = county_samples.index[0]
        print(f"\n验证县: {test_county}")
        validation = sampler.validate_sampler(test_county, n_samples=1000)
    
    # ========================================
    # 步骤7: 为EV充电网络仿真准备数据
    # ========================================
    print("\n" + "="*80)
    print("步骤7: 为EV充电网络仿真准备数据")
    print("="*80)
    
    print("\n生成的停电事件可以直接用于充电网络仿真：")
    print("""
    伪代码示例：
    
    # 加载EV充电网络
    network = load_charging_network()
    
    # 初始化韧性指标
    total_unmet_demand = 0
    total_downtime_hours = 0
    
    # 逐个停电事件评估影响
    for _, outage in df_multi.iterrows():
        county = outage['fips']
        duration_hr = outage['duration_hr']
        customers_pct = outage['customers_pct']
        day_of_year = outage['day_of_year']
        
        # 找到受影响的充电站
        affected_stations = network.get_stations_in_county(county)
        
        # 计算未满足的充电需求
        unmet_demand = 0
        for station in affected_stations:
            # 该站点在停电期间的预期需求
            expected_demand = station.get_demand(day_of_year, duration_hr)
            unmet_demand += expected_demand
        
        total_unmet_demand += unmet_demand
        total_downtime_hours += duration_hr * len(affected_stations)
    
    # 韧性指标
    print(f"Total unmet charging demand: {total_unmet_demand} kWh")
    print(f"Average downtime per station: {total_downtime_hours / len(network)} hours")
    """)
    
    print("\n" + "="*80)
    print("✅ 完整仿真流程演示完成！")
    print("="*80)
    
    print(f"\n输出文件位置: {output_path}")
    print("  - simulated_outages.csv: 所有模拟停电事件")
    print("  - simulation_summary.png: 可视化总结")
    
    return df_multi, sampler, simulator


def analyze_severity_by_county(df_outages, top_n=20):
    """分析各县的停电严重性"""
    print("\n" + "="*80)
    print(f"县级严重性分析 - Top {top_n}")
    print("="*80)
    
    # 计算每个县的综合严重性指标
    county_severity = df_outages.groupby('fips').agg({
        'duration_hr': ['count', 'mean', 'median', 'max'],
        'customers_pct': ['mean', 'max']
    }).reset_index()
    
    county_severity.columns = ['fips', 'n_outages', 'mean_duration', 
                               'median_duration', 'max_duration',
                               'mean_customers_pct', 'max_customers_pct']
    
    # 综合严重性得分 = 频率 × 平均时长 × 平均影响
    county_severity['severity_score'] = (
        county_severity['n_outages'] * 
        county_severity['mean_duration'] * 
        county_severity['mean_customers_pct']
    )
    
    # 排序
    county_severity = county_severity.sort_values('severity_score', ascending=False)
    
    print(f"\n最严重的 {top_n} 个县:")
    print("-"*80)
    print(county_severity.head(top_n).to_string(index=False))
    
    return county_severity


def export_for_charging_network(df_outages, output_file='outages_for_ev_network.csv'):
    """导出适合EV充电网络仿真的格式"""
    print("\n" + "="*80)
    print("导出EV充电网络仿真格式")
    print("="*80)
    
    # 添加一些有用的字段
    df_export = df_outages.copy()
    
    # 转换为时间戳（如果需要）
    if 'year' in df_export.columns and 'day_of_year' in df_export.columns:
        df_export['date'] = pd.to_datetime(
            df_export['year'].astype(str) + 
            df_export['day_of_year'].astype(str).str.zfill(3),
            format='%Y%j'
        )
    
    # 计算结束时间
    df_export['end_time'] = df_export['date'] + pd.to_timedelta(
        df_export['duration_hr'], unit='h'
    )
    
    # 保存
    df_export.to_csv(output_file, index=False)
    print(f"\n✓ 导出完成: {output_file}")
    print(f"  字段: {list(df_export.columns)}")
    print(f"  总事件数: {len(df_export)}")
    
    return df_export


if __name__ == '__main__':
    # 运行主流程
    df_outages, sampler, simulator = main()
    
    # 额外分析
    print("\n\n" + "="*80)
    print("额外分析")
    print("="*80)
    
    # 县级严重性分析
    severity_by_county = analyze_severity_by_county(df_outages, top_n=20)
    
    # 导出EV网络格式
    df_export = export_for_charging_network(
        df_outages, 
        output_file='./simulation_results/outages_for_ev_network.csv'
    )
    
    print("\n\n" + "="*80)
    print("🎉 全部完成！")
    print("="*80)
    print("\n下一步：")
    print("1. 查看 ./simulation_results/ 中的结果")
    print("2. 使用 outages_for_ev_network.csv 进行EV充电网络仿真")
    print("3. 根据需要调整参数重新运行")
