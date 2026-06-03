"""
使用示例 - 改进版

展示如何使用改进后的RealisticOutageSimulator
"""

from realistic_outage_simulator_improved import RealisticOutageSimulator, run_multi_state_comparison
import os


def example_single_state():
    """
    示例1: 单个州的分析
    """
    print("\n" + "="*70)
    print("示例1: 单州分析 - Alabama")
    print("="*70)
    
    # 初始化模拟器
    simulator = RealisticOutageSimulator(
        state_name='Alabama',
        data_path=r'C:\Users\maple\Desktop\EV_Project\codes\module 1\new\results_fixed\network_structures\complete_data\Alabama_complete.pkl',
        
        verbose=True
    )
    
    # 运行蒙特卡洛模拟
    print("\n运行1000次蒙特卡洛模拟...")
    results = simulator.monte_carlo_simulation(
        n_simulations=1000,
        recovery_model='constant',
        verbose=True
    )
    
    # 生成报告
    print("\n生成报告...")
    report = simulator.generate_report(results)
    print("\n" + "="*70)
    print("详细报告")
    print("="*70)
    print(report.to_string(index=False))
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    report.to_csv('results/Alabama_report.csv', index=False)
    print(f"\n✓ 报告已保存: results/Alabama_report.csv")


def example_analyze_single_event():
    """
    示例2: 分析单个停电事件
    """
    print("\n" + "="*70)
    print("示例2: 单个停电事件分析")
    print("="*70)
    
    simulator = RealisticOutageSimulator(
        state_name='California',
        data_path='complete_data/California_complete.pkl',
        verbose=False
    )
    
    # 模拟10个随机停电事件
    print("\n模拟10个随机停电事件:")
    print("-"*70)
    
    for i in range(10):
        event = simulator.simulate_single_outage_event()
        
        print(f"\n事件 {i+1}:")
        print(f"  County: {event['county']}")
        print(f"  持续时间: {event['duration']:.1f} 小时")
        print(f"  受影响客户: {event['affected_customers']:,}")
        print(f"  影响半径: {event['radius_km']:.1f} km")
        print(f"  受影响充电站: {event['affected_stations']}")
        print(f"  效率损失: {event['efficiency_loss']:.4f}")


def example_multi_state_comparison():
    """
    示例3: 多州对比
    """
    print("\n" + "="*70)
    print("示例3: 多州对比分析")
    print("="*70)
    
    # 配置要对比的州
    state_configs = {
        'Alabama': 'complete_data/Alabama_complete.pkl',
        'Arizona': 'complete_data/Arizona_complete.pkl',
        'California': 'complete_data/California_complete.pkl',
        'Oregon': 'complete_data/Oregon_complete.pkl',
        'Texas': 'complete_data/Texas_complete.pkl'
    }
    
    # 运行对比分析
    print("\n运行对比分析...")
    comparison_df = run_multi_state_comparison(
        state_configs=state_configs,
        n_simulations=1000,
        recovery_model='constant'
    )
    
    # 显示结果
    print("\n" + "="*70)
    print("多州对比结果")
    print("="*70)
    
    display_cols = [
        'State', 'N_Stations', 'N_Counties', 'Total_Lambda',
        'Mean_Annual_Loss', 'VaR_95', 'Loss_Per_Station',
        'Vulnerability_Rank'
    ]
    
    print(comparison_df[display_cols].to_string(index=False))
    
    # 保存
    os.makedirs('results', exist_ok=True)
    comparison_df.to_csv('results/multi_state_comparison.csv', index=False)
    print(f"\n✓ 对比结果已保存: results/multi_state_comparison.csv")


def example_county_vulnerability():
    """
    示例4: County级脆弱性分析
    """
    print("\n" + "="*70)
    print("示例4: County级脆弱性分析")
    print("="*70)
    
    simulator = RealisticOutageSimulator(
        state_name='Oregon',
        data_path='complete_data/Oregon_complete.pkl',
        verbose=False
    )
    
    print(f"\n分析{len(simulator.counties)}个counties的脆弱性...")
    
    county_vulnerability = []
    
    for county in simulator.counties[:10]:  # 分析前10个counties
        # 为每个county模拟20个事件
        losses = []
        for _ in range(20):
            event = simulator.simulate_single_outage_event(county=county)
            losses.append(event['efficiency_loss'])
        
        avg_loss_per_event = sum(losses) / len(losses)
        annual_expected_loss = avg_loss_per_event * simulator.lambda_county[county]
        
        county_vulnerability.append({
            'County': county,
            'Lambda': simulator.lambda_county[county],
            'Avg_Loss_Per_Event': avg_loss_per_event,
            'Annual_Expected_Loss': annual_expected_loss,
            'N_Stations': len(simulator.nodes_by_county.get(county, []))
        })
    
    # 排序
    county_vulnerability.sort(key=lambda x: x['Annual_Expected_Loss'], reverse=True)
    
    print("\n" + "-"*70)
    print("County脆弱性排名 (Top 10)")
    print("-"*70)
    print(f"{'Rank':<6} {'County':<10} {'Lambda':<8} {'Stations':<10} {'Annual Loss':<12}")
    print("-"*70)
    
    for i, cv in enumerate(county_vulnerability, 1):
        print(f"{i:<6} {cv['County']:<10} {cv['Lambda']:<8.2f} {cv['N_Stations']:<10} {cv['Annual_Expected_Loss']:<12.4f}")


def main():
    """
    主函数
    """
    print("\n" + "="*70)
    print("Realistic Outage Simulator - 使用示例")
    print("="*70)
    
    print("\n选择示例:")
    print("  1. 单州分析")
    print("  2. 单个事件分析")
    print("  3. 多州对比")
    print("  4. County脆弱性分析")
    print("  5. 运行所有示例")
    
    choice = input("\n输入选择 (1-5): ").strip()
    
    examples = {
        '1': example_single_state,
        '2': example_analyze_single_event,
        '3': example_multi_state_comparison,
        '4': example_county_vulnerability
    }
    
    if choice in examples:
        examples[choice]()
    elif choice == '5':
        for func in examples.values():
            func()
            print("\n" + "="*70)
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
