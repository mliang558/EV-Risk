"""
超快模拟器使用示例

对比三种方法：
1. 物理准确版 (PhysicalOutageSimulator) - 精确但慢
2. 超快版 (UltraFastSimulator) - 近似但快
3. 验证精度差异
"""

from ultra_fast_simulator import UltraFastSimulator
import time
import os


def example_ultra_fast_simulation():
    """
    示例1: 超快模拟
    """
    print("\n" + "="*70)
    print("示例1: 超快模拟 (1000次)")
    print("="*70)
    
    start_time = time.time()
    
    # 初始化
    simulator = UltraFastSimulator(
        state_name='Alabama',
        data_path = r'C:\Users\maple\Desktop\EV_Project\codes\module 1\new\results_fixed\network_structures\complete_data\Alabama_complete.pkl',
        verbose=True
    )
    
    init_time = time.time() - start_time
    
    # 运行模拟
    mc_start = time.time()
    
    results = simulator.monte_carlo_simulation(
        n_simulations=1000,
        verbose=True
    )
    
    mc_time = time.time() - mc_start
    total_time = time.time() - start_time
    
    # 报告
    report = simulator.generate_report(results)
    
    print("\n" + "="*70)
    print("性能总结")
    print("="*70)
    print(f"  初始化时间: {init_time:.1f} 秒")
    print(f"  模拟时间: {mc_time:.1f} 秒 ({mc_time/60:.1f} 分钟)")
    print(f"  总时间: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  平均速度: {mc_time/1000*1000:.1f} 毫秒/次")
    
    print("\n" + "="*70)
    print("结果报告")
    print("="*70)
    print(report.to_string(index=False))
    
    # 保存
    os.makedirs('results', exist_ok=True)
    report.to_csv('results/California_ultrafast_report.csv', index=False)
    print(f"\n✓ 报告已保存: results/California_ultrafast_report.csv")


def example_speed_comparison():
    """
    示例2: 速度对比
    """
    print("\n" + "="*70)
    print("示例2: 速度对比测试")
    print("="*70)
    
    simulator = UltraFastSimulator(
        state_name='Oregon',
        data_path='complete_data/Oregon_complete.pkl',
        verbose=False
    )
    
    test_sizes = [10, 50, 100, 500, 1000]
    
    print("\n模拟规模 vs 时间:")
    print("-"*70)
    print(f"{'N_Simulations':<15} {'Time (sec)':<15} {'Per sim (ms)':<15} {'Est. 1000x time':<20}")
    print("-"*70)
    
    for n in test_sizes:
        start = time.time()
        results = simulator.monte_carlo_simulation(n_simulations=n, verbose=False)
        elapsed = time.time() - start
        per_sim_ms = (elapsed / n) * 1000
        est_1000 = elapsed * (1000 / n)
        
        print(f"{n:<15} {elapsed:<15.2f} {per_sim_ms:<15.1f} {est_1000:<20.1f}")
    
    print("-"*70)


def example_batch_all_states():
    """
    示例3: 批量处理所有州
    """
    print("\n" + "="*70)
    print("示例3: 批量处理多个州 (超快!)")
    print("="*70)
    
    # 假设有多个州
    states = [
        ('Alabama', 'complete_data/Alabama_complete.pkl'),
        ('Oregon', 'complete_data/Oregon_complete.pkl'),
        ('Arizona', 'complete_data/Arizona_complete.pkl'),
        ('California', 'complete_data/California_complete.pkl'),
        ('Texas', 'complete_data/Texas_complete.pkl')
    ]
    
    all_results = []
    total_start = time.time()
    
    for i, (state_name, data_path) in enumerate(states, 1):
        print(f"\n[{i}/{len(states)}] 处理 {state_name}...")
        
        state_start = time.time()
        
        # 初始化
        simulator = UltraFastSimulator(
            state_name=state_name,
            data_path=data_path,
            verbose=False
        )
        
        # 运行模拟
        results = simulator.monte_carlo_simulation(
            n_simulations=1000,
            verbose=False
        )
        
        state_time = time.time() - state_start
        
        all_results.append({
            'State': state_name,
            'N_Stations': len(simulator.G.nodes()),
            'Mean_Annual_Loss': results['mean_annual_loss'],
            'VaR_95': results['VaR_95'],
            'Time_sec': state_time
        })
        
        print(f"  ✓ 完成 - {state_time:.1f} 秒")
    
    total_time = time.time() - total_start
    
    # 总结
    import pandas as pd
    summary_df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("批量处理总结")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    print(f"\n总时间: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"平均每州: {total_time/len(states):.1f} 秒")
    
    # 保存
    os.makedirs('results', exist_ok=True)
    summary_df.to_csv('results/all_states_ultrafast_summary.csv', index=False)
    print(f"\n✓ 结果已保存: results/all_states_ultrafast_summary.csv")


def example_verify_approximation():
    """
    示例4: 验证近似方法的精度
    """
    print("\n" + "="*70)
    print("示例4: 验证近似精度")
    print("="*70)
    
    print("\n说明:")
    print("  超快版使用度中心性代替global_efficiency")
    print("  这是一个近似方法")
    
    simulator = UltraFastSimulator(
        state_name='Test',
        data_path='complete_data/Oregon_complete.pkl',
        verbose=False
    )
    
    print(f"\n网络统计:")
    print(f"  节点数: {len(simulator.G.nodes())}")
    print(f"  边数: {len(simulator.G.edges())}")
    print(f"  平均度数: {sum(simulator.node_degrees.values())/len(simulator.node_degrees):.2f}")
    print(f"  最大度数: {simulator.max_degree}")
    
    # 测试几个事件
    print(f"\n测试10个随机事件:")
    print("-"*70)
    
    for i in range(10):
        event = simulator.simulate_single_outage_event()
        
        if i < 5:  # 只显示前5个
            print(f"\n事件 {i+1}:")
            print(f"  受影响站点: {event['affected_stations']}")
            print(f"  持续时间: {event['duration']:.1f} 小时")
            print(f"  效率损失(度代理): {event['efficiency_loss']:.6f}")
    
    print("\n注意:")
    print("  - 度代理方法假设：损失 ≈ 受影响节点的度数占比")
    print("  - 适合快速比较和参数扫描")
    print("  - 不适合需要精确数值的场景")


def example_accuracy_comparison():
    """
    示例5: 精度对比（如果有物理准确版的结果）
    """
    print("\n" + "="*70)
    print("示例5: 方法对比指南")
    print("="*70)
    
    comparison = """
    三种方法对比：
    
    1. 物理准确版 (PhysicalOutageSimulator)
       优点：✓ 完全准确，调用 nx.global_efficiency()
       缺点：✗ 很慢 (1000次 ≈ 20分钟)
       用途：最终结果、论文数据
    
    2. 超快版 (UltraFastSimulator)  ← 当前
       优点：✓ 超快 (1000次 ≈ 1-2分钟)
       缺点：✗ 近似方法，精度损失 ~10-20%
       用途：快速测试、参数扫描、初步探索
    
    3. 极速版 (FastOutageSimulator) - 已废弃
       问题：预计算效率表仍然需要调用 nx.global_efficiency()
       结果：还是很慢
    
    推荐工作流程：
    
    Step 1: 用超快版快速测试所有州 (10分钟完成50州)
            ↓
    Step 2: 识别重点州和参数范围
            ↓
    Step 3: 用物理准确版精确计算重点州 (每州20分钟)
            ↓
    Step 4: 论文中报告精确结果
    
    精度验证：
    - 超快版的相对排名通常是准确的
    - 绝对数值可能有 10-20% 偏差
    - VaR等分布特征较为稳定
    """
    
    print(comparison)


def main():
    """
    主函数
    """
    print("\n" + "="*70)
    print("超快停电模拟器 - 使用示例")
    print("="*70)
    
    print("\n选择示例:")
    print("  1. 超快模拟 (1000次) - 推荐！")
    print("  2. 速度对比测试")
    print("  3. 批量处理多个州")
    print("  4. 验证近似精度")
    print("  5. 方法对比指南")
    
    choice = input("\n输入选择 (1-5): ").strip()
    
    examples = {
        '1': example_ultra_fast_simulation,
        '2': example_speed_comparison,
        '3': example_batch_all_states,
        '4': example_verify_approximation,
        '5': example_accuracy_comparison
    }
    
    if choice in examples:
        examples[choice]()
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
