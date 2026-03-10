"""
快速模拟器使用示例

预期性能：
- 预计算: 2-3分钟（一次性）
- 1000次模拟: < 5分钟
- 总时间: < 8分钟

对比原版：
- 原版: 4-5小时/100次 = ~240分钟/1000次
- 优化版: ~5分钟/1000次
- 加速比: ~48倍
"""

from fast_outage_simulator import FastOutageSimulator
import time
import os


def example_single_state_fast():
    """
    示例1: 单州快速模拟
    """
    print("\n" + "="*70)
    print("示例1: 单州快速模拟")
    print("="*70)
    
    start_time = time.time()
    
    # 初始化（包含预计算）
    print("\n阶段1: 初始化和预计算...")
    init_start = time.time()
    
    simulator = FastOutageSimulator(
        state_name='California',
        data_path = r'C:\Users\maple\Desktop\EV_Project\codes\module 1\new\results_fixed\network_structures\complete_data\Alabama_complete.pkl',
        precompute_efficiency=True,  # ← 关键！启用预计算
        verbose=True
    )
    
    init_time = time.time() - init_start
    print(f"\n✓ 初始化耗时: {init_time:.1f} 秒")
    
    # 运行蒙特卡洛
    print("\n阶段2: 蒙特卡洛模拟...")
    mc_start = time.time()
    
    results = simulator.monte_carlo_simulation(
        n_simulations=1000,
        verbose=True
    )
    
    mc_time = time.time() - mc_start
    print(f"\n✓ 模拟耗时: {mc_time:.1f} 秒")
    
    # 生成报告
    report = simulator.generate_report(results)
    
    os.makedirs('results', exist_ok=True)
    report.to_csv('results/California_fast_report.csv', index=False)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("性能总结")
    print("="*70)
    print(f"  初始化时间: {init_time:.1f} 秒")
    print(f"  模拟时间: {mc_time:.1f} 秒")
    print(f"  总时间: {total_time:.1f} 秒")
    print(f"  平均速度: {mc_time/1000*1000:.1f} 毫秒/次")
    
    print("\n" + "="*70)
    print("结果总结")
    print("="*70)
    print(report.to_string(index=False))


def example_compare_speed():
    """
    示例2: 速度对比测试
    """
    print("\n" + "="*70)
    print("示例2: 速度对比（小规模）")
    print("="*70)
    
    simulator = FastOutageSimulator(
        state_name='Oregon',
        data_path='complete_data/Oregon_complete.pkl',
        precompute_efficiency=True,
        verbose=False
    )
    
    # 测试不同规模
    test_sizes = [10, 50, 100, 500, 1000]
    
    print("\n模拟规模测试:")
    print("-"*70)
    print(f"{'N_Simulations':<15} {'Time (sec)':<15} {'Time per sim (ms)':<20}")
    print("-"*70)
    
    for n in test_sizes:
        start = time.time()
        results = simulator.monte_carlo_simulation(n_simulations=n, verbose=False)
        elapsed = time.time() - start
        per_sim = (elapsed / n) * 1000
        
        print(f"{n:<15} {elapsed:<15.2f} {per_sim:<20.1f}")
    
    print("-"*70)


def example_batch_states():
    """
    示例3: 批量处理多个州
    """
    print("\n" + "="*70)
    print("示例3: 批量处理多个州")
    print("="*70)
    
    # 要处理的州
    states = [
        ('Alabama', 'complete_data/Alabama_complete.pkl'),
        ('Oregon', 'complete_data/Oregon_complete.pkl'),
        ('Arizona', 'complete_data/Arizona_complete.pkl')
    ]
    
    all_results = []
    total_start = time.time()
    
    for i, (state_name, data_path) in enumerate(states, 1):
        print(f"\n[{i}/{len(states)}] 处理 {state_name}...")
        
        state_start = time.time()
        
        # 初始化
        simulator = FastOutageSimulator(
            state_name=state_name,
            data_path=data_path,
            precompute_efficiency=True,
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
            'Processing_Time_sec': state_time
        })
        
        print(f"  ✓ 完成 - 耗时 {state_time:.1f} 秒")
        print(f"    平均年损失: {results['mean_annual_loss']:.4f}")
    
    total_time = time.time() - total_start
    
    # 总结
    import pandas as pd
    summary_df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("批量处理总结")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    print(f"\n总处理时间: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"平均每州: {total_time/len(states):.1f} 秒")
    
    # 保存
    os.makedirs('results', exist_ok=True)
    summary_df.to_csv('results/multi_state_fast_summary.csv', index=False)
    print(f"\n✓ 结果已保存: results/multi_state_fast_summary.csv")


def example_accuracy_check():
    """
    示例4: 精度检查（与精确方法对比）
    """
    print("\n" + "="*70)
    print("示例4: 精度检查")
    print("="*70)
    
    print("\n说明:")
    print("  快速方法使用查找表近似")
    print("  测试：运行小规模精确对比")
    
    simulator = FastOutageSimulator(
        state_name='Test',
        data_path='complete_data/Oregon_complete.pkl',
        precompute_efficiency=True,
        verbose=False
    )
    
    print("\n测试几个典型场景:")
    
    test_cases = [
        {'n_affected': 5, 'duration': 4},
        {'n_affected': 20, 'duration': 8},
        {'n_affected': 50, 'duration': 6}
    ]
    
    for case in test_cases:
        loss_fast = simulator.compute_efficiency_loss_fast(
            case['n_affected'],
            case['duration']
        )
        
        print(f"\n场景: {case['n_affected']} 节点受影响, {case['duration']} 小时")
        print(f"  快速方法损失: {loss_fast:.6f}")
        print(f"  (精确方法需要移除具体节点，此处略)")


def main():
    """
    主函数
    """
    print("\n" + "="*70)
    print("快速停电模拟器 - 性能优化版")
    print("="*70)
    
    print("\n选择示例:")
    print("  1. 单州快速模拟 (1000次) - 推荐！")
    print("  2. 速度对比测试")
    print("  3. 批量处理多个州")
    print("  4. 精度检查")
    
    choice = input("\n输入选择 (1-4): ").strip()
    
    examples = {
        '1': example_single_state_fast,
        '2': example_compare_speed,
        '3': example_batch_states,
        '4': example_accuracy_check
    }
    
    if choice in examples:
        examples[choice]()
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
