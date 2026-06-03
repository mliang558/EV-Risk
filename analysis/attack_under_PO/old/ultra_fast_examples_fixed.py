"""
超快模拟器使用示例 - 路径已修正（已归档到 old/ 目录）
"""

from ultra_fast_simulator import UltraFastSimulator
import time
import os


# ============================================================
# 配置数据路径
# ============================================================
DATA_BASE_DIR = r'C:\Users\maple\Desktop\EV_Project\codes\module 1\new\results_fixed\network_structures\complete_data'


def example_ultra_fast_simulation():
    """
    示例1: 超快模拟 (1000次)
    """
    print("\n" + "=" * 70)
    print("示例1: 超快模拟 (1000次)")
    print("=" * 70)

    # 选择一个州测试
    state = 'California'
    data_path = os.path.join(DATA_BASE_DIR, f'{state}_complete.pkl')

    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return

    start_time = time.time()

    # 初始化
    simulator = UltraFastSimulator(
        state_name=state,
        data_path=data_path,
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

    print("\n" + "=" * 70)
    print("性能总结")
    print("=" * 70)
    print(f"  初始化时间: {init_time:.1f} 秒")
    print(f"  模拟时间: {mc_time:.1f} 秒 ({mc_time/60:.1f} 分钟)")
    print(f"  总时间: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  平均速度: {mc_time/1000*1000:.1f} 毫秒/次")

    print("\n" + "=" * 70)
    print("结果报告")
    print("=" * 70)
    print(report.to_string(index=False))

    # 保存
    os.makedirs('results', exist_ok=True)
    report.to_csv(f'results/{state}_ultrafast_report.csv', index=False)
    print(f"\n✓ 报告已保存: results/{state}_ultrafast_report.csv")


def example_batch_all_states():
    """
    示例3: 批量处理多个州
    """
    print("\n" + "=" * 70)
    print("示例3: 批量处理多个州 (超快!)")
    print("=" * 70)

    print(f"\n数据目录: {DATA_BASE_DIR}")

    # 自动查找所有 *_complete.pkl 文件
    import glob
    pattern = os.path.join(DATA_BASE_DIR, '*_complete.pkl')
    all_files = glob.glob(pattern)

    if not all_files:
        print(f"\n❌ 未找到任何文件: {pattern}")
        print(f"\n请检查路径是否正确")
        return

    print(f"\n找到 {len(all_files)} 个州的数据文件")

    # 提取州名
    states_data = []
    for filepath in all_files:
        state_name = os.path.basename(filepath).replace('_complete.pkl', '')
        states_data.append((state_name, filepath))

    # 显示前10个
    print(f"\n数据文件列表 (前10个):")
    for i, (state, path) in enumerate(states_data[:10], 1):
        print(f"  {i}. {state}")

    if len(states_data) > 10:
        print(f"  ... 还有 {len(states_data) - 10} 个州")

    # 确认
    proceed = input(f"\n是否处理所有 {len(states_data)} 个州? (y/n) [y]: ").strip().lower()
    if proceed and proceed != 'y':
        print("取消处理")
        return

    # 模拟次数
    n_sims_input = input("\n每个州的模拟次数 (默认 1000): ").strip()
    n_sims = int(n_sims_input) if n_sims_input else 1000

    # 开始处理
    print("\n" + "=" * 70)
    print(f"开始批量处理 ({n_sims} 次模拟/州)")
    print("=" * 70)

    all_results = []
    total_start = time.time()

    for i, (state_name, filepath) in enumerate(states_data, 1):
        print(f"\n[{i}/{len(states_data)}] 处理 {state_name}...")

        state_start = time.time()

        try:
            # 初始化
            simulator = UltraFastSimulator(
                state_name=state_name,
                data_path=filepath,
                verbose=False
            )

            # 运行模拟
            results = simulator.monte_carlo_simulation(
                n_simulations=n_sims,
                verbose=False
            )

            state_time = time.time() - state_start

            all_results.append({
                'State': state_name,
                'N_Stations': len(simulator.G.nodes()),
                'N_Counties': len(simulator.counties),
                'Total_Lambda': simulator.total_lambda,
                'Mean_Annual_Loss': results['mean_annual_loss'],
                'Std_Annual_Loss': results['std_annual_loss'],
                'VaR_95': results['VaR_95'],
                'CVaR_95': results['CVaR_95'],
                'Mean_N_Events': results['mean_n_events'],
                'Time_sec': state_time
            })

            print(f"  ✓ 完成 - {state_time:.1f} 秒")
            print(f"    充电站: {len(simulator.G.nodes())}")
            print(f"    平均年损失: {results['mean_annual_loss']:.2f}")
            print(f"    年均事件数: {results['mean_n_events']:.0f}")

        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start

    # 总结
    if not all_results:
        print("\n❌ 没有成功处理任何州")
        return

    import pandas as pd
    summary_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("批量处理总结")
    print("=" * 70)

    # 按损失排序
    summary_df = summary_df.sort_values('Mean_Annual_Loss', ascending=False)

    print(f"\n总共处理: {len(summary_df)} 个州")
    print(f"总时间: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"平均每州: {total_time/len(summary_df):.1f} 秒")

    # 显示Top 10
    print("\n" + "=" * 70)
    print("Top 10 最脆弱的州")
    print("=" * 70)

    display_cols = ['State', 'N_Stations', 'Mean_Annual_Loss', 'Mean_N_Events', 'Time_sec']
    print(summary_df[display_cols].head(10).to_string(index=False))

    # 保存结果
    os.makedirs('results', exist_ok=True)
    output_path = 'results/all_states_ultrafast_summary.csv'
    summary_df.to_csv(output_path, index=False)

    print(f"\n✓ 完整结果已保存: {output_path}")


def example_speed_comparison():
    """
    示例2: 速度对比测试
    """
    print("\n" + "=" * 70)
    print("示例2: 速度对比测试")
    print("=" * 70)

    # 选择一个州
    state = 'Oregon'
    data_path = os.path.join(DATA_BASE_DIR, f'{state}_complete.pkl')

    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        # 尝试找第一个可用的文件
        import glob
        files = glob.glob(os.path.join(DATA_BASE_DIR, '*_complete.pkl'))
        if files:
            data_path = files[0]
            state = os.path.basename(data_path).replace('_complete.pkl', '')
            print(f"使用替代州: {state}")
        else:
            return

    simulator = UltraFastSimulator(
        state_name=state,
        data_path=data_path,
        verbose=False
    )

    test_sizes = [10, 50, 100, 500, 1000]

    print("\n模拟规模 vs 时间:")
    print("-" * 70)
    print(f"{'N_Simulations':<15} {'Time (sec)':<15} {'Per sim (ms)':<15} {'Est. 1000x time':<20}")
    print("-" * 70)

    for n in test_sizes:
        start = time.time()
        simulator.monte_carlo_simulation(n_simulations=n, verbose=False)
        elapsed = time.time() - start
        print(f"{n:<15} {elapsed:<15.2f} {elapsed/n*1000:<15.2f} {elapsed/ n * 1000:<20.2f}")

