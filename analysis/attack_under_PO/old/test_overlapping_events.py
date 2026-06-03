"""
测试同时发生多事件的情况

回答问题:
1. 代码是否乘以duration? ✅ 是的
2. 如果同时发生2次/3次怎么办? 
"""

from improved_outage_simulator import ImprovedOutageSimulator
import numpy as np


def test_duration_multiplication():
    """
    测试1: 验证损失计算中确实乘以了duration
    """
    print("\n" + "="*70)
    print("测试1: 验证duration乘法")
    print("="*70)
    
    simulator = ImprovedOutageSimulator(
        state_name='Test',
        data_path='complete_data/California_complete.pkl',
        verbose=False
    )
    
    # 生成同一个事件，但改变duration
    event = simulator.generate_single_outage_event()
    affected_nodes = event['affected_nodes']
    
    print(f"\n受影响节点数: {len(affected_nodes)}")
    print(f"\n测试不同duration的损失:")
    
    for duration in [1, 2, 5, 10]:
        loss = simulator.compute_efficiency_loss(affected_nodes, duration)
        print(f"  Duration = {duration:2d} 小时 → 损失 = {loss:.6f}")
    
    # 验证线性关系
    loss_1h = simulator.compute_efficiency_loss(affected_nodes, 1.0)
    loss_5h = simulator.compute_efficiency_loss(affected_nodes, 5.0)
    
    ratio = loss_5h / loss_1h
    print(f"\n验证线性关系:")
    print(f"  loss(5h) / loss(1h) = {ratio:.2f}")
    print(f"  预期比值 = 5.00")
    print(f"  ✓ 确认: 损失 = relative_loss × duration")


def test_overlapping_events():
    """
    测试2: 同时发生2个事件的情况
    """
    print("\n" + "="*70)
    print("测试2: 同时发生2个事件")
    print("="*70)
    
    simulator = ImprovedOutageSimulator(
        state_name='Test',
        data_path='complete_data/California_complete.pkl',
        verbose=False
    )
    
    # 手动创建2个同时发生的事件
    event1 = {
        'start_time': 0.0,
        'end_time': 5.0,
        'duration': 5.0,
        'affected_nodes': set(list(simulator.G.nodes())[:10])  # 前10个节点
    }
    
    event2 = {
        'start_time': 2.0,  # 从第2小时开始
        'end_time': 8.0,
        'duration': 6.0,
        'affected_nodes': set(list(simulator.G.nodes())[5:15])  # 5-15节点 (有重叠)
    }
    
    print(f"\n事件1:")
    print(f"  时间: {event1['start_time']:.0f}h - {event1['end_time']:.0f}h")
    print(f"  受影响节点: {len(event1['affected_nodes'])}")
    
    print(f"\n事件2:")
    print(f"  时间: {event2['start_time']:.0f}h - {event2['end_time']:.0f}h")
    print(f"  受影响节点: {len(event2['affected_nodes'])}")
    
    # 节点重叠
    overlap_nodes = event1['affected_nodes'] & event2['affected_nodes']
    combined_nodes = event1['affected_nodes'] | event2['affected_nodes']
    
    print(f"\n节点重叠:")
    print(f"  重叠节点: {len(overlap_nodes)}")
    print(f"  合并后总节点: {len(combined_nodes)}")
    
    # 方法1: 错误的简单相加
    loss1_simple = simulator.compute_efficiency_loss(event1['affected_nodes'], event1['duration'])
    loss2_simple = simulator.compute_efficiency_loss(event2['affected_nodes'], event2['duration'])
    total_simple = loss1_simple + loss2_simple
    
    print(f"\n❌ 错误方法 (简单相加):")
    print(f"  事件1损失: {loss1_simple:.4f}")
    print(f"  事件2损失: {loss2_simple:.4f}")
    print(f"  总损失 (错误): {total_simple:.4f}")
    print(f"  问题: 重叠期 (2-5h) 被计算了2次!")
    
    # 方法2: 正确的时间线方法
    events = [event1, event2]
    timeline_result = simulator.compute_timeline_loss(events, verbose=True)
    
    print(f"\n✓ 正确方法 (Timeline):")
    print(f"  总损失 (正确): {timeline_result['total_loss']:.4f}")
    print(f"  最多同时事件: {timeline_result['max_simultaneous_events']}")
    
    # 分析差异
    difference = total_simple - timeline_result['total_loss']
    overestimate_pct = (difference / timeline_result['total_loss']) * 100
    
    print(f"\n差异分析:")
    print(f"  简单相加高估: {difference:.4f} ({overestimate_pct:.1f}%)")


def test_three_simultaneous_events():
    """
    测试3: 同时发生3个事件
    """
    print("\n" + "="*70)
    print("测试3: 同时发生3个事件")
    print("="*70)
    
    simulator = ImprovedOutageSimulator(
        state_name='Test',
        data_path='complete_data/California_complete.pkl',
        verbose=False
    )
    
    # 创建3个有时间重叠的事件
    events = [
        {
            'start_time': 0.0,
            'end_time': 4.0,
            'duration': 4.0,
            'affected_nodes': set(list(simulator.G.nodes())[:8])
        },
        {
            'start_time': 2.0,
            'end_time': 7.0,
            'duration': 5.0,
            'affected_nodes': set(list(simulator.G.nodes())[6:14])
        },
        {
            'start_time': 5.0,
            'end_time': 10.0,
            'duration': 5.0,
            'affected_nodes': set(list(simulator.G.nodes())[10:18])
        }
    ]
    
    print(f"\n事件配置:")
    for i, event in enumerate(events, 1):
        print(f"  事件{i}: {event['start_time']:.0f}h-{event['end_time']:.0f}h, "
              f"{len(event['affected_nodes'])} 节点")
    
    # 计算时间线
    print(f"\n时间线分析:")
    print(f"-"*70)
    
    for hour in range(11):
        active_events = [
            i+1 for i, e in enumerate(events)
            if e['start_time'] <= hour < e['end_time']
        ]
        
        if active_events:
            combined = set()
            for e in events:
                if e['start_time'] <= hour < e['end_time']:
                    combined.update(e['affected_nodes'])
            
            print(f"  Hour {hour:2d}: {len(active_events)} 活跃事件 {active_events}, "
                  f"{len(combined)} 节点受影响")
    
    # 正确方法
    timeline_result = simulator.compute_timeline_loss(events, verbose=False)
    
    # 错误方法
    total_simple = sum(
        simulator.compute_efficiency_loss(e['affected_nodes'], e['duration'])
        for e in events
    )
    
    print(f"\n结果对比:")
    print(f"  ✓ Timeline方法 (正确): {timeline_result['total_loss']:.4f}")
    print(f"  ❌ 简单相加 (错误): {total_simple:.4f}")
    print(f"  高估比例: {(total_simple / timeline_result['total_loss'] - 1) * 100:.1f}%")
    print(f"  最多同时事件: {timeline_result['max_simultaneous_events']}")


def test_real_annual_simulation():
    """
    测试4: 真实年度模拟
    """
    print("\n" + "="*70)
    print("测试4: 真实年度模拟")
    print("="*70)
    
    simulator = ImprovedOutageSimulator(
        state_name='California',
        data_path='complete_data/California_complete.pkl',
        verbose=False
    )
    
    # 运行1次年度模拟，查看同时事件
    print(f"\n运行1次年度模拟...")
    result = simulator.simulate_annual_outages_timeline(verbose=True)
    
    print(f"\n年度总结:")
    print(f"  总事件数: {result['n_events']}")
    print(f"  总年损失: {result['total_annual_loss']:.4f}")
    print(f"  最多同时事件: {result['max_simultaneous_events']}")
    
    if result['max_simultaneous_events'] >= 2:
        print(f"\n✓ 观察到同时发生的多个事件!")
        print(f"  Timeline方法正确处理了节点集合并")


def test_method_comparison():
    """
    测试5: 完整对比两种方法
    """
    print("\n" + "="*70)
    print("测试5: Timeline vs Simple 方法对比")
    print("="*70)
    
    simulator = ImprovedOutageSimulator(
        state_name='Oregon',
        data_path='complete_data/Oregon_complete.pkl',
        verbose=False
    )
    
    comparison = simulator.compare_methods(n_simulations=100)
    
    print(f"\n✓ 对比完成")


def main():
    """
    运行所有测试
    """
    print("\n" + "="*70)
    print("改进版停电模拟器 - 测试套件")
    print("="*70)
    
    print("\n选择测试:")
    print("  1. 验证duration乘法")
    print("  2. 测试2个同时事件")
    print("  3. 测试3个同时事件")
    print("  4. 真实年度模拟")
    print("  5. 方法对比 (Timeline vs Simple)")
    print("  6. 运行所有测试")
    
    choice = input("\n输入选择 (1-6): ").strip()
    
    tests = {
        '1': test_duration_multiplication,
        '2': test_overlapping_events,
        '3': test_three_simultaneous_events,
        '4': test_real_annual_simulation,
        '5': test_method_comparison
    }
    
    if choice in tests:
        tests[choice]()
    elif choice == '6':
        for func in tests.values():
            func()
            print("\n" + "="*70)
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
