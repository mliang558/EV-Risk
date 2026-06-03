"""
诊断脚本：检查 Mean_N_Events 和 Lambda

帮助识别：
1. Mean_N_Events 是否异常
2. Lambda 数据是否正确
3. 各county的停电频率分布
"""

import pickle
import numpy as np
import pandas as pd
import os


def diagnose_single_state(data_path: str):
    """
    诊断单个州的停电统计
    """
    
    print("\n" + "="*70)
    print(f"诊断州数据: {data_path}")
    print("="*70)
    
    # 加载数据
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {data_path}")
        return
    
    # 提取信息
    network = data['network']
    lambda_county = data['county_lambdas']
    county_data = data['county_demographics']
    
    # 基本统计
    print(f"\n基本信息:")
    print(f"  充电站数量: {len(network.nodes())}")
    print(f"  Counties 数量: {len(lambda_county)}")
    
    # Lambda 统计
    lambdas = list(lambda_county.values())
    total_lambda = sum(lambdas)
    
    print(f"\nLambda 统计 (停电频率):")
    print(f"  总 Lambda: {total_lambda:.2f} events/year")
    print(f"  平均 Lambda: {np.mean(lambdas):.2f} events/year/county")
    print(f"  中位数 Lambda: {np.median(lambdas):.2f}")
    print(f"  最小 Lambda: {np.min(lambdas):.2f}")
    print(f"  最大 Lambda: {np.max(lambdas):.2f}")
    print(f"  标准差: {np.std(lambdas):.2f}")
    
    # 检查异常值
    print(f"\n异常检查:")
    
    # 检查1: Total Lambda 是否合理
    if total_lambda > 1000:
        print(f"  ⚠️  Total Lambda 非常高 ({total_lambda:.0f})!")
        print(f"      正常范围应该是 50-500 events/year")
        print(f"      可能原因:")
        print(f"        1. Lambda单位错误（应该是events/year，不是events/month）")
        print(f"        2. 数据重复计算")
        print(f"        3. 某些county的lambda异常高")
    elif total_lambda < 10:
        print(f"  ⚠️  Total Lambda 非常低 ({total_lambda:.0f})!")
        print(f"      可能导致模拟结果不稳定")
    else:
        print(f"  ✓ Total Lambda 在合理范围内 ({total_lambda:.0f})")
    
    # 检查2: 异常高的counties
    threshold = np.mean(lambdas) + 3 * np.std(lambdas)
    high_lambda_counties = [
        (county, lam) for county, lam in lambda_county.items() 
        if lam > threshold
    ]
    
    if high_lambda_counties:
        print(f"\n  ⚠️  发现 {len(high_lambda_counties)} 个异常高Lambda的counties:")
        for county, lam in sorted(high_lambda_counties, key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {county}: {lam:.2f} events/year (>{threshold:.1f})")
    
    # 检查3: Lambda为0的counties
    zero_lambda = [c for c, lam in lambda_county.items() if lam == 0]
    if zero_lambda:
        print(f"\n  ⚠️  发现 {len(zero_lambda)} 个Lambda为0的counties")
    
    # 显示Top 10 counties
    print(f"\n" + "="*70)
    print("Top 10 Counties (按Lambda排序)")
    print("="*70)
    
    county_list = [(county, lam) for county, lam in lambda_county.items()]
    county_list.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'County FIPS':<15} {'Lambda':<15} {'Population':<15}")
    print("-"*45)
    
    for county, lam in county_list[:10]:
        pop = county_data.get(county, {}).get('population', 0)
        print(f"{county:<15} {lam:<15.2f} {pop:<15,}")
    
    # 预期的年均事件数
    print(f"\n" + "="*70)
    print("预期模拟结果")
    print("="*70)
    
    print(f"\n在蒙特卡洛模拟中:")
    print(f"  Mean_N_Events 应该接近: {total_lambda:.1f}")
    print(f"  正常范围: [{total_lambda*0.8:.1f}, {total_lambda*1.2:.1f}]")
    
    if total_lambda > 1000:
        print(f"\n  ⚠️  警告: Total Lambda 过高!")
        print(f"  建议检查 bayesian_lambda_results.csv 中的数据单位")
    
    return {
        'total_lambda': total_lambda,
        'n_counties': len(lambda_county),
        'n_stations': len(network.nodes())
    }


def diagnose_all_states(data_dir: str):
    """
    诊断所有州
    """
    
    import glob
    
    print("\n" + "="*70)
    print("批量诊断所有州")
    print("="*70)
    
    # 查找所有文件
    pattern = os.path.join(data_dir, '*_complete.pkl')
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ 未找到任何文件: {pattern}")
        return
    
    print(f"\n找到 {len(files)} 个州的数据")
    
    results = []
    
    for filepath in files:
        state = os.path.basename(filepath).replace('_complete.pkl', '')
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            lambda_county = data['county_lambdas']
            total_lambda = sum(lambda_county.values())
            
            results.append({
                'State': state,
                'Total_Lambda': total_lambda,
                'N_Counties': len(lambda_county),
                'N_Stations': len(data['network'].nodes())
            })
            
        except Exception as e:
            print(f"  ❌ {state}: {str(e)}")
            continue
    
    # 生成报告
    df = pd.DataFrame(results)
    df = df.sort_values('Total_Lambda', ascending=False)
    
    print("\n" + "="*70)
    print("所有州的 Lambda 统计")
    print("="*70)
    
    print(df.to_string(index=False))
    
    # 统计
    print(f"\n总体统计:")
    print(f"  平均 Total Lambda: {df['Total_Lambda'].mean():.1f}")
    print(f"  中位数: {df['Total_Lambda'].median():.1f}")
    print(f"  最小: {df['Total_Lambda'].min():.1f}")
    print(f"  最大: {df['Total_Lambda'].max():.1f}")
    
    # 检查异常
    if df['Total_Lambda'].max() > 1000:
        print(f"\n⚠️  警告: 有些州的 Total Lambda 异常高!")
        high_states = df[df['Total_Lambda'] > 1000]
        print(f"\n异常高的州:")
        print(high_states.to_string(index=False))
    
    # 保存
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/lambda_diagnosis.csv', index=False)
    print(f"\n✓ 诊断结果已保存: results/lambda_diagnosis.csv")
    
    return df


def main():
    """
    主函数
    """
    
    print("\n" + "="*70)
    print("停电频率诊断工具")
    print("="*70)
    
    print("\n选择模式:")
    print("  1. 诊断单个州")
    print("  2. 诊断所有州")
    
    choice = input("\n输入选择 (1-2): ").strip()
    
    if choice == '1':
        # 单个州
        data_path = input("\n输入数据文件路径: ").strip()
        diagnose_single_state(data_path)
        
    elif choice == '2':
        # 所有州
        data_dir = input("\n输入数据目录路径: ").strip()
        diagnose_all_states(data_dir)
        
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
