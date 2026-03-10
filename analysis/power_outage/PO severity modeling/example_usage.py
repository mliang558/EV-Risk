#!/usr/bin/env python3
"""
县级停电严重程度分析 - 使用示例

这个脚本展示了如何使用 CountySeverityAnalyzer 进行分析
"""

from step2b_county_severity_fixed import CountySeverityAnalyzer
import pandas as pd

def example_basic_usage():
    """示例1: 基本使用"""
    print("="*80)
    print("示例1: 基本使用 - 运行完整分析")
    print("="*80)
    
    # 替换为你的数据文件路径
    data_file = 'your_outage_data.csv'
    
    # 创建分析器
    analyzer = CountySeverityAnalyzer(data_file)
    
    # 运行完整分析（一键完成所有步骤）
    analyzer.run_full_analysis('./results_basic')
    
    print("\n✓ 基本分析完成！")
    print("  查看 ./results_basic/ 目录获取结果")


def example_custom_thresholds():
    """示例2: 自定义阈值"""
    print("\n" + "="*80)
    print("示例2: 自定义严重程度阈值")
    print("="*80)
    
    data_file = 'your_outage_data.csv'
    
    # 创建分析器
    analyzer = CountySeverityAnalyzer(data_file)
    
    # 加载数据
    analyzer.load_data()
    
    # 自定义阈值
    analyzer.define_severity_levels(
        # 更严格的时长要求
        duration_thresholds=[6, 24, 72],  # 6小时、24小时、72小时
        
        # 更宽松的百分比要求  
        customer_pct_thresholds=[0.005, 0.01, 0.03],  # 0.5%, 1%, 3%
        
        # 自定义绝对数要求
        customer_abs_thresholds=[2000, 8000, 20000]  # 2千、8千、2万
    )
    
    # 继续分析
    analyzer.aggregate_to_county()
    analyzer.create_summary_report('./results_custom')
    analyzer.plot_results('./results_custom')
    
    print("\n✓ 自定义阈值分析完成！")


def example_step_by_step():
    """示例3: 分步骤执行并查看中间结果"""
    print("\n" + "="*80)
    print("示例3: 分步骤执行，查看中间结果")
    print("="*80)
    
    data_file = 'your_outage_data.csv'
    
    # 步骤1: 创建分析器并加载数据
    print("\n步骤1: 加载数据")
    analyzer = CountySeverityAnalyzer(data_file)
    analyzer.load_data()
    
    # 查看数据基本信息
    print(f"\n数据摘要:")
    print(analyzer.df[['fips', 'state', 'county', 'duration_hr', 
                       'mean_customers', 'customers_pct']].describe())
    
    # 步骤2: 定义严重程度
    print("\n步骤2: 定义严重程度标准")
    analyzer.define_severity_levels()
    
    # 查看定义
    print(f"\n创建了 {len(analyzer.severity_definitions)} 种严重程度定义")
    print("\n前5个定义:")
    print(analyzer.severity_def_df.head())
    
    # 步骤3: 聚合到县级
    print("\n步骤3: 聚合到县级")
    analyzer.aggregate_to_county()
    
    # 查看县级结果
    print(f"\n县级结果预览:")
    print(analyzer.county_results[['fips', 'state', 'county', 'n_events', 
                                   'duration_hr_mean', 'customers_pct_mean']].head(10))
    
    # 步骤4: 生成报告和图表
    print("\n步骤4: 生成报告和图表")
    analyzer.create_summary_report('./results_stepwise')
    analyzer.plot_results('./results_stepwise')
    
    print("\n✓ 分步骤分析完成！")


def example_analyze_specific_severity():
    """示例4: 分析特定的严重程度标准"""
    print("\n" + "="*80)
    print("示例4: 关注特定的严重程度标准")
    print("="*80)
    
    data_file = 'your_outage_data.csv'
    
    # 运行完整分析
    analyzer = CountySeverityAnalyzer(data_file)
    analyzer.run_full_analysis('./results_specific')
    
    # 读取结果
    county_results = pd.read_csv('./results_specific/county_severity_summary.csv')
    
    # 分析1: 使用中等标准（24小时 且 5%）
    print("\n使用标准: 24小时 且 5%客户受影响")
    print("-" * 80)
    
    if 'severe_combined_d24_p5_rate' in county_results.columns:
        # 找出高风险县（严重事件比例 > 30%）
        high_risk = county_results[
            county_results['severe_combined_d24_p5_rate'] > 0.3
        ].sort_values('severe_combined_d24_p5_rate', ascending=False)
        
        print(f"\n发现 {len(high_risk)} 个高风险县（严重事件率 > 30%）")
        print("\n前10个高风险县:")
        print(high_risk[['county', 'state', 'n_events', 
                        'severe_combined_d24_p5', 'severe_combined_d24_p5_rate']].head(10))
    
    # 分析2: 按州汇总
    print("\n\n按州汇总严重程度:")
    print("-" * 80)
    
    state_summary = county_results.groupby('state').agg({
        'n_events': 'sum',
        'severe_combined_d24_p5': 'sum'
    }).reset_index()
    
    state_summary['severe_rate'] = (
        state_summary['severe_combined_d24_p5'] / state_summary['n_events']
    )
    
    state_summary = state_summary.sort_values('severe_rate', ascending=False)
    print("\n前10个州:")
    print(state_summary.head(10))
    
    print("\n✓ 特定标准分析完成！")


def example_compare_definitions():
    """示例5: 比较不同严重程度定义"""
    print("\n" + "="*80)
    print("示例5: 比较不同严重程度定义")
    print("="*80)
    
    data_file = 'your_outage_data.csv'
    
    # 运行分析
    analyzer = CountySeverityAnalyzer(data_file)
    analyzer.run_full_analysis('./results_compare')
    
    # 读取定义汇总
    definitions = pd.read_csv('./results_compare/severity_definitions_summary.csv')
    
    # 比较不同方法
    print("\n按方法类型比较:")
    print("-" * 80)
    
    summary_by_method = definitions.groupby('method').agg({
        'n_severe': ['min', 'max', 'mean'],
        'pct_severe': ['min', 'max', 'mean']
    }).round(2)
    
    print(summary_by_method)
    
    # 找出最宽松和最严格的定义
    print("\n\n最宽松的定义（识别最多严重事件）:")
    most_lenient = definitions.nlargest(3, 'pct_severe')
    print(most_lenient[['column', 'method', 'n_severe', 'pct_severe']])
    
    print("\n最严格的定义（识别最少严重事件）:")
    most_strict = definitions.nsmallest(3, 'pct_severe')
    print(most_strict[['column', 'method', 'n_severe', 'pct_severe']])
    
    print("\n✓ 比较分析完成！")


def main():
    """主函数 - 选择要运行的示例"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     县级停电严重程度分析 - 使用示例                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

请选择要运行的示例（或修改此脚本中的 data_file 路径后运行）:

1. example_basic_usage()           - 基本使用
2. example_custom_thresholds()     - 自定义阈值
3. example_step_by_step()          - 分步骤执行
4. example_analyze_specific_severity() - 分析特定严重程度
5. example_compare_definitions()   - 比较不同定义

注意: 运行前请将 data_file 变量替换为你的实际数据文件路径！
    """)
    
    # 取消下面的注释来运行某个示例
    # example_basic_usage()
    # example_custom_thresholds()
    # example_step_by_step()
    # example_analyze_specific_severity()
    # example_compare_definitions()
    
    print("\n使用方法:")
    print("  1. 在脚本中将 'your_outage_data.csv' 替换为实际文件路径")
    print("  2. 取消注释想要运行的示例函数")
    print("  3. 运行: python example_usage.py")
    print("\n或者直接从命令行运行:")
    print("  python step2b_county_severity_fixed.py your_data.csv --output-dir ./results")


if __name__ == '__main__':
    main()
