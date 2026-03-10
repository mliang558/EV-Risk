"""
改进的脆弱性对比分析框架

解决问题：
1. 分离网络脆弱性和停电频率
2. 控制网络规模差异
3. 提供可解释的指标
"""

import pandas as pd
import numpy as np


def improved_vulnerability_analysis(results_csv: str):
    """
    改进的脆弱性分析
    
    分离三个维度：
    1. Network Vulnerability (网络结构脆弱性)
    2. Outage Exposure (停电暴露度)
    3. Overall Risk (综合风险)
    """
    
    print("\n" + "="*70)
    print("改进的脆弱性分析框架")
    print("="*70)
    
    # 读取数据
    df = pd.read_csv(results_csv)
    
    # ================================================================
    # 维度1: Network Vulnerability (网络结构脆弱性)
    # ================================================================
    
    print("\n" + "="*70)
    print("维度1: 网络结构脆弱性 (Network Vulnerability)")
    print("="*70)
    
    # 指标1a: 每次停电的平均损失
    df['Avg_Loss_Per_Event'] = df['Mean_Annual_Loss'] / df['Mean_N_Events']
    
    print("\n指标: Average Loss Per Event")
    print("含义: 每次停电事件平均造成多大损失")
    print("单位: (efficiency loss × hours) / event")
    print("解释: 控制了停电频率，纯粹反映网络结构的脆弱性")
    
    # 指标1b: 标准化到网络规模
    df['Loss_Per_Station'] = df['Mean_Annual_Loss'] / df['N_Stations']
    
    print("\n指标: Loss Per Station")
    print("含义: 每个充电站的年均损失")
    print("单位: (efficiency loss × hours) / station")
    print("解释: 控制了网络规模，反映单位基础设施的脆弱性")
    
    # 指标1c: 每次事件每个站点的损失
    df['Loss_Per_Event_Per_Station'] = (
        df['Avg_Loss_Per_Event'] / df['N_Stations']
    )
    
    print("\n指标: Loss Per Event Per Station")
    print("含义: 每次停电每个站点的平均损失")
    print("单位: (efficiency loss × hours) / (event × station)")
    print("解释: 最纯粹的网络脆弱性指标")
    
    # ================================================================
    # 维度2: Outage Exposure (停电暴露度)
    # ================================================================
    
    print("\n" + "="*70)
    print("维度2: 停电暴露度 (Outage Exposure)")
    print("="*70)
    
    # 指标2a: 年停电频率
    df['Annual_Outage_Rate'] = df['Mean_N_Events']
    
    print("\n指标: Annual Outage Rate")
    print("含义: 每年平均停电次数")
    print("单位: events/year")
    print("解释: 反映外部环境的停电风险")
    
    # 指标2b: 每日停电率
    df['Daily_Outage_Rate'] = df['Mean_N_Events'] / 365
    
    print("\n指标: Daily Outage Rate")
    print("含义: 每天平均停电次数")
    print("单位: events/day")
    print("解释: 更直观的频率指标")
    
    # ================================================================
    # 维度3: Overall Risk (综合风险)
    # ================================================================
    
    print("\n" + "="*70)
    print("维度3: 综合风险 (Overall Risk)")
    print("="*70)
    
    # 指标3a: 原始的总损失
    df['Total_Annual_Risk'] = df['Mean_Annual_Loss']
    
    print("\n指标: Total Annual Risk")
    print("含义: 网络脆弱性 × 停电频率")
    print("单位: efficiency loss × hours")
    print("解释: 综合风险，但难以解释单位")
    
    # 指标3b: 归一化风险
    df['Normalized_Risk'] = (
        (df['Mean_Annual_Loss'] - df['Mean_Annual_Loss'].min()) /
        (df['Mean_Annual_Loss'].max() - df['Mean_Annual_Loss'].min())
    )
    
    print("\n指标: Normalized Risk")
    print("含义: 相对风险评分")
    print("单位: [0, 1]")
    print("解释: 0=最低风险, 1=最高风险")
    
    # ================================================================
    # 排名对比
    # ================================================================
    
    print("\n" + "="*70)
    print("多维度排名对比")
    print("="*70)
    
    # 计算各维度排名
    df['Rank_Structure'] = df['Loss_Per_Event_Per_Station'].rank(ascending=False)
    df['Rank_Exposure'] = df['Annual_Outage_Rate'].rank(ascending=False)
    df['Rank_Overall'] = df['Total_Annual_Risk'].rank(ascending=False)
    
    # 综合评分
    df['Composite_Score'] = (
        0.4 * df['Rank_Structure'] + 
        0.3 * df['Rank_Exposure'] + 
        0.3 * df['Rank_Overall']
    )
    
    df['Final_Rank'] = df['Composite_Score'].rank(ascending=True)
    
    # 排序
    df = df.sort_values('Final_Rank')
    
    # ================================================================
    # 输出结果
    # ================================================================
    
    print("\n" + "="*70)
    print("Top 10 州 - 多维度对比")
    print("="*70)
    
    display_cols = [
        'State',
        'N_Stations',
        'Loss_Per_Event_Per_Station',  # 网络脆弱性
        'Annual_Outage_Rate',           # 停电频率
        'Total_Annual_Risk',            # 综合风险
        'Final_Rank'
    ]
    
    print("\n" + df[display_cols].head(10).to_string(index=False))
    
    # ================================================================
    # 分类分析
    # ================================================================
    
    print("\n" + "="*70)
    print("州分类")
    print("="*70)
    
    # 基于两个维度分类
    structure_median = df['Loss_Per_Event_Per_Station'].median()
    exposure_median = df['Annual_Outage_Rate'].median()
    
    def classify_state(row):
        high_vuln = row['Loss_Per_Event_Per_Station'] > structure_median
        high_expo = row['Annual_Outage_Rate'] > exposure_median
        
        if high_vuln and high_expo:
            return 'High-High (最危险)'
        elif high_vuln and not high_expo:
            return 'High-Low (结构脆弱)'
        elif not high_vuln and high_expo:
            return 'Low-High (频繁停电)'
        else:
            return 'Low-Low (相对安全)'
    
    df['Category'] = df.apply(classify_state, axis=1)
    
    print("\n州分类统计:")
    print(df['Category'].value_counts())
    
    print("\n各类别典型州:")
    for category in df['Category'].unique():
        states = df[df['Category'] == category]['State'].head(3).tolist()
        print(f"  {category}: {', '.join(states)}")
    
    # ================================================================
    # 可解释的对比
    # ================================================================
    
    print("\n" + "="*70)
    print("可解释的对比示例")
    print("="*70)
    
    # 取Top 5
    top5 = df.head(5)
    
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"\n{i}. {row['State']}:")
        print(f"   网络规模: {int(row['N_Stations'])} 充电站")
        print(f"   停电频率: {row['Annual_Outage_Rate']:.0f} 次/年 "
              f"(≈ {row['Daily_Outage_Rate']:.1f} 次/天)")
        print(f"   网络脆弱性: 每次停电每站损失 {row['Loss_Per_Event_Per_Station']:.6f}")
        print(f"   综合风险: {row['Total_Annual_Risk']:.2f}")
        print(f"   分类: {row['Category']}")
    
    # ================================================================
    # 保存结果
    # ================================================================
    
    import os
    os.makedirs('results', exist_ok=True)
    
    df.to_csv('results/improved_vulnerability_analysis.csv', index=False)
    print(f"\n✓ 完整分析已保存: results/improved_vulnerability_analysis.csv")
    
    return df


def create_recommendation_report(df: pd.DataFrame):
    """
    生成政策建议报告
    """
    
    print("\n" + "="*70)
    print("政策建议")
    print("="*70)
    
    report = """
基于多维度分析的政策建议：

1. 优先级分类：

   A. High-High (高脆弱性 + 高频率) 州：
      → 最紧急！需要立即行动
      → 建议：同时加固网络结构和应急响应能力
      
   B. High-Low (高脆弱性 + 低频率) 州：
      → 网络设计问题
      → 建议：优化网络拓扑，增加冗余连接
      
   C. Low-High (低脆弱性 + 高频率) 州：
      → 环境风险高
      → 建议：提升电网可靠性，建立应急预案
      
   D. Low-Low (双低) 州：
      → 相对安全
      → 建议：维持现状，定期评估

2. 投资优先级：
"""
    
    # High-High 州
    high_high = df[df['Category'] == 'High-High (最危险)']['State'].tolist()
    report += f"\n   第一优先级 (High-High): {', '.join(high_high[:5])}\n"
    
    # High-Low 州
    high_low = df[df['Category'] == 'High-Low (结构脆弱)']['State'].tolist()
    report += f"   第二优先级 (High-Low): {', '.join(high_low[:5])}\n"
    
    # Low-High 州
    low_high = df[df['Category'] == 'Low-High (频繁停电)']['State'].tolist()
    report += f"   第三优先级 (Low-High): {', '.join(low_high[:5])}\n"
    
    report += """
3. 具体措施：

   网络层面：
   - 识别关键节点（高betweenness）并加固
   - 增加跨县连接，减少孤立子网
   - 在高风险区域部署备用充电设施
   
   运营层面：
   - 建立实时监控系统
   - 制定停电应急响应预案
   - 与电网运营商协作提升可靠性
   
   政策层面：
   - 将充电基础设施纳入关键基础设施保护
   - 为高风险州提供联邦资金支持
   - 要求新建充电站满足抗灾标准
"""
    
    print(report)
    
    # 保存
    with open('results/policy_recommendations.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ 报告已保存: results/policy_recommendations.txt")


def main():
    """
    主函数
    """
    
    import os
    
    # 检查文件
    csv_path = 'results/all_states_ultrafast_summary.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        print(f"请先运行模拟生成结果")
        return
    
    # 分析
    df = improved_vulnerability_analysis(csv_path)
    
    # 生成建议
    create_recommendation_report(df)
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)


if __name__ == "__main__":
    main()
