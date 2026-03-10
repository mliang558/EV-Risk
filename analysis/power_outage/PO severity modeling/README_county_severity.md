# 县级停电严重程度分析 - 使用说明

## 概述

这个脚本用于计算县级别的停电严重程度，使用多个阈值定义来识别严重停电事件。

## 输入数据要求

CSV文件必须包含以下字段：
- `fips`: 5位县代码
- `state`: 州名
- `county`: 县名
- `start_time`: 停电开始时间
- `end_time`: 停电结束时间（可选）
- `mean_customers`: 受影响的平均客户数
- `duration_min`: 停电时长（分钟）
- `POPESTIMATE2022`: 县人口数

## 严重程度定义

脚本提供4种严重程度定义方法：

### 方法1: 仅基于停电时长
- 默认阈值: 4小时、12小时、24小时、48小时
- 示例: `severe_dur_24h` = 停电≥24小时

### 方法2: 仅基于受影响客户百分比
- 默认阈值: 1%, 2%, 5%, 10%
- 示例: `severe_pct_5pct` = 受影响客户≥县人口的5%

### 方法3: 仅基于受影响客户绝对数
- 默认阈值: 500, 1000, 5000, 10000人
- 示例: `severe_abs_5000` = 受影响客户≥5000人

### 方法4: 组合条件（推荐）
- 同时满足时长和百分比条件
- 关键组合:
  - `severe_combined_d4_p1`: 4小时 且 1% (宽松，早期预警)
  - `severe_combined_d12_p2`: 12小时 且 2% (中等)
  - `severe_combined_d24_p5`: 24小时 且 5% (标准)
  - `severe_combined_d48_p10`: 48小时 且 10% (保守，高严重性)

## 使用方法

### 基本用法

```bash
python step2b_county_severity_fixed.py your_data.csv
```

### 自定义阈值

```bash
python step2b_county_severity_fixed.py your_data.csv \
    --duration-thresholds 6 24 72 \
    --pct-thresholds 0.01 0.05 0.15 \
    --abs-thresholds 1000 10000 50000 \
    --output-dir ./my_results
```

### 在Python代码中使用

```python
from step2b_county_severity_fixed import CountySeverityAnalyzer

# 创建分析器
analyzer = CountySeverityAnalyzer('your_data.csv')

# 加载数据
analyzer.load_data()

# 定义严重程度标准（可自定义阈值）
analyzer.define_severity_levels(
    duration_thresholds=[4, 12, 24, 48],  # 小时
    customer_pct_thresholds=[0.01, 0.02, 0.05, 0.10],  # 百分比
    customer_abs_thresholds=[500, 1000, 5000, 10000]  # 绝对数
)

# 聚合到县级
analyzer.aggregate_to_county()

# 生成报告和图表
analyzer.create_summary_report('./results')
analyzer.plot_results('./results')

# 或者一键运行所有步骤
analyzer.run_full_analysis('./results')
```

## 输出文件

分析完成后会生成以下文件：

### 1. severity_definitions_summary.csv
所有严重程度定义的汇总，包含：
- `method`: 方法类型
- `column`: 列名
- `duration_threshold`: 时长阈值
- `customer_pct_threshold`: 百分比阈值
- `customer_abs_threshold`: 绝对数阈值
- `n_severe`: 严重事件数
- `pct_severe`: 严重事件百分比

### 2. county_severity_summary.csv
县级别的详细统计，包含：
- 基本信息: fips, state, county, population
- 事件统计: n_events (事件总数)
- 时长统计: duration_hr_mean, duration_hr_median, duration_hr_max
- 客户影响统计: mean_customers_mean, customers_pct_mean等
- 各种严重程度定义下的事件数和比例

### 3. state_severity_summary.csv
州级别的聚合统计

### 4. severity_analysis_summary.txt
文本格式的分析报告，包括：
- 数据概览
- 所有严重程度定义的详细说明
- 建议使用的标准
- 前10个高风险县

### 5. severity_overview.png
包含4个子图的可视化：
- 不同定义下的严重事件数量
- 各定义覆盖的事件比例
- 县级事件数vs严重程度散点图
- 前20个县的事件数量条形图

### 6. severity_heatmap.png
组合条件下的严重事件分布热力图

## 结果解读

### 如何选择严重程度标准

1. **早期预警系统**: 使用宽松标准
   - `severe_combined_d4_p1`: 捕获更多潜在严重事件

2. **日常运维监控**: 使用中等标准
   - `severe_combined_d12_p2` 或 `severe_combined_d24_p5`

3. **重大事故分析**: 使用保守标准
   - `severe_combined_d48_p10`: 仅识别最严重事件

4. **综合评估**: 查看多个标准
   - 对比不同阈值下的结果，找到适合您需求的平衡点

### 县级结果使用

县级汇总表中的 `*_rate` 列表示该县严重事件的发生率：
- 高比率 + 高事件数 = 高风险县，需要重点关注
- 高比率 + 低事件数 = 可能偶发，但需警惕
- 低比率 + 高事件数 = 总体稳定，但要监控趋势

## 注意事项

1. **数据质量**: 确保人口数据准确，否则百分比计算会有误差
2. **事件数阈值**: 默认只分析至少5个事件的县，避免小样本偏差
3. **阈值选择**: 根据您的具体需求和电网特点调整阈值
4. **缺失值**: 自动删除关键字段缺失的记录

## 常见问题

**Q: 为什么有些县没有出现在结果中？**
A: 如果某个县的停电事件少于5次，会被过滤掉。可以在代码中修改这个阈值。

**Q: 如何调整严重程度标准？**
A: 使用命令行参数或在Python代码中修改阈值列表。

**Q: 客户百分比超过100%怎么办？**
A: 代码会自动将百分比限制在0-1之间（使用`.clip(0, 1)`）。

**Q: 可以添加更多严重程度定义吗？**
A: 可以，修改 `define_severity_levels()` 方法中的阈值列表即可。

## 示例分析流程

```python
# 1. 导入库
from step2b_county_severity_fixed import CountySeverityAnalyzer

# 2. 创建分析器
analyzer = CountySeverityAnalyzer('outage_data.csv')

# 3. 运行完整分析
analyzer.run_full_analysis('./results')

# 4. 查看结果
import pandas as pd

# 读取县级结果
counties = pd.read_csv('./results/county_severity_summary.csv')

# 查看使用中等标准的高风险县
high_risk = counties.nlargest(20, 'severe_combined_d24_p5_rate')
print(high_risk[['county', 'state', 'n_events', 'severe_combined_d24_p5_rate']])

# 查看州级汇总
states = pd.read_csv('./results/state_severity_summary.csv')
print(states.sort_values('n_events', ascending=False))
```

## 联系与支持

如有问题或建议，请查看生成的 `severity_analysis_summary.txt` 文件，其中包含详细的分析结果和解释。
