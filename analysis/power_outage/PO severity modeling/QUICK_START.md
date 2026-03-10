# 快速使用指南

## 文件说明
- `full_pop_2022.csv`: 你的停电数据文件
- `run_county_severity_analysis.py`: 主分析脚本（可直接运行）

## 运行方法

### 方法1: 如果数据文件在当前目录
```bash
python run_county_severity_analysis.py
```

### 方法2: 指定数据文件路径
```bash
python run_county_severity_analysis.py /path/to/full_pop_2022.csv
```

### 方法3: 指定输出目录
```bash
python run_county_severity_analysis.py full_pop_2022.csv ./my_results
```

## 输出结果

分析完成后会在 `./severity_results/` 目录（或你指定的目录）生成以下文件：

### CSV文件（数据表格）
1. **severity_definitions.csv** - 所有严重程度定义及其统计
2. **county_severity_full.csv** - 每个县的详细统计（推荐重点查看）
3. **state_severity_summary.csv** - 按州汇总的结果

### 文本报告
4. **severity_report.txt** - 易读的分析报告，包含：
   - 数据概览
   - 推荐的严重程度标准
   - 前20个高风险县

### 可视化图表
5. **severity_overview.png** - 严重程度定义概览（4个子图）
6. **severity_heatmap.png** - 组合条件热力图
7. **county_severity_analysis.png** - 县级详细分析（4个子图）

## 推荐的严重程度标准

脚本会自动计算以下4个推荐标准：

| 标准名称 | 时长阈值 | 客户%阈值 | 适用场景 |
|---------|---------|----------|---------|
| `severe_comb_d4_p0` | ≥4小时 | ≥0.5% | 🟢 宽松-早期预警 |
| `severe_comb_d12_p1` | ≥12小时 | ≥1% | 🟡 中等-日常监控 |
| `severe_comb_d24_p2` | ≥24小时 | ≥2% | 🟠 标准-重要事件 |
| `severe_comb_d48_p5` | ≥48小时 | ≥5% | 🔴 严格-重大事故 |

## 如何解读结果

### 1. 查看 county_severity_full.csv
这是最重要的输出文件，包含每个县的：
- `n_events`: 总停电事件数
- `duration_hr_mean`: 平均停电时长（小时）
- `customers_pct_mean`: 平均受影响客户百分比
- `severe_comb_d12_p1`: 符合"中等标准"的事件数
- `severe_comb_d12_p1_rate`: 严重事件发生率（0-1之间）

**重点关注**: 
- `severe_comb_d12_p1_rate > 0.3` 的县 → 高风险县
- `n_events` 很大的县 → 频发停电区域

### 2. 查看 severity_report.txt
这是人类可读的分析报告，直接打开查看即可。

### 3. 查看图表
- **severity_overview.png**: 了解整体情况
- **county_severity_analysis.png**: 查看县级风险分布

## 示例：在Python中进一步分析

```python
import pandas as pd

# 读取县级结果
df = pd.read_csv('./severity_results/county_severity_full.csv')

# 找出高风险县（使用中等标准）
high_risk = df[df['severe_comb_d12_p1_rate'] > 0.3].sort_values(
    'severe_comb_d12_p1_rate', ascending=False
)

print("高风险县（严重事件率 > 30%）:")
print(high_risk[['county', 'state', 'n_events', 'severe_comb_d12_p1_rate']])

# 按州统计
state_summary = df.groupby('state').agg({
    'n_events': 'sum',
    'severe_comb_d12_p1': 'sum'
}).reset_index()

state_summary['severe_rate'] = (
    state_summary['severe_comb_d12_p1'] / state_summary['n_events']
)

print("\n各州严重程度排名:")
print(state_summary.sort_values('severe_rate', ascending=False).head(10))
```

## 常见问题

**Q: 没有生成某些图表？**
A: 可能是因为数据量不足。脚本会自动过滤事件数<5的县。

**Q: 想要修改严重程度标准？**
A: 编辑 `run_county_severity_analysis.py` 文件，修改以下部分：
```python
# 找到这几行（大约第31-37行）
duration_thresholds = [4, 12, 24, 48]  # 修改时长阈值
customer_pct_thresholds = [0.005, 0.01, 0.02, 0.05]  # 修改百分比阈值
customer_abs_thresholds = [500, 1000, 2500, 5000]  # 修改绝对数阈值
```

**Q: 内存不足或运行太慢？**
A: 可以修改 `min_events` 参数来过滤更多低频县：
```python
# 在脚本中找到这行（大约第415行）
county_results = aggregate_to_county(df, severity_definitions, min_events=10)  # 改为10
```

## 需要帮助？
如果遇到问题，请检查：
1. Python版本 ≥ 3.7
2. 已安装必需的库：pandas, numpy, matplotlib, seaborn
3. 数据文件路径正确
4. 数据文件包含所需字段

安装依赖：
```bash
pip install pandas numpy matplotlib seaborn
```
