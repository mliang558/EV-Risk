# County vs State Level Modeling: Complete Guide

## 🎯 **核心问题：哪个方法更好？**

你的notebook (county-level) vs 我之前写的代码 (state-level)

---

## 📊 **两种方法对比**

### **方法1: State-Level Direct (之前的代码)**

```
数据 → 按state汇总count → Bayesian建模 → State λ
```

**优点：**
- ✅ 快速（50个州，几分钟）
- ✅ 简单，易理解

**缺点：**
- ❌ **不可比**：Texas高因为county多，不是risk高
- ❌ 丢失county信息
- ❌ Shrinkage效果小（数据太多）

**结果解读：**
```
Texas: λ = 14,000 events/year
Rhode Island: λ = 200 events/year

不能说：Texas更危险 ❌
只能说：Texas事件多（因为大）✓
```

---

### **方法2: County-Level + Aggregation (你的notebook)**

```
数据 → 按county汇总 → County-level Bayesian → 汇总到State
```

**优点：**
- ✅ **可比**：可以算per-county平均
- ✅ 保留细节信息
- ✅ Shrinkage有意义（小county借信息）

**缺点：**
- ❌ 慢（3000+ counties，30分钟+）
- ❌ 复杂，需要后处理

**结果解读：**
```
Texas: 
  - Total λ = 14,000 events/year (绝对风险)
  - Per-county λ = 55 events/county/year (相对风险)

Rhode Island:
  - Total λ = 200 events/year
  - Per-county λ = 40 events/county/year

结论：Texas的county平均比RI更频繁 (55 > 40) ✓
```

---

## 🎯 **推荐使用：方法2 (County-Level)**

### **原因：**

1. **科学严谨**
   - 可以公平比较州之间的风险
   - 不被州大小影响

2. **灵活性**
   - 得到两种指标：
     - Total λ：用于绝对风险评估
     - Per-county λ：用于州际比较

3. **符合研究目标**
   - 你要做risk score和attack simulation
   - 需要"真实风险"，不是"事件总数"

---

## 💻 **新代码的工作流程**

### **Step 1: County-Level建模**

```python
# 层级结构：
National: μ_national, σ_state, σ_county
  ↓
State[s]: μ_state[s] ~ Normal(μ_national, σ_state)
  ↓  
County[c]: log(λ_c) ~ Normal(μ_state[s_c], σ_county)
  ↓
Observation: Y_c ~ Poisson(λ_c × T_c)

# 例如：
Harris County, TX (c=1234):
  log(λ_1234) ~ Normal(μ_Texas, σ_county)
  
Los Angeles County, CA (c=5678):
  log(λ_5678) ~ Normal(μ_California, σ_county)
```

---

### **Step 2: 后验采样**

```python
# 得到每个county的后验分布
λ_county[i] ~ Posterior samples (1000 samples)

# 例如：
Harris County: [52.3, 54.1, 53.8, ..., 51.9]  # 1000个样本
LA County: [48.2, 49.5, 47.8, ..., 50.1]
```

---

### **Step 3: 汇总到State**

#### **A) Total Lambda (州总频率)**
```python
# Texas有254个counties
λ_Texas_total = Σ λ_county_i  (i ∈ Texas counties)

# 对每个posterior sample:
Sample 1: λ_TX[1] = 52.3 + 48.7 + ... (254个) = 13,890
Sample 2: λ_TX[2] = 54.1 + 49.2 + ... (254个) = 14,120
...
Sample 1000: λ_TX[1000] = 51.9 + 48.3 + ... = 13,950

# 结果：
λ_Texas_total ~ Posterior distribution
  Mean: 14,020 events/year
  95% CI: [13,750, 14,290]
```

#### **B) Per-County Lambda (平均每县频率)**
```python
# 平均
λ_Texas_per_county = mean(λ_county_i)  (i ∈ Texas counties)

# 对每个posterior sample:
Sample 1: mean([52.3, 48.7, ...]) = 55.2
Sample 2: mean([54.1, 49.2, ...]) = 55.8
...

# 结果：
λ_Texas_per_county ~ Posterior distribution
  Mean: 55.5 events/county/year
  95% CI: [52.1, 58.9]
```

---

## 📈 **结果对比示例**

### **Texas vs Rhode Island**

| State | # Counties | Total λ | Per-County λ | 解读 |
|-------|-----------|---------|--------------|------|
| **Texas** | 254 | 14,020 | 55.5 | 大州，总事件多 |
| **Rhode Island** | 5 | 200 | 40.0 | 小州，总事件少 |
| **比较** | - | TX高250% | TX高39% | **Per-county才可比！** |

**结论：**
- ✅ Texas的county确实平均比RI更频繁停电 (55.5 > 40.0)
- ✅ 但差距没有total lambda显示的那么夸张 (39% vs 250%)

---

## 🚀 **使用新代码**

### **快速测试（10分钟）：**
```bash
python run_county_model.py
```

会自动：
- 用500 samples（快速测试）
- 生成州级别对比图
- 保存两种lambda指标

### **完整分析（30分钟）：**
```bash
python county_to_state_modeling.py cleaned_data_1117.csv \
  --samples 2000 \
  --tune 1000 \
  --output-dir ./final_results
```

---

## 📊 **输出文件**

### **1. `state_level_from_county_model.csv`**
包含每个州的：
- `lambda_total_mean`: 总频率（绝对风险）
- `lambda_per_county_mean`: 每县平均（相对风险）
- 95% credible intervals

### **2. `state_comparison_county_model.png`**
4个图：
- Top 20 by total λ
- Top 20 by per-county λ
- Scatter: Total vs Per-County
- Distribution comparison

### **3. `state_comparison_summary.txt`**
文字总结，包括：
- Top 10 by total
- Top 10 by per-county
- 使用建议

---

## 🎯 **在你的研究中如何使用**

### **Step 2B - Severity Modeling:**
```python
# 用per-county lambda，因为可比
risk_score = f(lambda_per_county, P_severe, network_vuln)
```

### **Step 3 - Attack Simulation:**
```python
# 可以用两种：
# 1. Per-county lambda → 识别高风险州
# 2. Total lambda → 评估绝对影响

target_states = states.sort_values('lambda_per_county', ascending=False).head(10)
```

---

## ⚠️ **重要提醒**

### **1. 计算时间**
```
Counties: 3000+
Samples: 2000
时间：~30分钟（单核）

建议：
- 测试：500 samples (10分钟)
- 最终：2000 samples (30分钟)
```

### **2. 内存需求**
```
Posterior: 2000 samples × 3000 counties × 4 bytes ≈ 24 MB
应该没问题
```

### **3. 收敛检查**
```python
# 代码会自动检查R-hat
# 如果 R-hat > 1.01，需要更多samples
```

---

## 🔬 **理论细节**

### **为什么County-Level更科学？**

#### **问题：Simpson's Paradox（辛普森悖论）**

```
假设：
County A (在TX): λ = 60 events/year
County B (在TX): λ = 50 events/year
Average TX: λ = 55 events/year

County C (在RI): λ = 40 events/year
Average RI: λ = 40 events/year

State-level直接建模：
  TX州: 看到254个counties × 10年 = 2540 county-years
  RI州: 看到5个counties × 10年 = 50 county-years
  
模型学到：TX = 254×55 = 13,970 (但这不是"风险"，是"规模")

County-level建模：
  每个county独立建模，然后平均
  TX = mean(60, 50, ...) = 55
  RI = 40
  
结论：TX每个county确实更危险 (55 > 40) ✓
```

---

## 📝 **总结**

| 问题 | State-Level | County-Level |
|------|-------------|--------------|
| **能否州际比较？** | ❌ 不能 | ✅ 能（用per-county） |
| **计算速度** | ✅ 快 | ❌ 慢 |
| **Shrinkage价值** | ❌ 小 | ✅ 大 |
| **适合下游分析？** | ❌ 误导 | ✅ 合理 |
| **推荐使用** | 仅探索 | **最终分析** |

---

## 🎓 **数学公式对比**

### **State-Level:**
```
Y_s ~ Poisson(λ_s × T_s)

where:
  λ_s = state-level rate
  T_s = years observed
  
问题：λ_s 混合了"风险"和"规模"
```

### **County-Level:**
```
Y_c ~ Poisson(λ_c × T_c)
λ_c ~ Hierarchical(state, national)

State aggregation:
  λ_s^{total} = Σ λ_c  (绝对风险)
  λ_s^{avg} = mean(λ_c)  (相对风险)
  
优点：分离了"风险"和"规模"
```

---

## ✅ **行动建议**

1. **立即运行**：
   ```bash
   python run_county_model.py  # 快速测试
   ```

2. **检查结果**：
   - 看 `state_comparison_county_model.png`
   - 比较 total vs per-county top 10

3. **如果满意**：
   ```bash
   # 运行完整版本
   python county_to_state_modeling.py cleaned_data_1117.csv --samples 2000
   ```

4. **用于下游分析**：
   - Severity modeling: 用 `lambda_per_county_mean`
   - Risk scoring: 用 `lambda_per_county_mean`
   - Attack simulation: 两者都可以用

---

## 📞 **如有问题**

常见问题：
- Q: 为什么这么慢？
  A: 3000+ counties每个都要估计分布

- Q: 能不能更快？
  A: 可以用500 samples测试，或用VI (variational inference)

- Q: Total和Per-County哪个更重要？
  A: 看目标：绝对影响用total，风险比较用per-county
