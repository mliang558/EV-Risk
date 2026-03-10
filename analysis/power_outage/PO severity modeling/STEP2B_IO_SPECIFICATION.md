# Step 2B: Severity Modeling - Input/Output Specification

## 📥 **INPUTS**

### **1. Cleaned Outage Data (`cleaned_data_1117.csv`)**

**Required columns:**
```csv
fips, state, county, start_time, end_time, duration_min, mean_customers
```

**Column descriptions:**
- `fips`: 5-digit county FIPS code (string, e.g., "06037" for LA County)
- `state`: 2-letter state abbreviation (string, e.g., "CA", "TX")
- `county`: County name (string, e.g., "Los Angeles")
- `start_time`: Event start timestamp (datetime, e.g., "2020-01-15 14:30:00")
- `end_time`: Event end timestamp (datetime, optional)
- `duration_min`: Duration in minutes (float/int)
- `mean_customers`: Number of customers affected (float/int)

**Example rows:**
```csv
fips,state,county,start_time,end_time,duration_min,mean_customers
06037,CA,Los Angeles,2020-01-15 14:30:00,2020-01-15 18:45:00,255,12500
48201,TX,Harris,2020-02-10 08:20:00,2020-02-10 10:15:00,115,8300
```

---

### **2. County Population Data (`co-est2024-alldata.csv`)**

**Required columns:**
```csv
STATE, COUNTY, POPESTIMATE2022
```

**Column descriptions:**
- `STATE`: 2-digit state FIPS code (string, e.g., "06" for California)
- `COUNTY`: 3-digit county FIPS code (string, e.g., "037" for LA County)
- `POPESTIMATE2022`: Population estimate (integer)

**Example rows:**
```csv
STATE,COUNTY,POPESTIMATE2022
06,037,9829544
48,201,4783740
```

**Data source:**
- U.S. Census Bureau County Population Estimates
- Download: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html
- File: `co-est2024-alldata.csv`

---

## 🔧 **PROCESSING LOGIC**

### **Step 1: Data Merging**
```python
# Create full FIPS code
outage['fips'] = "06037"  # Already in data
population['fips'] = STATE + COUNTY  # "06" + "037" = "06037"

# Merge
merged = outage.merge(population[['fips', 'POPESTIMATE2022']], on='fips')

# Calculate customer impact percentage
merged['customers_pct'] = mean_customers / POPESTIMATE2022
```

---

### **Step 2: Severity Definition**

**Three types of severity:**

#### **A. Duration-Based**
```python
severe_duration = (duration_hr > threshold)
# Thresholds: 2h, 4h, 8h
```

#### **B. Impact-Based (使用Percentage!)**
```python
severe_impact = (customers_pct > threshold)
# Thresholds: 0.5%, 1%, 5%
```

#### **C. Combined (Default, 推荐)**
```python
severe_combined = (duration_hr > 2) AND (customers_pct > 0.01)
# Captures events that are BOTH long AND widespread
```

---

### **Step 3: Duration-Customer Interaction**

**Implicit interaction captured by:**

```
P(severe_combined) = P(duration > 2h AND customers_pct > 1%)

This is NOT:
  P(duration > 2h) × P(customers_pct > 1%)  ← 假设独立

This IS:
  P(jointly severe)  ← 捕捉相互作用

Interaction analysis:
  - Correlation(duration, customers_pct) by state
  - Scatter plots showing joint distribution
  - Identifies if longer events → more customers
```

---

### **Step 4: State-Level Aggregation**

For each state:
```python
n_events = total events in state
n_severe = events meeting severity criteria
P_severe_raw = n_severe / n_events  # Empirical probability
```

---

### **Step 5: Bayesian Hierarchical Model**

**Model structure:**
```
National level:
  μ_logit ~ Normal(data_mean, data_sd × 2)
  σ_state ~ HalfNormal(1)

State level:
  logit(p_severe[s]) ~ Normal(μ_logit, σ_state)
  
Observation:
  n_severe[s] ~ Binomial(n_events[s], p_severe[s])

Transform:
  p_severe[s] = sigmoid(logit(p_severe[s]))
```

**Benefits:**
- Shrinkage: Small states borrow strength
- Uncertainty quantification: 95% credible intervals
- Hierarchical pooling: Balances state vs national info

---

## 📤 **OUTPUTS**

### **Output Directory: `./step2b_severity_results/`**

---

### **1. Main Results: `state_severity_combined.csv`**

**Columns:**
```csv
state, n_events, mean_duration_hr, mean_customers_pct, 
p_severe_mean, p_severe_median, p_severe_sd, 
p_severe_ci_lower, p_severe_ci_upper, 
p_severe_raw, shrinkage
```

**Column descriptions:**
- `state`: State abbreviation
- `n_events`: Total events in state
- `mean_duration_hr`: Average duration (hours)
- `mean_customers_pct`: Average customer impact (as % of county)
- `p_severe_mean`: **Posterior mean of P(severe)** ← **主要结果**
- `p_severe_median`: Posterior median
- `p_severe_sd`: Posterior standard deviation
- `p_severe_ci_lower`: 2.5th percentile (lower 95% CI)
- `p_severe_ci_upper`: 97.5th percentile (upper 95% CI)
- `p_severe_raw`: Raw empirical probability (for comparison)
- `shrinkage`: How much Bayesian estimate differs from raw

**Example rows:**
```csv
state,n_events,mean_duration_hr,mean_customers_pct,p_severe_mean,p_severe_ci_lower,p_severe_ci_upper
CA,67821,2.34,0.0123,0.189,0.183,0.195
TX,85234,2.56,0.0145,0.234,0.228,0.240
FL,54123,3.12,0.0167,0.267,0.260,0.274
NY,48765,2.89,0.0134,0.221,0.214,0.228
```

**Interpretation:**
```
California: 
  - 67,821 events analyzed
  - Average duration: 2.34 hours
  - Average impact: 1.23% of county population
  - P(severe) = 18.9% [18.3%, 19.5%]
  
Meaning: In California, about 19% of outage events are severe
         (>2h duration AND >1% county impact)
```

---

### **2. Visualizations: `state_severity_combined.png`**

**6 panels:**

1. **Top 20 States Ranking** (with 95% CI error bars)
   - Y-axis: States (sorted by P(severe))
   - X-axis: P(severe event)
   
2. **Shrinkage Effect** (Raw vs Bayesian)
   - X-axis: Raw P(severe)
   - Y-axis: Bayesian P(severe)
   - Size: Number of events
   - Shows how small-sample states shrink toward mean

3. **Distribution of P(severe)**
   - Histogram across all states
   - Red line: National mean

4. **P(severe) vs Mean Duration**
   - Shows if longer average duration → higher severity
   - Labels: Top 5 states

5. **P(severe) vs Customer Impact**
   - Shows if higher customer impact → higher severity
   - Labels: Top 5 states

6. **Uncertainty vs Sample Size**
   - X-axis: Number of events (log scale)
   - Y-axis: 95% CI width
   - Shows more events → less uncertainty

---

### **3. Interaction Analysis: `duration_customer_interaction.png`**

**4 panels:**

1. **Event-Level Scatter** (log-log scale)
   - X-axis: Duration (hours)
   - Y-axis: Customer % of county
   - Shows overall correlation
   - Sample of 10,000 events

2. **Density Hexbin**
   - Shows concentration of events
   - Identifies typical duration-impact combinations

3. **State-Level Medians**
   - X-axis: Median duration per state
   - Y-axis: Median customer % per state
   - Labels: Top 5 severity states
   - Shows state-level patterns

4. **Correlation by State** (bar chart)
   - Each bar: One state's duration-customer correlation
   - Identifies which states show strong interaction

**Interpretation:**
```
Positive correlation (r > 0.3):
  → Longer events tend to affect more customers
  → Strong duration-customer interaction
  
Negative/no correlation (r < 0.1):
  → Duration and impact relatively independent
  → Weak interaction
```

---

### **4. State Correlations: `state_correlations.csv`**

**Columns:**
```csv
state, correlation
```

**Example:**
```csv
state,correlation
FL,0.42
TX,0.38
CA,0.31
NY,0.28
...
```

**Interpretation:**
- High correlation (>0.3): Strong duration-customer link
- Use this to identify states where interaction is important

---

### **5. Summary Report: `severity_summary.txt`**

**Plain text summary:**
```
STATE-LEVEL SEVERITY MODELING RESULTS
============================================================

Severity definition: combined
Total states: 51
Total events analyzed: 2,456,789

Summary statistics:
  Mean P(severe): 0.2145
  Std P(severe): 0.0487
  Min P(severe): 0.0823
  Max P(severe): 0.3912

Top 10 states by P(severe):
   1. Florida         : 0.2671 [0.2604, 0.2738]
   2. Texas           : 0.2343 [0.2283, 0.2403]
   3. New York        : 0.2213 [0.2145, 0.2281]
   ...

============================================================
INTERPRETATION:
  P(severe) = Probability that an outage event is severe
  Severe = Duration > 2h AND Customer impact > 1% of county
  Use this metric to assess state-level risk in combination with
  frequency (lambda) and network vulnerability.
```

---

## 🔗 **INTEGRATION WITH OTHER STEPS**

### **Combine with Step 2A (Frequency):**

```python
# Load Step 2A results
freq = pd.read_csv('fast_results/state_lambda_normalized.csv')

# Load Step 2B results  
sev = pd.read_csv('step2b_severity_results/state_severity_combined.csv')

# Merge
risk_data = freq[['state', 'lambda_per_county_mean']].merge(
    sev[['state', 'p_severe_mean']],
    on='state'
)

# Add your network vulnerability
risk_data = risk_data.merge(
    network_vuln[['state', 'vulnerability']],
    on='state'
)

# Calculate risk score
risk_data['risk_score'] = (
    normalize(risk_data['lambda_per_county_mean']) * 0.4 +  # Frequency
    risk_data['p_severe_mean'] * 0.3 +                      # Severity
    risk_data['vulnerability'] * 0.3                        # Network
)
```

**Result:**
```csv
state,lambda_per_county,p_severe,vulnerability,risk_score
CA,62.1,0.189,0.45,0.512
TX,55.2,0.234,0.52,0.534
FL,48.3,0.267,0.38,0.487
```

---

## 🚀 **USAGE EXAMPLES**

### **Example 1: Basic Run**
```bash
python run_step2b.py
```

### **Example 2: Command Line with Custom Paths**
```bash
python step2b_severity_final.py \
  cleaned_data_1117.csv \
  co-est2024-alldata.csv \
  --output-dir ./my_results
```

### **Example 3: Python API**
```python
from step2b_severity_final import StateSeverityModeler

modeler = StateSeverityModeler(
    data_path='cleaned_data_1117.csv',
    population_path='co-est2024-alldata.csv'
)

results = modeler.run_full_pipeline(output_dir='./results')

# Access results
print(results[['state', 'p_severe_mean']].head(10))
```

---

## ⏱️ **EXPECTED RUNTIME**

**On typical laptop (single core):**
- Data loading & merging: ~10 seconds
- Aggregation: ~5 seconds
- Interaction analysis: ~15 seconds
- Bayesian sampling (1000 draws): ~2-3 minutes
- Plotting: ~10 seconds

**Total: ~3-4 minutes**

---

## 📊 **DATA REQUIREMENTS**

### **Minimum viable dataset:**
- ✅ At least 10 states with data
- ✅ At least 100 events per state (for stable estimates)
- ✅ Population data available for most counties

### **Ideal dataset:**
- ✅ All 50 states
- ✅ 1000+ events per state
- ✅ 100% population coverage

### **What if missing population data?**
```python
# Option 1: Drop events without population (done automatically)
# Option 2: Use state-level average population (not recommended)
# Option 3: Impute from similar counties (advanced)
```

---

## ✅ **CHECKLIST BEFORE RUNNING**

- [ ] `cleaned_data_1117.csv` exists and has required columns
- [ ] `co-est2024-alldata.csv` downloaded from Census
- [ ] FIPS codes are 5-digit strings in outage data
- [ ] STATE/COUNTY codes are properly formatted in population data
- [ ] Python packages installed: `pandas`, `numpy`, `pymc`, `arviz`, `matplotlib`, `seaborn`
- [ ] Sufficient disk space (~100MB for outputs)

---

## ❓ **TROUBLESHOOTING**

### **Error: "Population merge resulted in NaN"**
```
Solution: Check FIPS code formatting
  - Outage data: fips should be 5-digit string "06037"
  - Population: STATE "06" + COUNTY "037" = "06037"
```

### **Error: "customers_pct > 1.0"**
```
This can happen if:
  - mean_customers is annual total (should be per-event)
  - Population is too small
  - Data quality issue

Code automatically clips to 1.0 (100%)
```

### **Warning: "Some states have <100 events"**
```
This is OK, Bayesian model handles small samples via shrinkage
Results will have wider credible intervals
```

---

## 📚 **REFERENCES**

### **Severity Definition:**
- IEEE 1366-2012: Standard for reliability indices
- NERC: Bulk electric system reliability standards

### **Statistical Methods:**
- Gelman & Hill (2007): Hierarchical modeling
- McElreath (2020): Statistical Rethinking (Bayesian methods)

### **Data Sources:**
- U.S. Census Bureau: Population estimates
- EIA-861: Utility service territories
