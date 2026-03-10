# 🚀 Quick Fix Guide

## What Was Wrong?

Your code crashed with **TWO** errors:

### Error 1:
```
AttributeError: 'NoneType' object has no attribute 'head'
```
**Reason:** The code tried to use `self.state_results` before it was created.

### Error 2:
```
AttributeError: module 'scipy.stats' has no attribute 'expit'
```
**Reason:** The `expit` function is in `scipy.special`, not `scipy.stats`.

---

## How to Fix It

### Option 1: Use the Fixed File (Recommended)

1. **Replace your current file:**
   ```bash
   # Back up your original
   cp step2b_severity_final.py step2b_severity_final.py.backup
   
   # Use the fixed version
   cp step2b_severity_final.py step2b_severity_final.py
   ```

2. **Run it:**
   ```bash
   python run_step2b.py
   ```

### Option 2: Manual Fix (If You Want to Understand)

**Fix #1 - Import Error (Line ~21):**

Add to imports:
```python
from scipy.special import expit, logit  # Add this line
```

**Fix #2 - Function Call (Line ~276):**

Change:
```python
# OLD
print(f"    Implies p ~ {stats.expit(mu_logit_data):.3f} ± ...")

# NEW
print(f"    Implies p ~ {expit(mu_logit_data):.3f} ± ...")
```

**Fix #3 - Plot Labeling (Lines ~469-476):**

Change:
```python
# Line 469-476 (OLD - BROKEN)
for _, row in self.state_results.head(5).iterrows():
    state_data = state_medians[state_medians['state'] == row['state']]
    if len(state_data) > 0:
        ax.annotate(row['state'],
                  (state_data['duration_hr'].values[0], 
                   state_data['customers_pct'].values[0]),
                  fontsize=8, alpha=0.7)
```

**To this:**
```python
# Line 469-476 (NEW - FIXED)
top_states = corr_df.head(5)['state'].values
for state in top_states:
    state_data = state_medians[state_medians['state'] == state]
    if len(state_data) > 0:
        ax.annotate(state,
                  (state_data['duration_hr'].values[0], 
                   state_data['customers_pct'].values[0]),
                  fontsize=8, alpha=0.7)
```

Save and run!

---

## What Changed?

### Fix #1: Import Module
- **Before:** Tried to use `scipy.stats.expit` (doesn't exist)
- **After:** Use `scipy.special.expit` (correct module)

### Fix #2: Plot Labels  
- **Before:** Tried to label "top 5 states by severity probability" (data not ready yet)
- **After:** Label "top 5 states by duration-customer correlation" (data available + more meaningful)

---

## Verify It Works

After running, you should see:

```bash
✓ Saved interaction analysis: ./step2b_severity_results/duration_customer_interaction.png
✓ Saved correlations: ./step2b_severity_results/state_correlations.csv

[5/7] Building Bayesian hierarchical model...
  ✓ Model compiled successfully
  
[6/7] Sampling posterior...
  ✓ Sampling complete: 4 chains, 1000 draws each
  
... (continues to completion)
```

---

## Expected Outputs

After successful run, you'll have:

```
./step2b_severity_results/
├── state_severity_combined.csv          ← Main results
├── state_severity_combined.png          ← 6-panel visualization  
├── duration_customer_interaction.png    ← Relationship analysis
├── state_correlations.csv               ← State-level correlations
└── severity_summary.txt                 ← Text summary
```

---

## Still Having Issues?

### Issue 1: PyTensor Warning
```
WARNING: g++ not detected! PyTensor will be unable to compile C-implementations
```

**Fix:**
```bash
conda install gxx
# or
conda install -c conda-forge gxx_linux-64  # Linux
conda install -c conda-forge gxx_osx-64    # Mac
```

This is a performance warning, not critical. Code will still work, just slower.

### Issue 2: Font Warning
```
qt.qpa.fonts: Unable to open default EUDC font: "EUDC.TTE"
```

**Fix:** Ignore it! This is a Windows font warning and doesn't affect output.

### Issue 3: Different Error

Send me:
1. The full error message
2. The line number where it crashes
3. Your Python version (`python --version`)

---

## Questions?

**Q: Will this change my results?**
A: No! The bug was only in the plotting code, not in the calculations. Your severity probabilities will be identical.

**Q: Why is the new version better?**
A: The plot shows duration vs customer impact, so labeling states with **high correlation** between these variables makes more sense than labeling by overall severity.

**Q: Can I use the old labeling method?**
A: Yes, but you'd need to reorder the pipeline (see detailed report). The current fix is simpler and more meaningful.

---

Good luck! 🎉
