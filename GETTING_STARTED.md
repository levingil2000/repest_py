# Getting Started with RepEst

Complete guide for installing and using RepEst for the first time.

## Installation

### Method 1: Install from GitHub (Recommended)

```bash
pip install git+https://github.com/yourusername/repest_py.git
```

### Method 2: Install for Development

```bash
# Clone the repository
git clone https://github.com/yourusername/repest_py.git
cd repest_py

# Install in editable mode
pip install -e .
```

### Method 3: Install with Optional Dependencies

```bash
# For regression support
pip install git+https://github.com/yourusername/repest_py.git[regression]

# For development tools
pip install git+https://github.com/yourusername/repest_py.git[dev]
```

## Verify Installation

```python
import repest
print(repest.__version__)

# Check available survey configurations
from repest import SURVEY_CONFIGS
print(list(SURVEY_CONFIGS.keys()))
```

## Quick Start: 5-Minute Tutorial

### 1. Load Your Data

```python
import pandas as pd
from repest import RepEst, EstimationFunctions

# Load PISA data (example)
data = pd.read_csv('PISA_2015_student.csv')
```

### 2. Initialize RepEst

```python
rep = RepEst(
    data=data,
    survey='PISA2015',  # Automatically loads survey configuration
    pv_vars=['PV@MATH', 'PV@READ', 'PV@SCIE']  # @ = placeholder for PV number
)
```

### 3. Compute Statistics

```python
# Mean math scores by country
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH']),
    by='CNT'
)

print(results)
```

Output:
```
================================================================================
REPEST ESTIMATION RESULTS
================================================================================
   CNT  PV1MATH_m_b  PV1MATH_m_se
   USA       475.23          2.84
   GBR       492.11          3.12
   ...
================================================================================
```

## Data Cleaning Tutorial

### Remove Missing Values and Adjust Weights

```python
from repest import DataCleaner

# Initialize cleaner
cleaner = DataCleaner(data, survey='PISA2015')

# Remove rows with missing values and adjust weights
cleaned_data = cleaner.remove_na_adjust_weights(
    columns=['PV1MATH', 'PV1READ', 'ESCS'],  # Variables to check
    by='CNT',  # Adjust weights within countries
    adjust_replicates=True
)
```

**What this does:**
1. Removes rows with ANY missing values in specified columns
2. Adjusts weights so population totals remain constant within each country
3. Formula: `new_weight = old_weight × (original_sum / remaining_sum)`

### Create Flat Dataset (Apply Weights)

```python
# Flatten data by replicating rows according to weights
flat_data = cleaner.flatten_with_weights(
    round_weights=True,  # Round to integer counts
    max_rows=100000  # Limit output size
)

# Now you can use standard unweighted methods
print(flat_data['PV1MATH'].mean())
```

**Warning:** Flattening can create very large datasets! Use `max_rows` to limit size.

### One-Step Clean and Flatten

```python
flat_clean_data = cleaner.clean_and_flatten(
    columns=['PV1MATH', 'ESCS'],
    by='CNT',
    round_weights=True,
    max_rows=50000
)
```

## Common Use Cases

### 1. Compute Means by Multiple Groups

```python
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH', 'PV1READ']),
    by=['CNT', 'GENDER']  # Multiple grouping variables
)
```

### 2. Frequencies and Percentages

```python
# Gender distribution by country
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.freq(d, w, 'GENDER', levels=[0, 1]),
    by='CNT'
)
```

### 3. Percentiles and Summary Statistics

```python
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.summarize(
        d, w, ['PV1MATH'],
        stats=['mean', 'sd', 'p25', 'p50', 'p75']
    ),
    by='CNT'
)
```

### 4. Correlations

```python
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.corr(d, w, ['PV1MATH', 'PV1READ']),
    by='CNT'
)
```

### 5. Quantile Analysis

```python
# Math achievement by SES quartiles
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.quantiletable(
        d, w,
        index_var='ESCS',  # Create quartiles from this
        outcome_var='PV1MATH',  # Summarize this
        n_quantiles=4
    ),
    by='CNT'
)
```

## Understanding How RepEst Works

RepEst uses **replicate weights** for variance estimation:

1. **Point Estimate**: Computes statistic using final weight
2. **Replicate Estimates**: Computes statistic using each replicate weight (80 times)
3. **Variance**: Calculated from deviation of replicates from point estimate
4. **Plausible Values**: If present, combines across PVs using Rubin's rules

Your estimation function is called once per weight:
- 1 time with final weight
- 80 times with replicate weights
- Total: 81 function calls

## Tips and Best Practices

### 1. Always Specify PV Variables

```python
# Good
rep = RepEst(data, survey='PISA2015', pv_vars=['PV@MATH', 'PV@READ'])

# Bad (will treat PV1MATH as a regular variable)
rep = RepEst(data, survey='PISA2015')
```

### 2. Check Weight Totals After Cleaning

```python
cleaned = cleaner.remove_na_adjust_weights(columns=['MATH'], by='CNT')

# Verify totals maintained
for cnt in data['CNT'].unique():
    orig = data[data['CNT']==cnt]['W_FSTUWT'].sum()
    new = cleaned[cleaned['CNT']==cnt]['W_FSTUWT'].sum()
    print(f"{cnt}: {orig:.0f} → {new:.0f}")
```

### 3. Use max_rows When Flattening

```python
# Good (prevents memory issues)
flat = cleaner.flatten_with_weights(max_rows=100000)

# Bad (could create millions of rows)
flat = cleaner.flatten_with_weights()
```

### 4. Set Random State for Reproducibility

```python
flat = cleaner.flatten_with_weights(random_state=42)
```

### 5. Save Cleaned Data for Reuse

```python
cleaned = cleaner.remove_na_adjust_weights(columns=['MATH', 'READ'], by='CNT')
cleaned.to_csv('pisa_cleaned.csv', index=False)
```

## Working with Different Surveys

RepEst supports multiple survey types:

```python
# PISA (2000-2012)
rep = RepEst(data, survey='PISA', pv_vars=['PV@MATH'])

# PISA 2015+
rep = RepEst(data, survey='PISA2015', pv_vars=['PV@MATH'])

# PIAAC
rep = RepEst(data, survey='PIAAC', pv_vars=['PVLIT@', 'PVNUM@'])

# TIMSS
rep = RepEst(data, survey='TIMSS', pv_vars=['BSMMAT@'])

# TALIS (no PVs)
rep = RepEst(data, survey='TALISTCH')
```

### Custom Survey Configuration

```python
from repest import SurveyParameters

custom = SurveyParameters(
    name='MyCustomSurvey',
    n_pv=5,
    final_weight='FINALWT',
    rep_weight_prefix='REPWT',
    variance_factor=1/20,
    n_reps=80
)

rep = RepEst(data, survey=custom, pv_vars=['PV@SCORE'])
```

## Troubleshooting

### Problem: "Weight not found in data"

**Solution:** Check weight column names. RepEst tries both lowercase and uppercase:

```python
# If your weights are uppercase
data.columns = [c.upper() for c in data.columns]
```

### Problem: "Too many rows when flattening"

**Solution:** Use `max_rows` parameter:

```python
flat = cleaner.flatten_with_weights(max_rows=50000)
```

### Problem: "Standard errors seem too small"

**Solution:** Make sure you're using plausible values correctly:

```python
# Correct
rep = RepEst(data, survey='PISA2015', pv_vars=['PV@MATH'])
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH'])
)

# Wrong (will underestimate SE)
rep = RepEst(data, survey='PISA2015')  # No pv_vars!
```

## Next Steps

1. **Read the full documentation**: See `examples/usage_guide.py`
2. **Run example scripts**: Try `examples/basic_analysis.py`
3. **Learn about custom functions**: See `examples/custom_functions.py`
4. **Explore data cleaning**: See `examples/data_cleaning.py`

## Getting Help

- **Documentation**: Check the `examples/` directory
- **Issues**: Report bugs on GitHub
- **Questions**: Open a GitHub discussion

## Citation

If you use RepEst in your research:

```bibtex
@software{repest_py,
  title = {RepEst: Replicate Estimation for Survey Data},
  author = {RepEst Contributors},
  year = {2024},
  url = {https://github.com/yourusername/repest_py}
}
```