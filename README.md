# RepEst - Replicate Estimation for Survey Data

Python implementation of the REPEST package for analyzing international large-scale assessment data (PISA, PIAAC, TALIS, etc.) with:

- **Replicate weights** (BRR, JK, Fay's method) for variance estimation
- **Plausible values** for measurement error
- **Complex survey designs**
- **Data cleaning utilities** for PISA-standard NA removal and weight adjustment

## Installation

### From GitHub

```bash
pip install git+https://github.com/levingil2000/repest_py.git
```

### For development

```bash
git clone https://github.com/levingil2000/repest_py.git
cd repest_py
pip install -e .
```

### With optional dependencies

```bash
# For regression support
pip install git+https://github.com/levingil2000/repest_py.git[regression]

# For development tools
pip install git+https://github.com/levingil2000/repest_py.git[dev]
```

## Quick Start

```python
import pandas as pd
from repest import RepEst, EstimationFunctions, DataCleaner

# Load your survey data
data = pd.read_csv('pisa_data.csv')

# Initialize RepEst with survey configuration
rep = RepEst(
    data=data,
    survey='PISA2015',
    pv_vars=['PV@MATH', 'PV@READ', 'PV@SCIE']
)

# Compute means with standard errors
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH', 'PV1READ']),
    by='CNT',  # By country
    display=True
)
```

## Data Cleaning Features

### Remove NA and Adjust Weights (PISA Standard)

```python
from repest import DataCleaner

# Clean data by removing rows with missing values in specified columns
# and adjusting weights to maintain population totals
cleaner = DataCleaner(data, survey='PISA2015')

cleaned_data = cleaner.remove_na_adjust_weights(
    columns=['PV1MATH', 'ESCS', 'ST004D01T'],  # Columns to check for NA
    by='CNT',  # Adjust weights within each country
    inplace=False  # Return new dataframe
)

print(f"Original N: {len(data)}")
print(f"Cleaned N: {len(cleaned_data)}")
```

### Create Flat Dataset (Apply Weights)

```python
# Expand dataset by replicating rows according to weights
# Useful for standard statistical packages that don't handle weights
flat_data = cleaner.flatten_with_weights(
    weight_column='W_FSTUWT',
    round_weights=True,  # Round to integer counts
    by='CNT'  # Flatten within groups
)

# Now you can use standard unweighted methods
print(flat_data['PV1MATH'].mean())
```

### Combined Cleaning and Flattening

```python
# One-step cleaning and flattening
flat_clean_data = cleaner.clean_and_flatten(
    columns=['PV1MATH', 'ESCS'],
    weight_column='W_FSTUWT',
    by='CNT',
    round_weights=True
)
```

## Estimation Functions

### Means

```python
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH', 'PV1READ']),
    by='CNT'
)
```

### Frequencies/Percentages

```python
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.freq(d, w, 'GENDER', levels=[0, 1]),
    by='CNT'
)
```

### Summary Statistics

```python
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.summarize(
        d, w, ['PV1MATH'],
        stats=['mean', 'sd', 'p25', 'p50', 'p75']
    ),
    by='CNT'
)
```

### Correlations

```python
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.corr(d, w, ['PV1MATH', 'PV1READ', 'ESCS']),
    by='CNT'
)
```

### Quantile Tables

```python
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.quantiletable(
        d, w,
        index_var='ESCS',
        outcome_var='PV1MATH',
        n_quantiles=4
    ),
    by='CNT'
)
```

## Supported Surveys

- **PISA** (2000-2012)
- **PISA2015** (2015+)
- **PIAAC**
- **TALIS** (Teacher and School)
- **TIMSS**
- **PIRLS**
- **ICCS**
- **ICILS**
- **IELS**
- **SSES** (2019 and 2023)

## Custom Survey Configuration

```python
from repest import SurveyParameters

custom_survey = SurveyParameters(
    name='CustomSurvey',
    n_pv=5,
    final_weight='WEIGHT',
    rep_weight_prefix='REP_WEIGHT',
    variance_factor=1/20,
    n_reps=80
)

rep = RepEst(data, survey=custom_survey)
```

## How Replicate Weights Work

RepEst uses the **replicate weight** method for variance estimation:

1. Computes statistic with final weight → **Point estimate**
2. Computes statistic with each replicate weight → **Replicate estimates**
3. Calculates variance from deviation of replicates from point estimate
4. If plausible values present: Combines across PVs using **Rubin's rules**

Your estimation function is called once per weight (final + all replicates), so it should:
- Accept `(data, weight)` parameters
- Return a dict of statistics
- Compute using only the provided weight variable

## Examples

See the `examples/` directory for detailed examples:
- `basic_analysis.py` - Simple means and frequencies
- `advanced_analysis.py` - Quantile tables and correlations
- `data_cleaning.py` - NA removal and weight adjustment
- `custom_functions.py` - Writing custom estimation functions

## License

MIT License

## Citation

If you use this package in your research, please cite:

```bibtex
@software{repest_py,
  title = {Repest for python},
  author = {kuya kevin},
  year = {2026},
  url = {https://github.com/levingil2000/repest_py.git}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- OECD (2009). PISA Data Analysis Manual: SPSS, Second Edition
- Mislevy, R.J., et al. (1992). Estimating Population Characteristics from Sparse Matrix Samples of Item Responses
- Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys