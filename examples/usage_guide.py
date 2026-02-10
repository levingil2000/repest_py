"""
REPEST USAGE GUIDE
==================

Complete guide to using the RepEst package for survey data analysis.
"""

# =============================================================================
# 1. BASIC SETUP
# =============================================================================

import pandas as pd
import numpy as np
from repest import RepEst, EstimationFunctions, DataCleaner

# Load your data
data = pd.read_csv('pisa_data.csv')

# Initialize RepEst with a standard survey configuration
rep = RepEst(
    data=data,
    survey='PISA2015',  # Or 'PISA', 'PIAAC', 'TALIS', etc.
    pv_vars=['PV@MATH', 'PV@READ']  # @ is placeholder for PV number
)

# =============================================================================
# 2. COMPUTING MEANS
# =============================================================================

# Simple mean
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH'])
)

# Means by country
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH', 'PV1READ']),
    by='CNT'
)

# Means by country and gender
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH']),
    by=['CNT', 'GENDER']
)

# =============================================================================
# 3. FREQUENCIES AND PERCENTAGES
# =============================================================================

# Gender distribution by country
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.freq(d, w, 'GENDER', levels=[0, 1]),
    by='CNT'
)

# Multiple categories
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.freq(d, w, 'SES_QUARTILE', 
                                                levels=[1, 2, 3, 4]),
    by='CNT'
)

# =============================================================================
# 4. SUMMARY STATISTICS
# =============================================================================

# Full summary
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.summarize(
        d, w, ['PV1MATH'],
        stats=['mean', 'sd', 'min', 'max', 'p25', 'p50', 'p75', 'iqr']
    ),
    by='CNT'
)

# Just percentiles
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.summarize(
        d, w, ['PV1MATH'],
        stats=['p10', 'p25', 'p50', 'p75', 'p90']
    ),
    by='CNT'
)

# =============================================================================
# 5. CORRELATIONS
# =============================================================================

# Correlation between math and reading
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.corr(d, w, ['PV1MATH', 'PV1READ']),
    by='CNT'
)

# Correlation matrix for multiple variables
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.corr(
        d, w, ['PV1MATH', 'PV1READ', 'PV1SCIE', 'ESCS']
    ),
    by='CNT'
)

# Pairwise deletion (vs listwise)
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.corr(
        d, w, ['PV1MATH', 'PV1READ'], pairwise=True
    ),
    by='CNT'
)

# =============================================================================
# 6. QUANTILE TABLES
# =============================================================================

# Math achievement by SES quartiles
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.quantiletable(
        d, w,
        index_var='ESCS',  # Create quartiles based on this
        outcome_var='PV1MATH',  # Summarize this
        n_quantiles=4
    ),
    by='CNT'
)

# Decile analysis
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.quantiletable(
        d, w,
        index_var='ESCS',
        outcome_var='PV1MATH',
        n_quantiles=10,
        test=True  # Test difference between top and bottom
    ),
    by='CNT'
)

# =============================================================================
# 7. DATA CLEANING: REMOVE NA AND ADJUST WEIGHTS
# =============================================================================

# Initialize cleaner
cleaner = DataCleaner(data, survey='PISA2015')

# Remove rows with missing values and adjust weights
cleaned_data = cleaner.remove_na_adjust_weights(
    columns=['PV1MATH', 'PV1READ', 'ESCS'],  # Variables that must be complete
    by='CNT',  # Adjust weights within countries
    adjust_replicates=True,  # Also adjust replicate weights
    verbose=True  # Show summary
)

# Key features:
# - Removes rows with ANY missing values in specified columns
# - Adjusts weights so population totals are maintained within groups
# - Formula: new_weight = old_weight * (sum_original / sum_remaining)

# Verify weight totals are maintained
for cnt in data['CNT'].unique():
    orig_total = data[data['CNT'] == cnt]['W_FSTUWT'].sum()
    clean_total = cleaned_data[cleaned_data['CNT'] == cnt]['W_FSTUWT'].sum()
    print(f"{cnt}: Original={orig_total:.0f}, Cleaned={clean_total:.0f}")

# =============================================================================
# 8. DATA FLATTENING: CREATE UNWEIGHTED DATASET
# =============================================================================

# Flatten data by replicating rows according to weights
flattened_data = cleaner.flatten_with_weights(
    weight_column='W_FSTUWT',
    round_weights=True,  # Round to integer counts
    max_rows=100000,  # Limit output size
    random_state=42  # For reproducibility
)

# WARNING: This can create very large datasets!
# Each row appears approximately weight times
# Use max_rows to limit output size

# Now you can use standard unweighted methods
print(f"Mean math: {flattened_data['PV1MATH'].mean():.2f}")
print(f"SD math: {flattened_data['PV1MATH'].std():.2f}")

# =============================================================================
# 9. COMBINED CLEAN AND FLATTEN
# =============================================================================

# One-step workflow: clean then flatten
flat_clean_data = cleaner.clean_and_flatten(
    columns=['PV1MATH', 'PV1READ', 'ESCS'],  # Remove NA
    by='CNT',  # Adjust weights by country
    round_weights=True,
    max_rows=50000,
    random_state=42
)

# Result: Unweighted dataset with no missing values in key variables
# Ready for standard statistical analysis

# =============================================================================
# 10. MISSING DATA SUMMARY
# =============================================================================

# Summarize missing data patterns
summary = cleaner.summarize_missingness(
    columns=['PV1MATH', 'PV1READ', 'ESCS'],
    by='CNT'
)
print(summary)

# Get complete cases only (no weight adjustment)
complete_cases = cleaner.get_complete_cases(
    columns=['PV1MATH', 'PV1READ', 'ESCS']
)

# =============================================================================
# 11. OVER-VARIABLE COMPARISONS
# =============================================================================

# Compare math scores across different groups
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH']),
    by='CNT',
    over='GENDER'  # Compare across gender
)

# Test difference between groups
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH']),
    by='CNT',
    over=('GENDER', 'test')  # Compute gender difference
)

# =============================================================================
# 12. CUSTOM SURVEY CONFIGURATION
# =============================================================================

from repest import SurveyParameters

custom_survey = SurveyParameters(
    name='CustomSurvey',
    n_pv=5,  # Number of plausible values
    final_weight='WEIGHT',
    rep_weight_prefix='REP_WEIGHT',
    variance_factor=1/20,
    n_reps=80
)

rep = RepEst(data, survey=custom_survey, pv_vars=['PV@SCORE'])

# =============================================================================
# 13. ADVANCED: CUSTOM ESTIMATION FUNCTIONS
# =============================================================================

def custom_statistic(data, weight):
    """
    Custom estimation function
    
    Must take (data, weight) and return dict of statistics
    """
    weights = data[weight].values
    math_scores = data['PV1MATH'].values
    
    # Remove missing
    mask = ~(pd.isna(math_scores) | pd.isna(weights))
    scores = math_scores[mask]
    wts = weights[mask]
    
    # Compute custom statistic
    median = np.percentile(scores, 50)  # For simplicity, unweighted
    
    return {'math_median': median}

# Use with RepEst
results = rep.estimate(
    func=custom_statistic,
    by='CNT'
)

# =============================================================================
# 14. PERFORMANCE OPTIONS
# =============================================================================

# Fast mode: only compute replicates for first PV
# (Faster but may underestimate SEs slightly)
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH']),
    by='CNT',
    fast=True
)

# Coverage statistics: proportion of non-missing observations
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH'], coverage=True),
    by='CNT'
)

# =============================================================================
# 15. WORKING WITH RESULTS
# =============================================================================

# Results are returned as pandas DataFrames
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH', 'PV1READ']),
    by='CNT',
    display=False  # Don't print
)

# Access results
print(results['PV1MATH_m_b'])  # Point estimates
print(results['PV1MATH_m_se'])  # Standard errors

# Save to CSV
results.to_csv('results.csv', index=False)

# Compute t-statistics
results['PV1MATH_t'] = results['PV1MATH_m_b'] / results['PV1MATH_m_se']

# Filter countries
usa_results = results[results['CNT'] == 'USA']

# =============================================================================
# 16. BEST PRACTICES
# =============================================================================

# 1. Always specify pv_vars when working with plausible values
# 2. Use 'by' parameter to analyze within groups (e.g., countries)
# 3. Check weight totals after cleaning to ensure they're maintained
# 4. Use max_rows when flattening to avoid memory issues
# 5. Set random_state for reproducible results
# 6. Use fast=True for exploratory analysis, False for final results
# 7. Save intermediate results (cleaned data) for reuse
# 8. Document which variables were used for NA removal

# =============================================================================
# 17. COMMON WORKFLOWS
# =============================================================================

# Workflow 1: Standard Analysis with PVs
# --------------------------------------
rep = RepEst(data, survey='PISA2015', pv_vars=['PV@MATH'])
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH']),
    by='CNT'
)

# Workflow 2: Clean Then Analyze
# -------------------------------
cleaner = DataCleaner(data, survey='PISA2015')
cleaned = cleaner.remove_na_adjust_weights(
    columns=['PV1MATH', 'ESCS'],
    by='CNT'
)
rep = RepEst(cleaned, survey='PISA2015', pv_vars=['PV@MATH'])
results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH']),
    by='CNT'
)

# Workflow 3: Flatten for Standard Analysis
# ------------------------------------------
cleaner = DataCleaner(data, survey='PISA2015')
flat = cleaner.clean_and_flatten(
    columns=['PV1MATH', 'ESCS'],
    by='CNT',
    max_rows=50000
)
# Now use flat with sklearn, statsmodels, etc.
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(flat[['ESCS']], flat['PV1MATH'])

print("\n" + "="*80)
print("END OF USAGE GUIDE")
print("="*80)