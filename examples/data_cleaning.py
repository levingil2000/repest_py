"""
Data Cleaning Example

Demonstrates data cleaning utilities including:
- Removing missing values and adjusting weights
- Flattening weighted data
- Combined cleaning and flattening
"""

import pandas as pd
import numpy as np
from repest import DataCleaner

# Create synthetic PISA-like data with missing values
np.random.seed(42)
n_students = 2000

data = pd.DataFrame({
    'CNT': np.random.choice(['USA', 'GBR', 'JPN'], n_students),
    'SCHOOLID': np.random.randint(1, 101, n_students),
    'W_FSTUWT': np.random.uniform(0.8, 2.0, n_students),
})

# Add variables with missing values
data['MATH_SCORE'] = np.random.normal(500, 100, n_students)
data.loc[np.random.random(n_students) < 0.10, 'MATH_SCORE'] = np.nan

data['READ_SCORE'] = np.random.normal(480, 95, n_students)
data.loc[np.random.random(n_students) < 0.12, 'READ_SCORE'] = np.nan

data['ESCS'] = np.random.normal(0, 1, n_students)
data.loc[np.random.random(n_students) < 0.05, 'ESCS'] = np.nan

data['GENDER'] = np.random.choice([0, 1], n_students)

# Add replicate weights
for rep in range(1, 81):
    data[f'W_FSTURWT{rep}'] = data['W_FSTUWT'] * np.random.uniform(0.85, 1.15, n_students)

print("="*80)
print("DATA CLEANING EXAMPLE")
print("="*80)

# Initialize cleaner
cleaner = DataCleaner(data, survey='PISA2015')

print("\n" + "="*80)
print("EXAMPLE 1: Summarize Missing Data")
print("="*80)

summary = cleaner.summarize_missingness(
    columns=['MATH_SCORE', 'READ_SCORE', 'ESCS'],
    by='CNT'
)
print("\nMissing Data by Country:")
print(summary.to_string(index=False))

print("\n" + "="*80)
print("EXAMPLE 2: Remove NA and Adjust Weights")
print("="*80)

cleaned_data = cleaner.remove_na_adjust_weights(
    columns=['MATH_SCORE', 'READ_SCORE', 'ESCS'],
    by='CNT',  # Adjust weights within each country
    adjust_replicates=True,  # Also adjust replicate weights
    verbose=True
)

print(f"\nOriginal data shape: {data.shape}")
print(f"Cleaned data shape: {cleaned_data.shape}")

# Verify weight totals are maintained
print("\nWeight totals by country (should be equal before/after):")
for cnt in data['CNT'].unique():
    orig_total = data[data['CNT'] == cnt]['W_FSTUWT'].sum()
    clean_total = cleaned_data[cleaned_data['CNT'] == cnt]['W_FSTUWT'].sum()
    print(f"  {cnt}: Original={orig_total:.2f}, Cleaned={clean_total:.2f}, "
          f"Diff={abs(orig_total-clean_total):.4f}")

print("\n" + "="*80)
print("EXAMPLE 3: Flatten Data with Weights")
print("="*80)

# Create smaller dataset for flattening demo
small_data = data.head(100).copy()
small_cleaner = DataCleaner(small_data, survey='PISA2015')

flattened_data = small_cleaner.flatten_with_weights(
    round_weights=True,
    verbose=True
)

print(f"\nOriginal data: {len(small_data)} rows")
print(f"Flattened data: {len(flattened_data)} rows")
print(f"\nFlattened data columns: {list(flattened_data.columns)[:5]}...")
print(f"(Weight columns removed: {cleaner.final_weight}, replicate weights)")

print("\n" + "="*80)
print("EXAMPLE 4: Combined Clean and Flatten")
print("="*80)

# Use small dataset for demo
flat_clean_data = small_cleaner.clean_and_flatten(
    columns=['MATH_SCORE', 'READ_SCORE', 'ESCS'],
    by='CNT',
    round_weights=True,
    max_rows=5000,  # Limit output size
    random_state=42,
    verbose=True
)

print(f"\nFinal data shape: {flat_clean_data.shape}")

# Now you can use standard unweighted analysis
print("\nExample unweighted analysis on flattened data:")
print(f"  Mean math score: {flat_clean_data['MATH_SCORE'].mean():.2f}")
print(f"  Mean read score: {flat_clean_data['READ_SCORE'].mean():.2f}")
print(f"  SD math score: {flat_clean_data['MATH_SCORE'].std():.2f}")

print("\n" + "="*80)
print("EXAMPLE 5: Get Complete Cases Only (No Weight Adjustment)")
print("="*80)

complete_cases = cleaner.get_complete_cases(
    columns=['MATH_SCORE', 'READ_SCORE', 'ESCS']
)

print(f"Complete cases: {len(complete_cases)} / {len(data)} "
      f"({100*len(complete_cases)/len(data):.1f}%)")

print("\n" + "="*80)
print("DATA CLEANING EXAMPLES COMPLETE")
print("="*80)