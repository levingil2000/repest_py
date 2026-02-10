"""
Basic Analysis Example

Demonstrates basic usage of RepEst for computing means, frequencies,
and summary statistics with replicate weights and plausible values.
"""

import pandas as pd
import numpy as np
from repest import RepEst, EstimationFunctions

# Create synthetic PISA-like data
np.random.seed(42)
n_students = 5000
n_countries = 3

data = pd.DataFrame({
    'CNT': np.random.choice(['USA', 'GBR', 'JPN'], n_students),
    'SCHOOLID': np.random.randint(1, 201, n_students),
    'GENDER': np.random.choice([0, 1], n_students),
    'ESCS': np.random.normal(0, 1, n_students),
    'W_FSTUWT': np.random.uniform(0.5, 2.5, n_students),
})

# Add plausible values (normally distributed around true score)
for pv in range(1, 11):
    true_math = 500 + 50 * data['ESCS'] + 10 * data['GENDER'] + np.random.normal(0, 80, n_students)
    data[f'PV{pv}MATH'] = true_math + np.random.normal(0, 20, n_students)
    
    true_read = 480 + 40 * data['ESCS'] - 5 * data['GENDER'] + np.random.normal(0, 85, n_students)
    data[f'PV{pv}READ'] = true_read + np.random.normal(0, 20, n_students)

# Add replicate weights
for rep in range(1, 81):
    data[f'W_FSTURWT{rep}'] = data['W_FSTUWT'] * np.random.uniform(0.8, 1.2, n_students)

print("="*80)
print("REPEST BASIC ANALYSIS EXAMPLE")
print("="*80)

# Initialize RepEst
rep = RepEst(
    data=data,
    survey='PISA2015',
    pv_vars=['PV@MATH', 'PV@READ']
)

print("\n1. Compute Mean Math Scores by Country")
print("-"*80)

results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH']),
    by='CNT',
    display=True
)

print("\n2. Compute Mean Math and Reading Scores by Country and Gender")
print("-"*80)

results = rep.estimate(
    func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH', 'PV1READ']),
    by=['CNT', 'GENDER'],
    display=True
)

print("\n3. Compute Gender Frequencies by Country")
print("-"*80)

results = rep.estimate(
    func=lambda d, w: EstimationFunctions.freq(d, w, 'GENDER', levels=[0, 1]),
    by='CNT',
    display=True
)

print("\n4. Compute Summary Statistics for Math Scores")
print("-"*80)

results = rep.estimate(
    func=lambda d, w: EstimationFunctions.summarize(
        d, w, ['PV1MATH'],
        stats=['mean', 'sd', 'p25', 'p50', 'p75']
    ),
    by='CNT',
    display=True
)

print("\n5. Compute Correlation between Math and Reading")
print("-"*80)

results = rep.estimate(
    func=lambda d, w: EstimationFunctions.corr(d, w, ['PV1MATH', 'PV1READ']),
    by='CNT',
    display=True
)

print("\n" + "="*80)
print("EXAMPLE COMPLETE")
print("="*80)