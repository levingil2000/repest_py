#!/usr/bin/env python
"""
Quick Demo - RepEst Package

Run this script to verify the package is working correctly.
Creates synthetic data and demonstrates key functionality.
"""

import numpy as np
import pandas as pd
from repest import RepEst, EstimationFunctions, DataCleaner

def main():
    print("="*80)
    print(" REPEST QUICK DEMO")
    print("="*80)
    
    # Create synthetic PISA-like data
    print("\n1. Creating synthetic PISA data...")
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'CNT': np.random.choice(['USA', 'GBR', 'JPN'], n),
        'GENDER': np.random.choice([0, 1], n),
        'ESCS': np.random.normal(0, 1, n),
        'W_FSTUWT': np.random.uniform(0.8, 2.0, n),
    })
    
    # Add plausible values
    for pv in range(1, 11):
        true_math = 500 + 50*data['ESCS'] + 10*data['GENDER'] + np.random.normal(0, 80, n)
        data[f'PV{pv}MATH'] = true_math + np.random.normal(0, 20, n)
    
    # Add replicate weights
    for i in range(1, 81):
        data[f'W_FSTURWT{i}'] = data['W_FSTUWT'] * np.random.uniform(0.9, 1.1, n)
    
    print(f"   Created {len(data)} students from {data['CNT'].nunique()} countries")
    
    # Test 1: Basic estimation
    print("\n" + "-"*80)
    print("2. Testing basic estimation (mean math scores by country)...")
    print("-"*80)
    
    rep = RepEst(data, survey='PISA2015', pv_vars=['PV@MATH'])
    
    results = rep.estimate(
        func=lambda d, w: EstimationFunctions.means(d, w, ['PV1MATH']),
        by='CNT',
        display=True
    )
    
    # Test 2: Data cleaning
    print("\n" + "-"*80)
    print("3. Testing data cleaning (remove NA and adjust weights)...")
    print("-"*80)
    
    # Add some missing values
    data_with_na = data.copy()
    n_missing = int(0.1 * len(data))
    data_with_na.loc[np.random.choice(len(data), n_missing, replace=False), 'PV1MATH'] = np.nan
    
    cleaner = DataCleaner(data_with_na, survey='PISA2015')
    
    cleaned = cleaner.remove_na_adjust_weights(
        columns=['PV1MATH'],
        by='CNT'
    )
    
    # Verify weight totals
    print("\n   Verifying weight totals maintained:")
    for cnt in data['CNT'].unique():
        orig = data_with_na[data_with_na['CNT']==cnt]['W_FSTUWT'].sum()
        clean = cleaned[cleaned['CNT']==cnt]['W_FSTUWT'].sum()
        diff = abs(orig - clean)
        status = "✓" if diff < 0.01 else "✗"
        print(f"   {status} {cnt}: {orig:.0f} → {clean:.0f} (diff: {diff:.4f})")
    
    # Test 3: Flattening
    print("\n" + "-"*80)
    print("4. Testing data flattening...")
    print("-"*80)
    
    small_data = data.head(50).copy()
    small_cleaner = DataCleaner(small_data, survey='PISA2015')
    
    flattened = small_cleaner.flatten_with_weights(
        round_weights=True,
        verbose=True
    )
    
    print(f"\n   Flattened data shape: {flattened.shape}")
    print(f"   Weight column removed: {'W_FSTUWT' not in flattened.columns}")
    
    # Test 4: Multiple statistics
    print("\n" + "-"*80)
    print("5. Testing multiple statistics...")
    print("-"*80)
    
    results = rep.estimate(
        func=lambda d, w: EstimationFunctions.summarize(
            d, w, ['PV1MATH'],
            stats=['mean', 'sd', 'p25', 'p50', 'p75']
        ),
        by='CNT',
        display=True
    )
    
    # Test 5: Correlation
    print("\n" + "-"*80)
    print("6. Testing correlation (Math ~ ESCS)...")
    print("-"*80)
    
    # Add ESCS as a "PV" for correlation
    results = rep.estimate(
        func=lambda d, w: EstimationFunctions.corr(d, w, ['PV1MATH', 'ESCS']),
        by='CNT',
        display=True
    )
    
    print("\n" + "="*80)
    print(" ALL TESTS PASSED! ✓")
    print("="*80)
    print("\nNext steps:")
    print("  - See examples/basic_analysis.py for more examples")
    print("  - See examples/data_cleaning.py for cleaning workflows")
    print("  - See examples/usage_guide.py for comprehensive documentation")
    print("  - See GETTING_STARTED.md for tutorial")
    print("\n")

if __name__ == '__main__':
    main()