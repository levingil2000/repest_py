"""
Basic tests for RepEst package
"""

import numpy as np
import pandas as pd
import pytest
from repest import RepEst, EstimationFunctions, DataCleaner, SurveyParameters


def create_test_data(n=1000):
    """Create synthetic test data"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'CNT': np.random.choice(['USA', 'GBR'], n),
        'W_FSTUWT': np.random.uniform(0.5, 2.0, n),
        'GENDER': np.random.choice([0, 1], n),
        'MATH': np.random.normal(500, 100, n),
        'READ': np.random.normal(480, 95, n),
        'ESCS': np.random.normal(0, 1, n),
    })
    
    # Add replicate weights
    for i in range(1, 81):
        data[f'W_FSTURWT{i}'] = data['W_FSTUWT'] * np.random.uniform(0.9, 1.1, n)
    
    # Add plausible values
    for pv in range(1, 11):
        data[f'PV{pv}MATH'] = data['MATH'] + np.random.normal(0, 10, n)
        data[f'PV{pv}READ'] = data['READ'] + np.random.normal(0, 10, n)
    
    return data


class TestEstimationFunctions:
    """Test estimation functions"""
    
    def test_means(self):
        data = create_test_data(100)
        result = EstimationFunctions.means(data, 'W_FSTUWT', ['MATH', 'READ'])
        
        assert 'MATH_m' in result
        assert 'READ_m' in result
        assert isinstance(result['MATH_m'], (int, float))
        assert not np.isnan(result['MATH_m'])
    
    def test_freq(self):
        data = create_test_data(100)
        result = EstimationFunctions.freq(data, 'W_FSTUWT', 'GENDER', levels=[0, 1])
        
        assert 'GENDER_0' in result
        assert 'GENDER_1' in result
        # Frequencies should sum to ~100
        assert abs(result['GENDER_0'] + result['GENDER_1'] - 100) < 1
    
    def test_summarize(self):
        data = create_test_data(100)
        result = EstimationFunctions.summarize(
            data, 'W_FSTUWT', ['MATH'],
            stats=['mean', 'sd', 'p25', 'p50', 'p75']
        )
        
        assert 'MATH_mean' in result
        assert 'MATH_sd' in result
        assert 'MATH_p50' in result
        assert result['MATH_sd'] > 0
    
    def test_corr(self):
        data = create_test_data(100)
        result = EstimationFunctions.corr(data, 'W_FSTUWT', ['MATH', 'READ'])
        
        assert 'c_MATH_READ' in result
        assert -1 <= result['c_MATH_READ'] <= 1


class TestRepEst:
    """Test RepEst main class"""
    
    def test_initialization(self):
        data = create_test_data(100)
        rep = RepEst(data, survey='PISA2015', pv_vars=['PV@MATH'])
        
        assert rep.final_weight is not None
        assert len(rep.rep_weights) == 80
        assert rep.has_pv
        assert rep.n_pv == 10
    
    def test_estimate_no_pv(self):
        data = create_test_data(100)
        rep = RepEst(data, survey='PISA2015')  # No PVs
        
        results = rep.estimate(
            func=lambda d, w: EstimationFunctions.means(d, w, ['MATH']),
            display=False
        )
        
        assert not results.empty
        assert 'MATH_m_b' in results.columns
        assert 'MATH_m_se' in results.columns
    
    def test_estimate_by_group(self):
        data = create_test_data(100)
        rep = RepEst(data, survey='PISA2015')
        
        results = rep.estimate(
            func=lambda d, w: EstimationFunctions.means(d, w, ['MATH']),
            by='CNT',
            display=False
        )
        
        assert len(results) == 2  # USA and GBR
        assert 'CNT' in results.columns


class TestDataCleaner:
    """Test data cleaning functions"""
    
    def test_initialization(self):
        data = create_test_data(100)
        cleaner = DataCleaner(data, survey='PISA2015')
        
        assert cleaner.final_weight is not None
        assert len(cleaner.rep_weights) == 80
    
    def test_remove_na_adjust_weights(self):
        data = create_test_data(100)
        # Add some missing values
        data.loc[0:9, 'MATH'] = np.nan
        
        cleaner = DataCleaner(data, survey='PISA2015')
        cleaned = cleaner.remove_na_adjust_weights(
            columns=['MATH'],
            verbose=False
        )
        
        assert len(cleaned) == 90  # 10 removed
        assert cleaned['MATH'].notna().all()
        
        # Check weight totals maintained
        orig_total = data['W_FSTUWT'].sum()
        clean_total = cleaned['W_FSTUWT'].sum()
        assert abs(orig_total - clean_total) < 0.01
    
    def test_flatten_with_weights(self):
        data = create_test_data(20)  # Small for testing
        cleaner = DataCleaner(data, survey='PISA2015')
        
        flattened = cleaner.flatten_with_weights(
            round_weights=True,
            verbose=False
        )
        
        assert len(flattened) > len(data)
        assert 'W_FSTUWT' not in flattened.columns  # Weight removed
    
    def test_clean_and_flatten(self):
        data = create_test_data(20)
        data.loc[0:1, 'MATH'] = np.nan
        
        cleaner = DataCleaner(data, survey='PISA2015')
        result = cleaner.clean_and_flatten(
            columns=['MATH'],
            round_weights=True,
            verbose=False
        )
        
        assert result['MATH'].notna().all()
        assert 'W_FSTUWT' not in result.columns
    
    def test_summarize_missingness(self):
        data = create_test_data(100)
        data.loc[0:9, 'MATH'] = np.nan
        data.loc[0:14, 'READ'] = np.nan
        
        cleaner = DataCleaner(data, survey='PISA2015')
        summary = cleaner.summarize_missingness(columns=['MATH', 'READ'])
        
        assert len(summary) == 2
        assert 'variable' in summary.columns
        assert 'pct_missing' in summary.columns
        assert summary.loc[summary['variable'] == 'MATH', 'pct_missing'].values[0] == 10.0


def test_survey_configs():
    """Test that survey configurations are valid"""
    from repest import SURVEY_CONFIGS
    
    assert 'PISA' in SURVEY_CONFIGS
    assert 'PISA2015' in SURVEY_CONFIGS
    assert 'PIAAC' in SURVEY_CONFIGS
    
    pisa = SURVEY_CONFIGS['PISA']
    assert pisa.n_pv == 5
    assert pisa.n_reps == 80


if __name__ == '__main__':
    pytest.main([__file__, '-v'])