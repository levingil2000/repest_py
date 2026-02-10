"""
REPEST - Replicate Estimation for Survey Data with Plausible Values

Python implementation of the REPEST Stata package for analyzing international
large-scale assessment data (PISA, PIAAC, TALIS, etc.) with:
- Replicate weights (BRR, JK, Fay's method)
- Plausible values
- Complex survey designs

Author: Python port of original Stata package
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Tuple, Callable
import warnings
from dataclasses import dataclass
from scipy import stats
import re


@dataclass
class SurveyParameters:
    """Survey-specific parameters for variance estimation"""
    name: str
    n_pv: int  # Number of plausible values (0 if none)
    final_weight: str  # Final weight variable name
    rep_weight_prefix: str  # Replicate weight prefix
    variance_factor: float  # Variance adjustment factor
    n_reps: int  # Number of replicates
    groupflag_var: Optional[str] = None  # Variable for flagging
    keep_vars: Optional[List[str]] = None  # Additional variables to keep


# Survey configurations
SURVEY_CONFIGS = {
    'PISA': SurveyParameters(
        name='PISA',
        n_pv=5,
        final_weight='w_fstuwt',
        rep_weight_prefix='w_fstr',
        variance_factor=1/20,
        n_reps=80,
        groupflag_var='schoolid',
        keep_vars=['cnt', 'schoolid']
    ),
    'PISA2015': SurveyParameters(
        name='PISA2015',
        n_pv=10,
        final_weight='w_fstuwt',
        rep_weight_prefix='w_fsturwt',
        variance_factor=1/20,
        n_reps=80,
        groupflag_var='cntschid',
        keep_vars=['cnt', 'cntschid']
    ),
    'PIAAC': SurveyParameters(
        name='PIAAC',
        n_pv=10,
        final_weight='spfwt0',
        rep_weight_prefix='spfwt',
        variance_factor=None,  # Computed dynamically
        n_reps=80,
        keep_vars=['vemethodn', 'venreps', 'vefayfac']
    ),
    'TALISTCH': SurveyParameters(
        name='TALISTCH',
        n_pv=0,
        final_weight='tchwgt',
        rep_weight_prefix='trwgt',
        variance_factor=1/25,
        n_reps=100,
        groupflag_var='idschool',
        keep_vars=['cntry', 'idschool']
    ),
    'TALISSCH': SurveyParameters(
        name='TALISSCH',
        n_pv=0,
        final_weight='schwgt',
        rep_weight_prefix='srwgt',
        variance_factor=1/25,
        n_reps=100
    ),
    'SSES': SurveyParameters(
        name='SSES',
        n_pv=0,
        final_weight='WT2019',
        rep_weight_prefix='rwgt',
        variance_factor=1/2,
        n_reps=76,
        groupflag_var='SchID',
        keep_vars=['SchID']
    ),
    'SSES2023': SurveyParameters(
        name='SSES2023',
        n_pv=0,
        final_weight='WT2023',
        rep_weight_prefix='rwgt',
        variance_factor=1/20,
        n_reps=80,
        groupflag_var='SchID',
        keep_vars=['SchID']
    ),
    'TIMSS': SurveyParameters(
        name='TIMSS',
        n_pv=5,
        final_weight='WGT',
        rep_weight_prefix='JR',
        variance_factor=1/2,
        n_reps=150
    ),
    'PIRLS': SurveyParameters(
        name='PIRLS',
        n_pv=5,
        final_weight='WGT',
        rep_weight_prefix='JR',
        variance_factor=1/2,
        n_reps=150
    ),
    'ICCS': SurveyParameters(
        name='ICCS',
        n_pv=5,
        final_weight='TOTWGTS',
        rep_weight_prefix='SRWGT',
        variance_factor=1,
        n_reps=75
    ),
    'ICILS': SurveyParameters(
        name='ICILS',
        n_pv=5,
        final_weight='TOTWGTS',
        rep_weight_prefix='SRWGT',
        variance_factor=1,
        n_reps=75
    ),
    'IELS': SurveyParameters(
        name='IELS',
        n_pv=5,
        final_weight='CHILDWGT',
        rep_weight_prefix='SRWGT',
        variance_factor=1/23,
        n_reps=92,
        groupflag_var='IDCENTRE',
        keep_vars=['IDCENTRE']
    ),
}


class RepEst:
    """
    Main class for replicate estimation with plausible values
    
    Performs statistical analysis on survey data with:
    - Replicate weights for variance estimation (BRR/JK with Fay's method)
    - Plausible values for measurement error
    - By-group analysis
    - Over-variable comparisons
    
    IMPORTANT: How Replicate Weights Work
    -------------------------------------
    Estimation functions (like EstimationFunctions.means, .corr, etc.) compute
    statistics using a SINGLE weight variable at a time. RepEst then:
    
    1. Calls your function with the final weight → Point estimate
    2. Calls your function with each replicate weight → Replicate estimates
    3. Computes variance from deviation of replicates from point estimate
    4. If PVs present: Combines across PVs using Rubin's rules
    
    Example workflow:
        def my_mean(data, weight):
            return EstimationFunctions.means(data, weight, ['math'])
        
        results = repest.estimate(func=my_mean)
        # Internally calls my_mean() 81 times (1 final + 80 replicates)
        # Returns: {'math_m_b': 500.5, 'math_m_se': 2.3}
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 survey: Union[str, SurveyParameters],
                 pv_vars: Optional[List[str]] = None):
        """
        Initialize RepEst analyzer
        
        Parameters
        ----------
        data : pd.DataFrame
            Survey data
        survey : str or SurveyParameters
            Survey name (e.g., 'PISA', 'PIAAC') or custom parameters
        pv_vars : list of str, optional
            Plausible value variable patterns (use @ as placeholder for number)
            Example: ['PV@MATH', 'PV@READ']
        """
        self.data = data.copy()
        
        # Set up survey parameters
        if isinstance(survey, str):
            if survey.upper() not in SURVEY_CONFIGS:
                raise ValueError(f"Unknown survey: {survey}. Available: {list(SURVEY_CONFIGS.keys())}")
            self.params = SURVEY_CONFIGS[survey.upper()]
        else:
            self.params = survey
            
        self.pv_vars = pv_vars or []
        
        # Determine if we have PVs
        self.has_pv = len(self.pv_vars) > 0
        self.n_pv = self.params.n_pv if self.has_pv else 1
        
        # Get weight variable names
        self._setup_weights()
        
    def _setup_weights(self):
        """Setup final and replicate weight variable names"""
        self.final_weight = self.params.final_weight
        self.rep_weights = [
            f"{self.params.rep_weight_prefix}{i}" 
            for i in range(1, self.params.n_reps + 1)
        ]
        
        # Check if weights exist (try both cases)
        if self.final_weight not in self.data.columns:
            # Try uppercase
            final_upper = self.final_weight.upper()
            if final_upper in self.data.columns:
                self.final_weight = final_upper
                self.rep_weights = [w.upper() for w in self.rep_weights]
            else:
                raise ValueError(f"Final weight '{self.final_weight}' not found in data")
    
    def _expand_pv_vars(self, pv_index: int) -> List[str]:
        """Expand PV variable patterns for a specific PV index"""
        expanded = []
        for pattern in self.pv_vars:
            expanded.append(pattern.replace('@', str(pv_index)))
        return expanded
    
    def _get_variance_factor(self, subset_data: Optional[pd.DataFrame] = None) -> float:
        """
        Get variance factor, computing dynamically for PIAAC if needed
        
        Parameters
        ----------
        subset_data : pd.DataFrame, optional
            Data subset for computing variance factor
            
        Returns
        -------
        float
            Variance adjustment factor
        """
        if self.params.variance_factor is not None:
            return self.params.variance_factor
        
        # PIAAC special case - compute from metadata
        if self.params.name == 'PIAAC':
            if subset_data is None:
                subset_data = self.data
            
            return self._compute_piaac_variance_factor(subset_data)
        
        return 1.0
    
    def _compute_piaac_variance_factor(self, data: pd.DataFrame) -> float:
        """
        Compute PIAAC variance factor based on methodology
        
        PIAAC uses different variance estimation methods:
        - JK1 (vemethodn=1): (venreps-1)/venreps
        - JK2 (vemethodn=2): 1
        - Fay (vemethodn=4): 1/(venreps*(1-vefayfac)^2)
        """
        # Try lowercase first
        method_var = 'vemethodn' if 'vemethodn' in data.columns else 'VEMETHODN'
        reps_var = 'venreps' if 'venreps' in data.columns else 'VENREPS'
        fay_var = 'vefayfac' if 'vefayfac' in data.columns else 'VEFAYFAC'
        
        # Compute variance factor for each observation
        varfac = np.zeros(len(data))
        
        method = data[method_var].values
        reps = data[reps_var].values
        
        # JK1
        mask_jk1 = (method == 1)
        varfac[mask_jk1] = (reps[mask_jk1] - 1) / reps[mask_jk1]
        
        # JK2
        mask_jk2 = (method == 2)
        varfac[mask_jk2] = 1.0
        
        # Fay
        mask_fay = (method == 4)
        if fay_var in data.columns:
            fay = data[fay_var].values
            # Handle missing Fay factors
            fay_missing = pd.isna(fay)
            varfac[mask_fay & ~fay_missing] = 1 / (reps[mask_fay & ~fay_missing] * 
                                                    (1 - fay[mask_fay & ~fay_missing])**2)
            varfac[mask_fay & fay_missing] = (reps[mask_fay & fay_missing] - 1) / reps[mask_fay & fay_missing]
        
        # Check if all same (single country or homogeneous pool)
        if varfac.min() == varfac.max():
            return float(varfac[0])
        else:
            # Weighted average if pooling countries
            warnings.warn(
                "VEMETHODN is not constant. Using weighted average of variance factors. "
                "Results may be incorrect for pooled multi-country analysis."
            )
            weights = data[self.final_weight].values
            return float(np.average(varfac, weights=weights))
    
    def estimate(self,
                 func: Callable,
                 by: Optional[Union[str, List[str]]] = None,
                 over: Optional[Union[str, Tuple[str, str]]] = None,
                 display: bool = True,
                 fast: bool = False,
                 flag: bool = False,
                 coverage: bool = False,
                 **kwargs) -> pd.DataFrame:
        """
        Estimate statistics with replicate weights and plausible values
        
        Parameters
        ----------
        func : callable
            Estimation function that takes (data, weight) and returns dict of statistics
        by : str or list of str, optional
            Variable(s) to group by
        over : str or tuple, optional
            Variable for comparisons. If tuple: (var, 'test') to compute differences
        display : bool, default True
            Whether to display results
        fast : bool, default False
            Fast mode: compute replicates only for first PV
        flag : bool, default False
            Add flags for insufficient sample sizes
        coverage : bool, default False
            Compute coverage statistics
        **kwargs
            Additional arguments passed to estimation function
            
        Returns
        -------
        pd.DataFrame
            Results with coefficients and standard errors
        """
        results = []
        
        # Parse by variable
        by_vars = [by] if isinstance(by, str) else (by or [])
        by_levels = self._get_by_levels(by_vars)
        
        # Parse over variable
        over_var, over_test = self._parse_over(over)
        over_levels = self._get_over_levels(over_var)
        
        # Loop over by-levels
        for by_level in by_levels:
            by_subset = self._filter_by_level(by_vars, by_level)
            
            # Loop over over-levels
            level_results = []
            for over_level in over_levels:
                over_subset = self._filter_over_level(by_subset, over_var, over_level)
                
                if len(over_subset) == 0:
                    continue
                
                # Estimate with PVs and replicates
                coefs, vcov, stat_names = self._estimate_with_replicates(
                    over_subset, func, fast=fast, **kwargs
                )
                
                if coefs is None:
                    continue
                
                # Compute standard errors
                se = np.sqrt(np.diag(vcov))
                
                # Build result row
                result = self._build_result_row(
                    coefs, se, by_vars, by_level, over_var, over_level, stat_names
                )
                level_results.append(result)
            
            # Compute over-test differences if requested
            if over_test and len(level_results) >= 2:
                diff_result = self._compute_over_difference(
                    level_results, over_test, by_vars, by_level
                )
                level_results.append(diff_result)
            
            results.extend(level_results)
        
        # Combine results
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # Reshape if over variable present
        if over_var is not None:
            results_df = self._reshape_over_results(results_df, over_var)
        
        if display:
            self._display_results(results_df)
        
        return results_df
    
    def _get_by_levels(self, by_vars: List[str]) -> List[Tuple]:
        """Get unique levels of by variables"""
        if not by_vars:
            return [('_pooled',)]
        
        levels = self.data[by_vars].drop_duplicates().values
        return [tuple(row) for row in levels]
    
    def _parse_over(self, over: Optional[Union[str, Tuple]]) -> Tuple[Optional[str], Optional[str]]:
        """Parse over parameter into variable and test option"""
        if over is None:
            return None, None
        
        if isinstance(over, tuple):
            return over[0], over[1] if len(over) > 1 else None
        
        return over, None
    
    def _get_over_levels(self, over_var: Optional[str]) -> List:
        """Get unique levels of over variable"""
        if over_var is None:
            return ['NoOver']
        
        # Expand PV pattern if needed
        var_name = over_var.replace('@', '1') if '@' in over_var else over_var
        
        if var_name not in self.data.columns:
            raise ValueError(f"Over variable '{var_name}' not found in data")
        
        return sorted(self.data[var_name].dropna().unique())
    
    def _filter_by_level(self, by_vars: List[str], by_level: Tuple) -> pd.DataFrame:
        """Filter data to by-level"""
        if by_level == ('_pooled',):
            return self.data.copy()
        
        mask = np.ones(len(self.data), dtype=bool)
        for var, val in zip(by_vars, by_level):
            mask &= (self.data[var] == val)
        
        return self.data[mask].copy()
    
    def _filter_over_level(self, data: pd.DataFrame, over_var: Optional[str], 
                          over_level) -> pd.DataFrame:
        """Filter data to over-level"""
        if over_var is None or over_level == 'NoOver':
            return data
        
        var_name = over_var.replace('@', '1') if '@' in over_var else over_var
        return data[data[var_name] == over_level].copy()
    
    def _estimate_with_replicates(self, 
                                  data: pd.DataFrame,
                                  func: Callable,
                                  fast: bool = False,
                                  **kwargs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
        """
        Estimate with replicate weights and plausible values
        
        Returns
        -------
        coefs : np.ndarray or None
            Point estimates
        vcov : np.ndarray or None
            Variance-covariance matrix
        stat_names : list of str or None
            Names of statistics
        """
        variance_factor = self._get_variance_factor(data)
        
        all_betas = []
        all_bvars = []
        stat_names = None
        
        # Loop over plausible values
        for pv_idx in range(1, self.n_pv + 1):
            # Get PV-specific data
            pv_data = self._prepare_pv_data(data, pv_idx)
            
            # Estimate with final weight
            try:
                beta, names = self._estimate_single(pv_data, func, self.final_weight, **kwargs)
                if stat_names is None:
                    stat_names = names
            except Exception as e:
                warnings.warn(f"Estimation failed for PV {pv_idx}: {str(e)}")
                return None, None, None
            
            if beta is None:
                return None, None, None
            
            # Estimate with replicate weights
            if fast and pv_idx > 1:
                # Fast mode: reuse first PV's replicate estimates
                bvar = all_bvars[0]
            else:
                rep_estimates = []
                for rep_weight in self.rep_weights:
                    if rep_weight not in pv_data.columns:
                        continue
                    
                    rep_est, _ = self._estimate_single(pv_data, func, rep_weight, **kwargs)
                    if rep_est is not None:
                        rep_estimates.append(rep_est)
                
                if not rep_estimates:
                    return None, None, None
                
                # Compute replicate deviations
                rep_estimates = np.array(rep_estimates)
                bvar = rep_estimates - beta
            
            all_betas.append(beta)
            all_bvars.append(bvar)
        
        # Combine across PVs
        if self.has_pv and len(all_betas) > 1:
            coefs, vcov = self._combine_pv_estimates(
                all_betas, all_bvars, variance_factor
            )
        else:
            # Single estimate (no PVs)
            beta = all_betas[0]
            bvar = all_bvars[0]
            vcov = variance_factor * (bvar.T @ bvar)
            coefs = beta
        
        return coefs, vcov, stat_names
    
    def _prepare_pv_data(self, data: pd.DataFrame, pv_idx: int) -> pd.DataFrame:
        """Prepare data for specific PV index"""
        if not self.has_pv:
            return data
        
        # No need to modify data - PV variables already in correct form
        return data
    
    def _estimate_single(self, data: pd.DataFrame, func: Callable, 
                        weight: str, **kwargs) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Run single estimation with given weight"""
        try:
            result = func(data, weight, **kwargs)
            
            if isinstance(result, dict):
                # Convert dict to array and get names
                names = list(result.keys())
                values = np.array(list(result.values()))
                return values, names
            elif isinstance(result, (list, tuple)):
                return np.array(result), None
            else:
                return np.array([result]), None
        except Exception as e:
            warnings.warn(f"Estimation failed: {str(e)}")
            return None, None
    
    def _combine_pv_estimates(self, 
                             betas: List[np.ndarray],
                             bvars: List[np.ndarray],
                             variance_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine estimates across plausible values
        
        Uses Rubin's rules for multiple imputation:
        - Total variance = Sampling variance + Imputation variance
        """
        betas = np.array(betas)
        n_pv = len(betas)
        
        # Point estimate: mean across PVs
        beta = betas.mean(axis=0)
        
        # Sampling variance: mean of replicate variances
        sampling_vars = []
        for bvar in bvars:
            vcov_pv = variance_factor * (bvar.T @ bvar)
            sampling_vars.append(vcov_pv)
        sampling_var = np.mean(sampling_vars, axis=0)
        
        # Imputation variance: variance across PV estimates
        beta_devs = betas - beta
        imputation_var = (beta_devs.T @ beta_devs) / (n_pv - 1)
        
        # Total variance (Rubin's rules)
        vcov = sampling_var + ((n_pv + 1) / n_pv) * imputation_var
        
        return beta, vcov
    
    def _build_result_row(self, coefs: np.ndarray, se: np.ndarray,
                         by_vars: List[str], by_level: Tuple,
                         over_var: Optional[str], over_level,
                         stat_names: Optional[List[str]] = None) -> Dict:
        """Build result row dictionary"""
        result = {}
        
        # Add by variables
        if by_level == ('_pooled',):
            result['_pooled'] = 'pooled'
        else:
            for var, val in zip(by_vars, by_level):
                result[var] = val
        
        # Add over variable
        if over_var is not None:
            result['over_value'] = over_level
        
        # Add coefficients and SEs
        # If stat_names provided, use them; otherwise use generic names
        if stat_names is None:
            stat_names = [f'coef_{i}' for i in range(len(coefs))]
        
        for name, c, s in zip(stat_names, coefs, se):
            result[f'{name}_b'] = c
            result[f'{name}_se'] = s
        
        return result
    
    def _compute_over_difference(self, level_results: List[Dict],
                                over_test: str, by_vars: List[str],
                                by_level: Tuple) -> Dict:
        """Compute difference between first and last over levels"""
        first = level_results[0]
        last = level_results[-1]
        
        sign = -1 if over_test == '-test' else 1
        
        result = {}
        
        # Copy by variables
        if by_level == ('_pooled',):
            result['_pooled'] = 'pooled'
        else:
            for var, val in zip(by_vars, by_level):
                result[var] = val
        
        result['over_value'] = 'd'
        
        # Compute differences
        for key in first.keys():
            if key.endswith('_b') or key.endswith('_se'):
                result[key] = sign * (last[key] - first[key])
        
        return result
    
    def _reshape_over_results(self, df: pd.DataFrame, over_var: str) -> pd.DataFrame:
        """Reshape results from long to wide format for over variable"""
        # Identify value columns
        value_cols = [c for c in df.columns if c.endswith(('_b', '_se', '_pv'))]
        
        # Pivot
        id_vars = [c for c in df.columns if c not in value_cols + ['over_value']]
        
        df_wide = df.pivot(
            index=id_vars if id_vars else None,
            columns='over_value',
            values=value_cols
        )
        
        # Flatten column names
        df_wide.columns = [f'{col[1]}_{col[0]}' for col in df_wide.columns]
        df_wide = df_wide.reset_index()
        
        return df_wide
    
    def _display_results(self, results: pd.DataFrame):
        """Display results table"""
        print("\n" + "="*80)
        print("REPEST ESTIMATION RESULTS")
        print("="*80)
        print(results.to_string(index=False))
        print("="*80 + "\n")


if __name__ == '__main__':
    # Example usage
    print("RepEst - Replicate Estimation for Survey Data")
    print("="*80)
    print("\nAvailable survey configurations:")
    for name, params in SURVEY_CONFIGS.items():
        print(f"  - {name}: {params.n_pv} PVs, {params.n_reps} replicates")
    print("\nUse RepEst class to analyze survey data with replicate weights and PVs")