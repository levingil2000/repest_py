"""
Estimation Functions for RepEst

Specialized statistical functions for survey data analysis:
- Means and percentages
- Frequencies
- Summary statistics
- Quantile tables
- Correlations
- Regression integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from scipy import stats
import warnings

# Optional: statsmodels for regression support
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols, logit, probit
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class EstimationFunctions:
    """Collection of estimation functions for use with RepEst"""
    
    @staticmethod
    def means(data: pd.DataFrame, weight: str, variables: List[str],
              pct: bool = False, coverage: bool = False) -> Dict[str, float]:
        """
        Compute weighted means
        
        NOTE: This function computes means using a SINGLE weight variable.
        When used with RepEst.estimate(), it will be called multiple times:
        - Once with final weight (for point estimate)
        - Once for each replicate weight (for variance estimation)
        The framework automatically computes standard errors from replicate variability.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data
        weight : str
            Weight variable name (e.g., 'w_fstuwt' or 'w_fstr1')
        variables : list of str
            Variables to compute means for
        pct : bool, default False
            Multiply by 100 (for percentages)
        coverage : bool, default False
            Include coverage statistics
            
        Returns
        -------
        dict
            Dictionary of statistics (e.g., {'var1_m': 500.5})
        """
        results = {}
        weights = data[weight].values
        
        for var in variables:
            if var not in data.columns:
                continue
                
            values = data[var].values
            
            # Remove missing
            mask = ~(pd.isna(values) | pd.isna(weights))
            vals_clean = values[mask]
            wts_clean = weights[mask]
            
            if len(vals_clean) == 0:
                results[f'{var}_m'] = np.nan
                if coverage:
                    results[f'{var}_x'] = 0.0
                continue
            
            # Compute mean
            mean_val = np.average(vals_clean, weights=wts_clean)
            if pct:
                mean_val *= 100
            
            results[f'{var}_m'] = mean_val
            
            # Coverage: proportion non-missing
            if coverage:
                coverage_val = mask.sum() / len(mask)
                results[f'{var}_x'] = coverage_val
        
        return results
    
    @staticmethod
    def freq(data: pd.DataFrame, weight: str, variable: str,
             levels: List[Union[int, str]], count: bool = False,
             coverage: bool = False) -> Dict[str, float]:
        """
        Compute weighted frequencies/percentages
        
        Parameters
        ----------
        data : pd.DataFrame
            Data
        weight : str
            Weight variable
        variable : str
            Variable to tabulate
        levels : list
            Levels to compute frequencies for
        count : bool, default False
            Return counts instead of percentages
        coverage : bool, default False
            Include coverage statistics
            
        Returns
        -------
        dict
            Dictionary of frequencies/percentages
        """
        results = {}
        
        if variable not in data.columns:
            for level in levels:
                lev_str = str(level).replace('-', 'm')
                results[f'{variable}_{lev_str}'] = np.nan
            return results
        
        weights = data[weight].values
        values = data[variable].values
        
        # Remove missing
        mask = ~(pd.isna(values) | pd.isna(weights))
        vals_clean = values[mask]
        wts_clean = weights[mask]
        
        if len(vals_clean) == 0:
            for level in levels:
                lev_str = str(level).replace('-', 'm')
                results[f'{variable}_{lev_str}'] = 0.0 if count else 0.0
            if coverage:
                results[f'{variable}_x'] = 0.0
            return results
        
        total_weight = wts_clean.sum()
        
        # Compute for each level
        for level in levels:
            lev_str = str(level).replace('-', 'm')
            level_mask = (vals_clean == level)
            level_weight = wts_clean[level_mask].sum()
            
            if count:
                results[f'{variable}_{lev_str}'] = level_weight
            else:
                pct = 100 * level_weight / total_weight if total_weight > 0 else 0.0
                results[f'{variable}_{lev_str}'] = pct
        
        # Coverage
        if coverage:
            coverage_val = mask.sum() / len(mask)
            results[f'{variable}_x'] = coverage_val
        
        return results
    
    @staticmethod
    def summarize(data: pd.DataFrame, weight: str, variables: List[str],
                  stats: List[str] = ['mean', 'sd'],
                  coverage: bool = False) -> Dict[str, float]:
        """
        Compute summary statistics
        
        Parameters
        ----------
        data : pd.DataFrame
            Data
        weight : str
            Weight variable
        variables : list of str
            Variables to summarize
        stats : list of str
            Statistics to compute. Options:
            - mean, sd, min, max, sum_w, N, Var, sum
            - p1, p5, p10, p25, p50, p75, p90, p95, p99 (percentiles)
            - skewness, kurtosis
            - iqr (interquartile range: p75 - p25)
            - idr (interdecile range: p90 - p10)
            - sd_ub, Var_ub (unbiased versions)
        coverage : bool, default False
            Include coverage statistics
            
        Returns
        -------
        dict
            Dictionary of summary statistics
        """
        results = {}
        weights_all = data[weight].values
        
        for var in variables:
            if var not in data.columns:
                for stat in stats:
                    results[f'{var}_{stat}'] = np.nan
                continue
            
            values = data[var].values
            
            # Remove missing
            mask = ~(pd.isna(values) | pd.isna(weights_all))
            vals = values[mask]
            wts = weights_all[mask]
            
            if len(vals) == 0:
                for stat in stats:
                    results[f'{var}_{stat}'] = np.nan
                if coverage:
                    results[f'{var}_x'] = 0.0
                continue
            
            # Compute each statistic
            for stat in stats:
                if stat == 'mean':
                    val = np.average(vals, weights=wts)
                
                elif stat == 'sd':
                    # Biased SD (population)
                    mean = np.average(vals, weights=wts)
                    var = np.average((vals - mean)**2, weights=wts)
                    val = np.sqrt(var)
                
                elif stat == 'sd_ub':
                    # Unbiased SD (sample)
                    mean = np.average(vals, weights=wts)
                    var = np.average((vals - mean)**2, weights=wts)
                    n = len(vals)
                    val = np.sqrt(var * n / (n - 1)) if n > 1 else np.nan
                
                elif stat == 'Var':
                    # Biased variance
                    mean = np.average(vals, weights=wts)
                    val = np.average((vals - mean)**2, weights=wts)
                
                elif stat == 'Var_ub':
                    # Unbiased variance
                    mean = np.average(vals, weights=wts)
                    var = np.average((vals - mean)**2, weights=wts)
                    n = len(vals)
                    val = var * n / (n - 1) if n > 1 else np.nan
                
                elif stat == 'min':
                    val = vals.min()
                
                elif stat == 'max':
                    val = vals.max()
                
                elif stat == 'sum_w':
                    val = wts.sum()
                
                elif stat == 'sum':
                    val = (vals * wts).sum()
                
                elif stat == 'N':
                    val = len(vals)
                
                elif stat == 'skewness':
                    mean = np.average(vals, weights=wts)
                    m2 = np.average((vals - mean)**2, weights=wts)
                    m3 = np.average((vals - mean)**3, weights=wts)
                    val = m3 / (m2**1.5) if m2 > 0 else np.nan
                
                elif stat == 'kurtosis':
                    mean = np.average(vals, weights=wts)
                    m2 = np.average((vals - mean)**2, weights=wts)
                    m4 = np.average((vals - mean)**4, weights=wts)
                    val = m4 / (m2**2) if m2 > 0 else np.nan
                
                elif stat.startswith('p'):
                    # Percentile
                    try:
                        pct = int(stat[1:])
                        val = EstimationFunctions._weighted_percentile(vals, wts, pct)
                    except:
                        val = np.nan
                
                elif stat == 'iqr':
                    p75 = EstimationFunctions._weighted_percentile(vals, wts, 75)
                    p25 = EstimationFunctions._weighted_percentile(vals, wts, 25)
                    val = p75 - p25
                
                elif stat == 'idr':
                    p90 = EstimationFunctions._weighted_percentile(vals, wts, 90)
                    p10 = EstimationFunctions._weighted_percentile(vals, wts, 10)
                    val = p90 - p10
                
                else:
                    val = np.nan
                
                results[f'{var}_{stat}'] = val
            
            # Coverage
            if coverage:
                coverage_val = mask.sum() / len(mask)
                results[f'{var}_x'] = coverage_val
        
        return results
    
    @staticmethod
    def _weighted_percentile(values: np.ndarray, weights: np.ndarray, 
                            percentile: float) -> float:
        """Compute weighted percentile"""
        # Sort by values
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Cumulative weights
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        
        # Find percentile position
        target = (percentile / 100) * total_weight
        
        # Linear interpolation
        idx = np.searchsorted(cum_weights, target)
        if idx == 0:
            return sorted_values[0]
        elif idx >= len(sorted_values):
            return sorted_values[-1]
        else:
            # Interpolate
            w0 = cum_weights[idx - 1]
            w1 = cum_weights[idx]
            v0 = sorted_values[idx - 1]
            v1 = sorted_values[idx]
            
            if w1 - w0 > 0:
                frac = (target - w0) / (w1 - w0)
                return v0 + frac * (v1 - v0)
            else:
                return v0
    
    @staticmethod
    def quantiletable(data: pd.DataFrame, weight: str, 
                     index_var: str, outcome_var: str,
                     n_quantiles: int = 4,
                     index_quantiles: bool = True,
                     outcome_quantiles: bool = True,
                     test: bool = False,
                     relrisk: bool = False,
                     oddsratio: bool = False,
                     summarize_var: Optional[str] = None,
                     coverage: bool = False) -> Dict[str, float]:
        """
        Compute quantile table statistics
        
        Creates quantiles based on index_var and computes statistics for outcome_var
        
        Parameters
        ----------
        data : pd.DataFrame
            Data
        weight : str
            Weight variable
        index_var : str
            Variable to create quantiles from
        outcome_var : str
            Outcome variable to summarize
        n_quantiles : int, default 4
            Number of quantiles (e.g., 4 for quartiles)
        index_quantiles : bool, default True
            Compute mean of index_var in each quantile
        outcome_quantiles : bool, default True
            Compute mean of outcome_var in each quantile
        test : bool, default False
            Test difference between top and bottom quantile
        relrisk : bool, default False
            Compute relative risk
        oddsratio : bool, default False
            Compute odds ratio
        summarize_var : str, optional
            Additional variable to summarize
        coverage : bool, default False
            Include coverage statistics
            
        Returns
        -------
        dict
            Dictionary of quantile statistics
        """
        results = {}
        
        if index_var not in data.columns or outcome_var not in data.columns:
            return results
        
        weights = data[weight].values
        index_vals = data[index_var].values
        outcome_vals = data[outcome_var].values
        
        # Remove missing from index (keep outcome for some stats)
        mask = ~(pd.isna(index_vals) | pd.isna(weights))
        
        if mask.sum() == 0:
            return results
        
        # Add random noise to break ties (for percentile computation)
        np.random.seed(5094)
        index_with_noise = index_vals.copy()
        index_with_noise[mask] += 0.0001 * np.random.uniform(size=mask.sum())
        
        # Compute quantile thresholds
        thresholds = []
        for q in range(1, n_quantiles):
            pct = 100 * q / n_quantiles
            threshold = EstimationFunctions._weighted_percentile(
                index_with_noise[mask], weights[mask], pct
            )
            thresholds.append(threshold)
        
        # Assign quantiles
        quantile_assignment = np.zeros(len(data), dtype=int)
        quantile_assignment[~mask] = -1  # Missing
        
        for i in range(n_quantiles):
            if i == 0:
                q_mask = mask & (index_with_noise <= thresholds[0])
            elif i == n_quantiles - 1:
                q_mask = mask & (index_with_noise > thresholds[-1])
            else:
                q_mask = mask & (index_with_noise > thresholds[i-1]) & (index_with_noise <= thresholds[i])
            
            quantile_assignment[q_mask] = i + 1
        
        # Compute statistics by quantile
        for q in range(1, n_quantiles + 1):
            q_mask = (quantile_assignment == q)
            
            if q_mask.sum() == 0:
                if index_quantiles:
                    results[f'{index_var}_q{q}'] = np.nan
                if outcome_quantiles:
                    results[f'{outcome_var}_q{q}'] = np.nan
                continue
            
            # Index mean in quantile
            if index_quantiles:
                idx_mean = np.average(index_vals[q_mask], weights=weights[q_mask])
                results[f'{index_var}_q{q}'] = idx_mean
            
            # Outcome mean in quantile (remove outcome missing)
            if outcome_quantiles:
                out_mask = q_mask & ~pd.isna(outcome_vals)
                if out_mask.sum() > 0:
                    out_mean = np.average(outcome_vals[out_mask], weights=weights[out_mask])
                    results[f'{outcome_var}_q{q}'] = out_mean
                else:
                    results[f'{outcome_var}_q{q}'] = np.nan
        
        # Difference between top and bottom quantile
        if test and outcome_quantiles:
            top_key = f'{outcome_var}_q{n_quantiles}'
            bot_key = f'{outcome_var}_q1'
            if top_key in results and bot_key in results:
                results[f'{outcome_var}_qd'] = results[top_key] - results[bot_key]
        
        # Relative risk and odds ratio
        if relrisk or oddsratio:
            # Create binary index (bottom quantile vs rest)
            index_binary = (quantile_assignment == 1).astype(float)
            index_binary[quantile_assignment == -1] = np.nan
            
            # Create binary outcome (bottom quantile vs rest based on outcome)
            outcome_with_noise = outcome_vals.copy()
            out_mask = ~pd.isna(outcome_vals) & ~pd.isna(weights)
            if out_mask.sum() > 0:
                outcome_with_noise[out_mask] += 0.0001 * np.random.uniform(size=out_mask.sum())
                outcome_threshold = EstimationFunctions._weighted_percentile(
                    outcome_with_noise[out_mask], weights[out_mask], 100 / n_quantiles
                )
                
                # 2x2 table
                both_mask = ~pd.isna(index_binary) & ~pd.isna(outcome_vals)
                if both_mask.sum() > 0:
                    # Q1 index & Q1 outcome
                    mask_11 = both_mask & (index_binary == 1) & (outcome_with_noise <= outcome_threshold)
                    p11 = weights[mask_11].sum() / weights[both_mask & (outcome_with_noise <= outcome_threshold)].sum()
                    
                    # Q1 index & rest outcome
                    mask_10 = both_mask & (index_binary == 1) & (outcome_with_noise > outcome_threshold)
                    p10 = weights[mask_10].sum() / weights[both_mask & (outcome_with_noise > outcome_threshold)].sum()
                    
                    if relrisk:
                        rr = p11 / p10 if p10 > 0 else np.nan
                        results['rrisk'] = rr
                    
                    if oddsratio:
                        odds11 = p11 / (1 - p11) if p11 < 1 else np.inf
                        odds10 = p10 / (1 - p10) if p10 < 1 else np.inf
                        OR = odds11 / odds10 if odds10 > 0 else np.nan
                        results['oratio'] = OR
        
        # Summarize additional variable
        if summarize_var and summarize_var in data.columns:
            sum_vals = data[summarize_var].values
            sum_mask = ~(pd.isna(sum_vals) | pd.isna(weights))
            
            if sum_mask.sum() > 0:
                mean = np.average(sum_vals[sum_mask], weights=weights[sum_mask])
                var = np.average((sum_vals[sum_mask] - mean)**2, weights=weights[sum_mask])
                sd = np.sqrt(var)
                
                results[f'{summarize_var}_mean'] = mean
                results[f'{summarize_var}_sd'] = sd
        
        # Coverage
        if coverage:
            cov_mask = ~(pd.isna(index_vals) | pd.isna(outcome_vals))
            results['e_coverage'] = cov_mask.sum() / len(cov_mask)
        
        return results
    
    @staticmethod
    def corr(data: pd.DataFrame, weight: str, variables: List[str],
             pairwise: bool = False, coverage: bool = False) -> Dict[str, float]:
        """
        Compute weighted correlations
        
        NOTE: This function computes correlation using a SINGLE weight variable.
        When used with RepEst.estimate(), it will be called 81 times:
        - Once with final weight (for point estimate)
        - 80 times with replicate weights (for variance estimation)
        The standard error comes from variability across replicate estimates.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data
        weight : str
            Weight variable name (e.g., 'w_fstuwt' or 'w_fstr1')
        variables : list of str
            Variables to correlate
        pairwise : bool, default False
            Use pairwise deletion (vs listwise)
        coverage : bool, default False
            Include coverage statistics
            
        Returns
        -------
        dict
            Dictionary of correlations (e.g., {'c_var1_var2': 0.85})
        """
        results = {}
        weights = data[weight].values
        
        n_vars = len(variables)
        
        if not pairwise:
            # Listwise deletion
            values = np.column_stack([data[v].values for v in variables])
            mask = ~np.any(pd.isna(values), axis=1) & ~pd.isna(weights)
            
            if mask.sum() < 2:
                # Not enough observations
                for i in range(n_vars):
                    for j in range(i + 1, n_vars):
                        var1 = variables[i][:12]
                        var2 = variables[j][:12]
                        results[f'c_{var1}_{var2}'] = np.nan
                return results
            
            vals_clean = values[mask]
            wts_clean = weights[mask]
            
            # Compute correlations
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    v1 = vals_clean[:, i]
                    v2 = vals_clean[:, j]
                    
                    mean1 = np.average(v1, weights=wts_clean)
                    mean2 = np.average(v2, weights=wts_clean)
                    
                    cov = np.average((v1 - mean1) * (v2 - mean2), weights=wts_clean)
                    var1 = np.average((v1 - mean1)**2, weights=wts_clean)
                    var2 = np.average((v2 - mean2)**2, weights=wts_clean)
                    
                    corr = cov / np.sqrt(var1 * var2) if (var1 * var2) > 0 else np.nan
                    
                    var1_name = variables[i][:12]
                    var2_name = variables[j][:12]
                    results[f'c_{var1_name}_{var2_name}'] = corr
            
            # Coverage
            if coverage:
                results['e_coverage'] = mask.sum() / len(mask)
        
        else:
            # Pairwise deletion
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    v1 = data[variables[i]].values
                    v2 = data[variables[j]].values
                    
                    mask = ~(pd.isna(v1) | pd.isna(v2) | pd.isna(weights))
                    
                    if mask.sum() < 2:
                        var1_name = variables[i][:12]
                        var2_name = variables[j][:12]
                        results[f'pwc_{var1_name}_{var2_name}'] = np.nan
                        if coverage:
                            results[f'x_{var1_name}_{var2_name}'] = 0.0
                        continue
                    
                    v1_clean = v1[mask]
                    v2_clean = v2[mask]
                    wts_clean = weights[mask]
                    
                    mean1 = np.average(v1_clean, weights=wts_clean)
                    mean2 = np.average(v2_clean, weights=wts_clean)
                    
                    cov = np.average((v1_clean - mean1) * (v2_clean - mean2), weights=wts_clean)
                    var1 = np.average((v1_clean - mean1)**2, weights=wts_clean)
                    var2 = np.average((v2_clean - mean2)**2, weights=wts_clean)
                    
                    corr = cov / np.sqrt(var1 * var2) if (var1 * var2) > 0 else np.nan
                    
                    var1_name = variables[i][:12]
                    var2_name = variables[j][:12]
                    results[f'pwc_{var1_name}_{var2_name}'] = corr
                    
                    # Coverage
                    if coverage:
                        results[f'x_{var1_name}_{var2_name}'] = mask.sum() / len(mask)
        
        return results