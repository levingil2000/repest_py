"""
Data Cleaning Utilities for Survey Data

Provides PISA-standard data cleaning operations:
- Remove missing values and adjust weights to maintain population totals
- Flatten weighted data for use with standard statistical packages
- Combined cleaning and flattening workflows
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict
import warnings
from .core import SURVEY_CONFIGS, SurveyParameters


class DataCleaner:
    """
    Data cleaning utilities for survey data
    
    Provides methods to:
    1. Remove rows with missing values in specified columns
    2. Adjust weights to maintain population totals (PISA standard)
    3. Flatten weighted data by replicating rows
    4. Combined cleaning and flattening workflows
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 survey: Union[str, SurveyParameters, None] = None,
                 final_weight: Optional[str] = None,
                 rep_weight_prefix: Optional[str] = None,
                 n_reps: Optional[int] = None):
        """
        Initialize DataCleaner
        
        Parameters
        ----------
        data : pd.DataFrame
            Survey data
        survey : str or SurveyParameters, optional
            Survey name (e.g., 'PISA', 'PIAAC') or custom parameters
            If None, must provide final_weight
        final_weight : str, optional
            Final weight column name (required if survey is None)
        rep_weight_prefix : str, optional
            Replicate weight prefix (for adjusting all weights)
        n_reps : int, optional
            Number of replicate weights
        """
        self.data = data
        
        # Setup survey parameters
        if survey is not None:
            if isinstance(survey, str):
                if survey.upper() not in SURVEY_CONFIGS:
                    raise ValueError(f"Unknown survey: {survey}. Available: {list(SURVEY_CONFIGS.keys())}")
                self.params = SURVEY_CONFIGS[survey.upper()]
            else:
                self.params = survey
            
            self.final_weight = self.params.final_weight
            self.rep_weight_prefix = self.params.rep_weight_prefix
            self.n_reps = self.params.n_reps
        else:
            if final_weight is None:
                raise ValueError("Must provide either 'survey' or 'final_weight'")
            self.params = None
            self.final_weight = final_weight
            self.rep_weight_prefix = rep_weight_prefix
            self.n_reps = n_reps or 0
        
        # Check if weight exists (try both cases)
        if self.final_weight not in self.data.columns:
            final_upper = self.final_weight.upper()
            if final_upper in self.data.columns:
                self.final_weight = final_upper
            else:
                raise ValueError(f"Final weight '{self.final_weight}' not found in data")
        
        # Setup replicate weights if available
        self.rep_weights = []
        if self.rep_weight_prefix and self.n_reps:
            self.rep_weights = [
                f"{self.rep_weight_prefix}{i}" 
                for i in range(1, self.n_reps + 1)
            ]
            # Check case
            if self.rep_weights[0] not in self.data.columns:
                rep_upper = self.rep_weights[0].upper()
                if rep_upper in self.data.columns:
                    self.rep_weights = [w.upper() for w in self.rep_weights]
    
    def remove_na_adjust_weights(self,
                                 columns: List[str],
                                 by: Optional[Union[str, List[str]]] = None,
                                 adjust_replicates: bool = True,
                                 inplace: bool = False,
                                 verbose: bool = True) -> pd.DataFrame:
        """
        Remove rows with missing values and adjust weights (PISA standard)
        
        This method:
        1. Identifies rows with missing values in specified columns
        2. Removes these rows
        3. Adjusts weights within groups so totals are maintained
        
        The weight adjustment formula (within each group):
        new_weight = old_weight * (sum_original_weights / sum_remaining_weights)
        
        This ensures the weighted population size remains constant after
        removing missing data.
        
        Parameters
        ----------
        columns : list of str
            Columns to check for missing values
        by : str or list of str, optional
            Variable(s) to group by for weight adjustment
            (e.g., 'CNT' to adjust within countries)
            If None, adjusts globally
        adjust_replicates : bool, default True
            Also adjust replicate weights
        inplace : bool, default False
            Modify data in place
        verbose : bool, default True
            Print summary statistics
            
        Returns
        -------
        pd.DataFrame
            Cleaned data with adjusted weights
        """
        if inplace:
            result = self.data
        else:
            result = self.data.copy()
        
        # Identify complete cases
        missing_mask = pd.DataFrame(result[columns]).isna().any(axis=1)
        n_missing = missing_mask.sum()
        n_total = len(result)
        
        if verbose:
            print(f"\nData Cleaning Summary:")
            print(f"  Total rows: {n_total:,}")
            print(f"  Rows with missing values: {n_missing:,} ({100*n_missing/n_total:.2f}%)")
            print(f"  Rows retained: {n_total - n_missing:,} ({100*(n_total-n_missing)/n_total:.2f}%)")
        
        # Parse by variables
        by_vars = [by] if isinstance(by, str) else (by or [])
        
        # Adjust weights within groups
        weight_cols = [self.final_weight]
        if adjust_replicates and self.rep_weights:
            weight_cols.extend(self.rep_weights)
        
        for weight_col in weight_cols:
            if weight_col not in result.columns:
                continue
            
            result[weight_col] = self._adjust_weights_by_group(
                result, weight_col, ~missing_mask, by_vars
            )
        
        # Remove missing rows
        result = result[~missing_mask].reset_index(drop=True)
        
        if verbose and by_vars:
            print(f"  Weights adjusted within: {', '.join(by_vars)}")
            
            # Show weight adjustment by group
            print(f"\n  Weight adjustment factors by group:")
            for group_vals, group_data in result.groupby(by_vars):
                group_label = group_vals if isinstance(group_vals, tuple) else (group_vals,)
                original_sum = self.data[self.data[by_vars[0] if len(by_vars)==1 else by_vars].isin([group_label[0] if len(by_vars)==1 else group_vals])][self.final_weight].sum()
                new_sum = group_data[self.final_weight].sum()
                print(f"    {' | '.join(str(v) for v in group_label)}: {original_sum/new_sum:.4f}x")
        
        return result
    
    def _adjust_weights_by_group(self, 
                                 data: pd.DataFrame,
                                 weight_col: str,
                                 keep_mask: pd.Series,
                                 by_vars: List[str]) -> pd.Series:
        """
        Adjust weights within groups to maintain totals
        
        Parameters
        ----------
        data : pd.DataFrame
            Data
        weight_col : str
            Weight column to adjust
        keep_mask : pd.Series
            Boolean mask of rows to keep
        by_vars : list of str
            Grouping variables
            
        Returns
        -------
        pd.Series
            Adjusted weights
        """
        adjusted_weights = data[weight_col].copy()
        
        if not by_vars:
            # Global adjustment
            original_sum = data[weight_col].sum()
            remaining_sum = data.loc[keep_mask, weight_col].sum()
            
            if remaining_sum > 0:
                adjustment_factor = original_sum / remaining_sum
                adjusted_weights[keep_mask] *= adjustment_factor
        else:
            # Group-wise adjustment
            for group_vals, group_data in data.groupby(by_vars):
                group_mask = data.index.isin(group_data.index)
                group_keep_mask = group_mask & keep_mask
                
                original_sum = data.loc[group_mask, weight_col].sum()
                remaining_sum = data.loc[group_keep_mask, weight_col].sum()
                
                if remaining_sum > 0:
                    adjustment_factor = original_sum / remaining_sum
                    adjusted_weights[group_keep_mask] *= adjustment_factor
        
        return adjusted_weights
    
    def flatten_with_weights(self,
                            weight_column: Optional[str] = None,
                            round_weights: bool = True,
                            min_weight: float = 0.5,
                            by: Optional[Union[str, List[str]]] = None,
                            max_rows: Optional[int] = None,
                            random_state: Optional[int] = None,
                            verbose: bool = True) -> pd.DataFrame:
        """
        Flatten weighted data by replicating rows according to weights
        
        Creates an "unweighted" dataset where each row appears approximately
        weight times. Useful for statistical packages that don't support weights.
        
        WARNING: This can create very large datasets!
        
        Parameters
        ----------
        weight_column : str, optional
            Weight column to use (default: final weight)
        round_weights : bool, default True
            Round weights to integers (vs random sampling)
        min_weight : float, default 0.5
            Minimum weight for inclusion (when rounding)
        by : str or list of str, optional
            Variables to preserve in output
        max_rows : int, optional
            Maximum rows in output (random sample if exceeded)
        random_state : int, optional
            Random seed for reproducibility
        verbose : bool, default True
            Print summary statistics
            
        Returns
        -------
        pd.DataFrame
            Flattened data (unweighted)
        """
        if weight_column is None:
            weight_column = self.final_weight
        
        if weight_column not in self.data.columns:
            raise ValueError(f"Weight column '{weight_column}' not found in data")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Determine replication counts
        weights = self.data[weight_column].values
        
        if round_weights:
            # Round to nearest integer (minimum min_weight)
            counts = np.round(weights)
            counts[counts < min_weight] = 0
            counts = counts.astype(int)
        else:
            # Stochastic rounding (fractional weights -> probability)
            integer_part = np.floor(weights).astype(int)
            fractional_part = weights - integer_part
            
            # Add 1 with probability = fractional part
            random_draws = np.random.uniform(size=len(weights))
            counts = integer_part + (random_draws < fractional_part).astype(int)
        
        # Calculate expected output size
        total_rows = counts.sum()
        
        if verbose:
            print(f"\nFlattening Data:")
            print(f"  Original rows: {len(self.data):,}")
            print(f"  Expected output rows: {total_rows:,}")
            print(f"  Expansion factor: {total_rows/len(self.data):.2f}x")
        
        # Check if too large
        if max_rows and total_rows > max_rows:
            warnings.warn(
                f"Flattened data would have {total_rows:,} rows (>{max_rows:,}). "
                f"Randomly sampling {max_rows:,} rows instead."
            )
            
            # Sample proportionally
            sample_fraction = max_rows / total_rows
            counts = (counts * sample_fraction).round().astype(int)
            total_rows = counts.sum()
            
            if verbose:
                print(f"  Sampling to {total_rows:,} rows")
        
        # Replicate rows
        flattened_data = self.data.loc[self.data.index.repeat(counts)].reset_index(drop=True)
        
        # Remove weight column (now meaningless)
        if weight_column in flattened_data.columns:
            flattened_data = flattened_data.drop(columns=[weight_column])
        
        # Remove replicate weights too
        for rep_weight in self.rep_weights:
            if rep_weight in flattened_data.columns:
                flattened_data = flattened_data.drop(columns=[rep_weight])
        
        if verbose:
            print(f"  Final rows: {len(flattened_data):,}")
        
        return flattened_data
    
    def clean_and_flatten(self,
                         columns: List[str],
                         weight_column: Optional[str] = None,
                         by: Optional[Union[str, List[str]]] = None,
                         round_weights: bool = True,
                         adjust_replicates: bool = False,
                         max_rows: Optional[int] = None,
                         random_state: Optional[int] = None,
                         verbose: bool = True) -> pd.DataFrame:
        """
        Combined workflow: Clean data then flatten
        
        Convenience method that:
        1. Removes missing values in specified columns
        2. Adjusts weights to maintain population totals
        3. Flattens data by replicating rows according to weights
        
        Parameters
        ----------
        columns : list of str
            Columns to check for missing values
        weight_column : str, optional
            Weight column to use (default: final weight)
        by : str or list of str, optional
            Variables to group by for weight adjustment
        round_weights : bool, default True
            Round weights to integers when flattening
        adjust_replicates : bool, default False
            Also adjust replicate weights (not needed for flattening)
        max_rows : int, optional
            Maximum rows in flattened output
        random_state : int, optional
            Random seed for reproducibility
        verbose : bool, default True
            Print summary statistics
            
        Returns
        -------
        pd.DataFrame
            Cleaned and flattened data (unweighted)
        """
        # Step 1: Clean and adjust weights
        if verbose:
            print("="*80)
            print("STEP 1: CLEANING DATA AND ADJUSTING WEIGHTS")
            print("="*80)
        
        cleaned = self.remove_na_adjust_weights(
            columns=columns,
            by=by,
            adjust_replicates=adjust_replicates,
            inplace=False,
            verbose=verbose
        )
        
        # Step 2: Flatten
        if verbose:
            print("\n" + "="*80)
            print("STEP 2: FLATTENING DATA")
            print("="*80)
        
        # Create temporary cleaner with cleaned data
        temp_cleaner = DataCleaner(
            data=cleaned,
            final_weight=self.final_weight,
            rep_weight_prefix=self.rep_weight_prefix,
            n_reps=self.n_reps
        )
        
        flattened = temp_cleaner.flatten_with_weights(
            weight_column=weight_column,
            round_weights=round_weights,
            by=by,
            max_rows=max_rows,
            random_state=random_state,
            verbose=verbose
        )
        
        if verbose:
            print("\n" + "="*80)
            print("CLEANING AND FLATTENING COMPLETE")
            print("="*80 + "\n")
        
        return flattened
    
    def get_complete_cases(self,
                          columns: List[str],
                          return_mask: bool = False) -> Union[pd.DataFrame, pd.Series]:
        """
        Get complete cases (no missing values in specified columns)
        
        Parameters
        ----------
        columns : list of str
            Columns to check for missing values
        return_mask : bool, default False
            Return boolean mask instead of data
            
        Returns
        -------
        pd.DataFrame or pd.Series
            Complete cases or boolean mask
        """
        complete_mask = ~pd.DataFrame(self.data[columns]).isna().any(axis=1)
        
        if return_mask:
            return complete_mask
        else:
            return self.data[complete_mask].copy()
    
    def summarize_missingness(self,
                             columns: Optional[List[str]] = None,
                             by: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Summarize missing data patterns
        
        Parameters
        ----------
        columns : list of str, optional
            Columns to check (default: all columns)
        by : str or list of str, optional
            Group by variables
            
        Returns
        -------
        pd.DataFrame
            Summary of missing data
        """
        if columns is None:
            columns = self.data.columns.tolist()
        
        results = []
        
        if by is None:
            # Overall summary
            for col in columns:
                n_missing = self.data[col].isna().sum()
                n_total = len(self.data)
                pct_missing = 100 * n_missing / n_total
                
                results.append({
                    'variable': col,
                    'n_total': n_total,
                    'n_missing': n_missing,
                    'pct_missing': pct_missing
                })
        else:
            # By-group summary
            by_vars = [by] if isinstance(by, str) else by
            
            for group_vals, group_data in self.data.groupby(by_vars):
                for col in columns:
                    n_missing = group_data[col].isna().sum()
                    n_total = len(group_data)
                    pct_missing = 100 * n_missing / n_total
                    
                    result = {'variable': col}
                    
                    # Add group values
                    if isinstance(group_vals, tuple):
                        for var, val in zip(by_vars, group_vals):
                            result[var] = val
                    else:
                        result[by_vars[0]] = group_vals
                    
                    result.update({
                        'n_total': n_total,
                        'n_missing': n_missing,
                        'pct_missing': pct_missing
                    })
                    
                    results.append(result)
        
        return pd.DataFrame(results)


if __name__ == '__main__':
    # Example usage
    print("DataCleaner - Survey Data Cleaning Utilities")
    print("="*80)
    
    # Create synthetic test data
    np.random.seed(42)
    n = 1000
    
    test_data = pd.DataFrame({
        'CNT': np.random.choice(['USA', 'GBR', 'JPN'], n),
        'W_FSTUWT': np.random.uniform(0.5, 2.0, n),
        'MATH_SCORE': np.where(np.random.random(n) < 0.1, np.nan, np.random.normal(500, 100, n)),
        'READ_SCORE': np.where(np.random.random(n) < 0.15, np.nan, np.random.normal(500, 100, n)),
        'ESCS': np.where(np.random.random(n) < 0.05, np.nan, np.random.normal(0, 1, n)),
        'GENDER': np.random.choice([0, 1], n)
    })
    
    print("\nTest Data Created:")
    print(f"  Rows: {len(test_data)}")
    print(f"  Columns: {list(test_data.columns)}")
    
    # Test cleaner
    cleaner = DataCleaner(test_data, final_weight='W_FSTUWT')
    
    print("\n" + "="*80)
    print("Test 1: Remove NA and Adjust Weights")
    print("="*80)
    
    cleaned = cleaner.remove_na_adjust_weights(
        columns=['MATH_SCORE', 'READ_SCORE', 'ESCS'],
        by='CNT'
    )
    
    print("\n" + "="*80)
    print("Test 2: Missingness Summary")
    print("="*80)
    
    summary = cleaner.summarize_missingness(
        columns=['MATH_SCORE', 'READ_SCORE', 'ESCS'],
        by='CNT'
    )
    print(summary.to_string(index=False))
    
    print("\n" + "="*80)
    print("All tests completed successfully!")