#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sort Script
----------
This script standardizes financial ratios by period, creates a composite score,
sorts stocks into deciles, and analyzes returns by group with industry fixed-effects.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path


def load_make_data(file_path='database/make.csv'):
    """
    Load the processed financial data.
    
    Args:
        file_path (str): Path to the processed CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Loaded financial data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading financial data: {e}")
        return None


def standardize_ratios_by_period(df):
    """
    Standardize financial ratios by period (年月).
    
    Args:
        df (pd.DataFrame): Input dataframe with financial ratios
        
    Returns:
        pd.DataFrame: Dataframe with standardized ratios
    """
    if df is None or df.empty:
        return None
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if all required columns exist
    required_columns = ["證券代碼", "年月", "公司名稱", "gpoa", "roe", "cfoa", "gmar", "acc", "rtn"]
    
    # Check all necessary columns
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None
    
    # Ratio columns to standardize
    ratio_cols = ['gpoa', 'roe', 'cfoa', 'gmar', 'acc']
    
    # Standardize ratios by period (年月)
    print("Standardizing financial ratios by period...")
    
    # Create standardized ratio columns with 'z_' prefix
    for ratio in ratio_cols:
        result_df[f'z_{ratio}'] = np.nan
    
    # Group by period and standardize
    for period in result_df['年月'].unique():
        period_mask = result_df['年月'] == period
        
        for ratio in ratio_cols:
            # Get values for the current period
            values = result_df.loc[period_mask, ratio]
            
            # Calculate z-scores (standardized values)
            mean = values.mean()
            std = values.std()
            
            # Avoid division by zero
            if std > 0:
                z_scores = (values - mean) / std
                result_df.loc[period_mask, f'z_{ratio}'] = z_scores
            else:
                # If standard deviation is zero, set z-scores to zero
                result_df.loc[period_mask, f'z_{ratio}'] = 0
    
    # Create composite score (sum of standardized ratios)
    print("Creating composite score...")
    result_df['score'] = sum(result_df[f'z_{ratio}'] for ratio in ratio_cols)
    
    return result_df


def create_decile_groups(df):
    """
    Create decile groups based on composite score by period.
    
    Args:
        df (pd.DataFrame): Input dataframe with standardized ratios and score
        
    Returns:
        pd.DataFrame: Dataframe with decile group assignments
    """
    if df is None or df.empty:
        return None
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Create decile groups by period
    print("Creating decile groups by period...")
    
    # Initialize the group column as numeric
    result_df['group'] = np.nan
    
    # Group by period and assign decile groups
    for period in result_df['年月'].unique():
        period_data = result_df[result_df['年月'] == period].copy()
        
        if len(period_data) >= 10:  # Ensure enough data for 10 groups
            # Create decile based on score (10 groups, with 1 being highest score)
            # qcut returns pandas.Categorical, which we convert to numeric
            try:
                # Calculate decile groups and map to integers 1-10 (reversed so 1 is highest score)
                period_data['temp_group'] = pd.qcut(
                    period_data['score'], 
                    10, 
                    labels=False,  # Use numeric labels
                    duplicates='drop'
                )
                # Reverse order so 1 is highest score (10-group)
                period_data['temp_group'] = 10 - period_data['temp_group']
                
                # Update the main dataframe with the calculated groups
                result_df.loc[result_df['年月'] == period, 'group'] = period_data['temp_group'].values
            except ValueError as e:
                print(f"Warning: Could not create 10 groups for period {period}: {e}")
                # If we can't create exactly 10 groups, try with fewer bins
                try:
                    # Count unique values
                    unique_scores = period_data['score'].nunique()
                    if unique_scores >= 2:
                        # Use as many bins as unique values, max 10
                        n_bins = min(unique_scores, 10)
                        period_data['temp_group'] = pd.qcut(
                            period_data['score'], 
                            n_bins, 
                            labels=False, 
                            duplicates='drop'
                        )
                        # Reverse order and rescale to 1-10 range
                        period_data['temp_group'] = n_bins - period_data['temp_group']
                        # Map to 1-10 scale
                        period_data['temp_group'] = period_data['temp_group'].map(
                            lambda x: int(1 + (x - 1) * (10 - 1) / (n_bins - 1)) if n_bins > 1 else 5
                        )
                        result_df.loc[result_df['年月'] == period, 'group'] = period_data['temp_group'].values
                    else:
                        # All scores are the same, assign to middle group (5)
                        result_df.loc[result_df['年月'] == period, 'group'] = 5
                except Exception as inner_e:
                    print(f"Error creating groups for period {period}: {inner_e}")
                    # Assign to middle group as fallback
                    result_df.loc[result_df['年月'] == period, 'group'] = 5
        else:
            # Not enough data for 10 groups, assign to middle group
            print(f"Warning: Period {period} has fewer than 10 stocks, cannot create deciles")
            result_df.loc[result_df['年月'] == period, 'group'] = 5
    
    # Ensure group is integer type
    result_df['group'] = result_df['group'].fillna(5).astype(int)
    
    return result_df


def analyze_returns_by_group(df):
    """
    Analyze returns by decile group with industry fixed-effects.
    
    Args:
        df (pd.DataFrame): Input dataframe with decile groups
        
    Returns:
        tuple: (summary_df, difference_df, ttest_results, industry_effects_df)
    """
    if df is None or df.empty:
        return None, None, None, None
    
    # Drop rows with NaN returns (last period for each company)
    analysis_df = df.dropna(subset=['rtn']).copy()
    
    print("Analyzing returns by group...")
    
    # Extract industry code (first 3 digits of TEJ產業_代碼)
    if 'TEJ產業_代碼' in analysis_df.columns:
        analysis_df['industry_code'] = analysis_df['TEJ產業_代碼'].astype(str).str[:3]
    else:
        print("Warning: 'TEJ產業_代碼' column not found. Using placeholder industry codes.")
        analysis_df['industry_code'] = '000'  # Placeholder
    
    # Group by decile and calculate return statistics
    group_stats = analysis_df.groupby('group')['rtn'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).reset_index()
    
    # Calculate t-statistic for mean return different from zero
    group_stats['t_stat'] = group_stats.apply(
        lambda x: stats.ttest_1samp(
            analysis_df.loc[analysis_df['group'] == x['group'], 'rtn'], 0
        ).statistic if x['count'] > 1 else np.nan,
        axis=1
    )
    
    # Calculate p-value
    group_stats['p_value'] = group_stats.apply(
        lambda x: stats.ttest_1samp(
            analysis_df.loc[analysis_df['group'] == x['group'], 'rtn'], 0
        ).pvalue if x['count'] > 1 else np.nan,
        axis=1
    )
    
    # Sort by group number
    group_stats = group_stats.sort_values('group')
    
    # Calculate high-minus-low (H-L) portfolio returns
    # Group 1 (highest score) minus Group 10 (lowest score)
    high_returns = analysis_df.loc[analysis_df['group'] == 1, ['年月', 'rtn', 'industry_code']]
    low_returns = analysis_df.loc[analysis_df['group'] == 10, ['年月', 'rtn', 'industry_code']]
    
    # Initialize industry effects dataframe
    industry_effects = pd.DataFrame()
    
    if not high_returns.empty and not low_returns.empty:
        # Calculate overall high-minus-low returns for each period (without industry matching)
        high_df = high_returns.copy().rename(columns={'rtn': 'high_rtn'})
        low_df = low_returns.copy().rename(columns={'rtn': 'low_rtn'})
        
        # Merge high and low returns by period
        overall_hl_df = pd.merge(
            high_df.groupby('年月')['high_rtn'].mean().reset_index(),
            low_df.groupby('年月')['low_rtn'].mean().reset_index(),
            on='年月', how='inner'
        )
        
        # Calculate overall H-L returns
        overall_hl_df['hl_rtn'] = overall_hl_df['high_rtn'] - overall_hl_df['low_rtn']
        
        # Now calculate industry-adjusted H-L returns
        # Group high and low returns by period and industry
        high_by_ind = high_returns.groupby(['年月', 'industry_code'])['rtn'].mean().reset_index()
        high_by_ind = high_by_ind.rename(columns={'rtn': 'high_rtn'})
        
        low_by_ind = low_returns.groupby(['年月', 'industry_code'])['rtn'].mean().reset_index()
        low_by_ind = low_by_ind.rename(columns={'rtn': 'low_rtn'})
        
        # Merge high and low returns by period and industry
        ind_hl_df = pd.merge(high_by_ind, low_by_ind, on=['年月', 'industry_code'], how='inner')
        
        # Calculate H-L returns by industry
        ind_hl_df['hl_rtn'] = ind_hl_df['high_rtn'] - ind_hl_df['low_rtn']
        
        # Calculate industry-specific H-L statistics
        industry_effects = ind_hl_df.groupby('industry_code')['hl_rtn'].agg([
            'count', 'mean', 'std'
        ]).reset_index()
        
        # Calculate t-statistics for each industry
        industry_effects['t_stat'] = industry_effects.apply(
            lambda x: stats.ttest_1samp(
                ind_hl_df.loc[ind_hl_df['industry_code'] == x['industry_code'], 'hl_rtn'], 0
            ).statistic if x['count'] > 1 else np.nan,
            axis=1
        )
        
        # Calculate p-values for each industry
        industry_effects['p_value'] = industry_effects.apply(
            lambda x: stats.ttest_1samp(
                ind_hl_df.loc[ind_hl_df['industry_code'] == x['industry_code'], 'hl_rtn'], 0
            ).pvalue if x['count'] > 1 else np.nan,
            axis=1
        )
        
        # Sort by industry code
        industry_effects = industry_effects.sort_values('industry_code')
        
        # Calculate average across all industries (industry-adjusted H-L)
        ind_adj_hl_mean = ind_hl_df['hl_rtn'].mean()
        ind_adj_hl_std = ind_hl_df['hl_rtn'].std()
        ind_adj_hl_count = len(ind_hl_df)
        ind_adj_hl_min = ind_hl_df['hl_rtn'].min()
        ind_adj_hl_max = ind_hl_df['hl_rtn'].max()
        
        # Calculate t-statistic for industry-adjusted H-L returns
        ind_adj_hl_tstat, ind_adj_hl_pvalue = stats.ttest_1samp(ind_hl_df['hl_rtn'], 0) if ind_adj_hl_count > 1 else (np.nan, np.nan)
        
        # Calculate statistics for overall (non-industry-adjusted) H-L
        overall_hl_count = len(overall_hl_df)
        overall_hl_mean = overall_hl_df['hl_rtn'].mean()
        overall_hl_std = overall_hl_df['hl_rtn'].std()
        overall_hl_min = overall_hl_df['hl_rtn'].min()
        overall_hl_max = overall_hl_df['hl_rtn'].max()
        
        # Calculate t-statistic for overall H-L returns
        overall_hl_tstat, overall_hl_pvalue = stats.ttest_1samp(overall_hl_df['hl_rtn'], 0) if overall_hl_count > 1 else (np.nan, np.nan)
        
        # Create H-L summary dataframe with both overall and industry-adjusted results
        hl_summary = pd.DataFrame({
            'portfolio': ['H-L (1-10)', 'Industry-Adjusted H-L'],
            'count': [overall_hl_count, ind_adj_hl_count],
            'mean': [overall_hl_mean, ind_adj_hl_mean],
            'std': [overall_hl_std, ind_adj_hl_std],
            'min': [overall_hl_min, ind_adj_hl_min],
            'max': [overall_hl_max, ind_adj_hl_max],
            't_stat': [overall_hl_tstat, ind_adj_hl_tstat],
            'p_value': [overall_hl_pvalue, ind_adj_hl_pvalue]
        })
        
        # Return hl_df as a combination of both for detailed analysis
        hl_df = pd.merge(
            overall_hl_df.rename(columns={'hl_rtn': 'overall_hl_rtn'}),
            ind_hl_df.groupby('年月')['hl_rtn'].mean().reset_index().rename(columns={'hl_rtn': 'ind_adj_hl_rtn'}),
            on='年月', how='outer'
        )
    else:
        # Create empty summary if not enough data
        hl_summary = pd.DataFrame({
            'portfolio': ['H-L (1-10)', 'Industry-Adjusted H-L'],
            'count': [0, 0],
            'mean': [np.nan, np.nan],
            'std': [np.nan, np.nan],
            'min': [np.nan, np.nan],
            'max': [np.nan, np.nan],
            't_stat': [np.nan, np.nan],
            'p_value': [np.nan, np.nan]
        })
        hl_df = None
    
    return group_stats, hl_summary, hl_df, industry_effects


def panel_regression_analysis(df):
    """
    Conduct panel regression with industry fixed-effects.
    
    Args:
        df (pd.DataFrame): Input dataframe with returns and groups
        
    Returns:
        pd.DataFrame: Regression results summary
    """
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        # Prepare data for regression
        reg_df = df.dropna(subset=['rtn']).copy()
        
        # Create dummy for high group (group 1)
        reg_df['high_dummy'] = (reg_df['group'] == 1).astype(int)
        
        # Create dummy for low group (group 10)
        reg_df['low_dummy'] = (reg_df['group'] == 10).astype(int)
        
        # Extract industry code (first 3 digits)
        if 'TEJ產業_代碼' in reg_df.columns:
            reg_df['industry_code'] = reg_df['TEJ產業_代碼'].astype(str).str[:3]
        else:
            print("Warning: 'TEJ產業_代碼' column not found. Using placeholder industry codes.")
            reg_df['industry_code'] = '000'  # Placeholder
        
        # Create industry dummies
        industry_dummies = pd.get_dummies(reg_df['industry_code'], prefix='ind')
        reg_df = pd.concat([reg_df, industry_dummies], axis=1)
        
        # Drop the first industry dummy to avoid the dummy variable trap
        industry_dummy_cols = industry_dummies.columns[1:]
        
        # Formula for regression with industry fixed-effects
        formula = 'rtn ~ high_dummy + low_dummy + ' + ' + '.join(industry_dummy_cols)
        
        # Run regression
        model = ols(formula, data=reg_df).fit()
        
        return model
        
    except ImportError as e:
        print(f"Error: statsmodels package not available. {e}")
        return None
    except Exception as e:
        print(f"Error in panel regression: {e}")
        return None


def save_sorted_data(df, output_path='database/sort.csv'):
    """
    Save the sorted dataframe to a new CSV file.
    
    Args:
        df (pd.DataFrame): Sorted dataframe
        output_path (str): Path to save the output CSV
    """
    try:
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Sorted data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def save_analysis_results(group_stats, hl_summary, hl_df=None, industry_effects=None, 
                          regression_model=None, output_dir='database'):
    """
    Save analysis results to CSV files.
    
    Args:
        group_stats (pd.DataFrame): Group statistics dataframe
        hl_summary (pd.DataFrame): H-L summary dataframe
        hl_df (pd.DataFrame): H-L returns by period dataframe
        industry_effects (pd.DataFrame): Industry fixed-effects analysis
        regression_model (statsmodels.regression.linear_model.RegressionResultsWrapper): Regression model
        output_dir (str): Directory to save output files
    """
    try:
        # Create directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save group statistics
        group_stats.to_csv(output_path / 'group_stats.csv', index=False, encoding='utf-8')
        print(f"Group statistics saved to {output_path / 'group_stats.csv'}")
        
        # Save H-L summary
        hl_summary.to_csv(output_path / 'hl_summary.csv', index=False, encoding='utf-8')
        print(f"H-L summary saved to {output_path / 'hl_summary.csv'}")
        
        # Save H-L returns by period if available
        if hl_df is not None:
            hl_df.to_csv(output_path / 'hl_returns.csv', index=False, encoding='utf-8')
            print(f"H-L returns saved to {output_path / 'hl_returns.csv'}")
            
        # Save industry fixed-effects analysis if available
        if industry_effects is not None and not industry_effects.empty:
            industry_effects.to_csv(output_path / 'industry_effects.csv', index=False, encoding='utf-8')
            print(f"Industry fixed-effects analysis saved to {output_path / 'industry_effects.csv'}")
            
        # Save regression results if available
        if regression_model is not None:
            with open(output_path / 'regression_results.txt', 'w') as f:
                f.write(str(regression_model.summary()))
            print(f"Regression results saved to {output_path / 'regression_results.txt'}")
    except Exception as e:
        print(f"Error saving analysis results: {e}")


def run_sort_process(input_file='database/make.csv', output_file='database/sort.csv'):
    """
    Run the complete sort process from loading to saving.
    
    Args:
        input_file (str): Input CSV file path
        output_file (str): Output CSV file path
        
    Returns:
        bool: Success status
    """
    print(f"Starting sort process from {input_file}")
    
    # Load make data
    make_df = load_make_data(input_file)
    
    if make_df is not None:
        # Standardize ratios by period
        std_df = standardize_ratios_by_period(make_df)
        
        if std_df is not None:
            # Create decile groups
            group_df = create_decile_groups(std_df)
            
            if group_df is not None:
                # Save sorted data
                save_sorted_data(group_df, output_file)
                
                # Analyze returns by group with industry fixed-effects
                group_stats, hl_summary, hl_df, industry_effects = analyze_returns_by_group(group_df)
                
                # Run panel regression with industry fixed-effects
                regression_model = panel_regression_analysis(group_df)
                
                # Save analysis results
                save_analysis_results(group_stats, hl_summary, hl_df, industry_effects, regression_model)
                
                # Print analysis results
                print("\nGroup Statistics:")
                print(group_stats.to_string(index=False))
                
                print("\nHigh-Minus-Low (H-L) Portfolio Summary:")
                print(hl_summary.to_string(index=False))
                
                print("\nIndustry Fixed-Effects Analysis:")
                if industry_effects is not None and not industry_effects.empty:
                    print(industry_effects.to_string(index=False))
                    
                if regression_model is not None:
                    print("\nPanel Regression with Industry Fixed-Effects:")
                    print(regression_model.summary())
                
                return True
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sort and analyze financial data with industry fixed-effects')
    parser.add_argument('--input', default='database/make.csv', help='Input CSV file path')
    parser.add_argument('--output', default='database/sort.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    success = run_sort_process(args.input, args.output)
    
    if success:
        print("Sort process completed successfully!")
    else:
        print("Sort process failed.")