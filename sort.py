#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sort Script
----------
This script standardizes financial ratios by period, creates a composite score,
sorts stocks into deciles, and analyzes returns by group.
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
    Analyze returns by decile group.
    
    Args:
        df (pd.DataFrame): Input dataframe with decile groups
        
    Returns:
        tuple: (summary_df, difference_df, ttest_results)
    """
    if df is None or df.empty:
        return None, None, None
    
    # Drop rows with NaN returns (last period for each company)
    analysis_df = df.dropna(subset=['rtn']).copy()
    
    print("Analyzing returns by group...")
    
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
    high_returns = analysis_df.loc[analysis_df['group'] == 1, 'rtn']
    low_returns = analysis_df.loc[analysis_df['group'] == 10, 'rtn']
    
    if not high_returns.empty and not low_returns.empty:
        # Calculate high-minus-low returns for each period
        high_df = analysis_df.loc[analysis_df['group'] == 1, ['年月', 'rtn']].rename(columns={'rtn': 'high_rtn'})
        low_df = analysis_df.loc[analysis_df['group'] == 10, ['年月', 'rtn']].rename(columns={'rtn': 'low_rtn'})
        
        # Merge high and low returns by period
        hl_df = pd.merge(high_df, low_df, on='年月', how='inner')
        
        # Calculate H-L returns
        hl_df['hl_rtn'] = hl_df['high_rtn'] - hl_df['low_rtn']
        
        # Calculate H-L statistics
        hl_count = len(hl_df)
        hl_mean = hl_df['hl_rtn'].mean()
        hl_std = hl_df['hl_rtn'].std()
        hl_min = hl_df['hl_rtn'].min()
        hl_max = hl_df['hl_rtn'].max()
        
        # Calculate t-statistic for H-L returns different from zero
        hl_tstat, hl_pvalue = stats.ttest_1samp(hl_df['hl_rtn'], 0) if hl_count > 1 else (np.nan, np.nan)
        
        # Create H-L summary dataframe
        hl_summary = pd.DataFrame({
            'portfolio': ['H-L (1-10)'],
            'count': [hl_count],
            'mean': [hl_mean],
            'std': [hl_std],
            'min': [hl_min],
            'max': [hl_max],
            't_stat': [hl_tstat],
            'p_value': [hl_pvalue]
        })
    else:
        hl_summary = pd.DataFrame({
            'portfolio': ['H-L (1-10)'],
            'count': [0],
            'mean': [np.nan],
            'std': [np.nan],
            'min': [np.nan],
            'max': [np.nan],
            't_stat': [np.nan],
            'p_value': [np.nan]
        })
    
    return group_stats, hl_summary, hl_df if 'hl_df' in locals() else None


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


def save_analysis_results(group_stats, hl_summary, hl_df=None, output_dir='database'):
    """
    Save analysis results to CSV files.
    
    Args:
        group_stats (pd.DataFrame): Group statistics dataframe
        hl_summary (pd.DataFrame): H-L summary dataframe
        hl_df (pd.DataFrame): H-L returns by period dataframe
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
                
                # Analyze returns by group
                group_stats, hl_summary, hl_df = analyze_returns_by_group(group_df)
                
                # Save analysis results
                save_analysis_results(group_stats, hl_summary, hl_df)
                
                # Print analysis results
                print("\nGroup Statistics:")
                print(group_stats.to_string(index=False))
                
                print("\nHigh-Minus-Low (H-L) Portfolio Summary:")
                print(hl_summary.to_string(index=False))
                
                return True
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sort and analyze financial data')
    parser.add_argument('--input', default='database/make.csv', help='Input CSV file path')
    parser.add_argument('--output', default='database/sort.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    success = run_sort_process(args.input, args.output)
    
    if success:
        print("Sort process completed successfully!")
    else:
        print("Sort process failed.")