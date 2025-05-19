#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sort4 Script
-----------
This script standardizes both level and delta financial ratios by period, 
creates composite scores for profitability, growth, and safety, sorts stocks into deciles, 
and analyzes returns by group with industry fixed-effects.
Uses the make4.csv file which includes current ratio in O-Score calculation.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

def load_make_data(file_path='database/make4.csv'):
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
        
        # Filter data to include only periods after 200600 for more complete data
        df_filtered = df[df['年月'] > 200600].copy()
        print(f"Filtered data shape (年月 > 200600): {df_filtered.shape}")
        
        return df_filtered
    except Exception as e:
        print(f"Error loading financial data: {e}")
        return None


def standardize_ratios_by_period(df):
    """
    Standardize financial ratios and their deltas by period (年月).
    
    Args:
        df (pd.DataFrame): Input dataframe with financial ratios
        
    Returns:
        pd.DataFrame: Dataframe with standardized ratios
    """
    if df is None or df.empty:
        return None
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Level ratio columns to standardize for profitability
    profitability_cols = ['gpoa', 'roe', 'cfoa', 'gmar', 'acc']
    
    # Delta ratio columns to standardize for growth
    growth_cols = ['delta of gpoa', 'delta of roe', 'delta of cfoa', 
                   'delta of gmar', 'delta of acc']
    
    # Safety ratio columns to standardize
    safety_cols = ['bab', 'lev', 'o', 'z', 'evol']
    
    # Define which safety indicators need to be reversed
    reverse_safety = {
        'bab': False,  # -0.0041 負相關，不反向
        'lev': False,   # -0.0005 負相關，需要反向
        'o': False,    # -0.0009 負相關，但反向後更差，不反向
        'z': False,    # -0.0009 負相關，分布問題嚴重，不反向
        'evol': False  # +0.0033 正相關，不反向
    }
    
    # Check if all required columns exist
    required_columns = ["證券代碼", "年月", "公司名稱", "rtn"] + profitability_cols + growth_cols + safety_cols
    
    # Check all necessary columns
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None
    
    # Create standardized columns
    print("Standardizing financial ratios by period...")
    
    # Initialize all standardized columns
    all_cols = profitability_cols + growth_cols + safety_cols
    for col in all_cols:
        result_df[f'z_{col}'] = np.nan
    
    # Group by period and standardize
    for period in result_df['年月'].unique():
        period_mask = result_df['年月'] == period
        
        # Standardize all ratio columns
        for col in all_cols:
            # Get values for the current period
            values = result_df.loc[period_mask, col]
            
            # For delta columns, handle NaN values properly
            if col in growth_cols:
                valid_values = values.dropna()
                if len(valid_values) > 0:
                    mean = valid_values.mean()
                    std = valid_values.std()
                    
                    if std > 0:
                        z_scores = (values - mean) / std
                        result_df.loc[period_mask, f'z_{col}'] = z_scores
                    else:
                        result_df.loc[period_mask, f'z_{col}'] = 0
                else:
                    result_df.loc[period_mask, f'z_{col}'] = np.nan
            else:
                # For non-delta columns
                mean = values.mean()
                std = values.std()
                
                if std > 0:
                    z_scores = (values - mean) / std
                    result_df.loc[period_mask, f'z_{col}'] = z_scores
                else:
                    result_df.loc[period_mask, f'z_{col}'] = 0
    
    # Create profitability score
    print("Creating profitability score...")
    result_df['profitability'] = sum(result_df[f'z_{col}'] for col in profitability_cols)
    
    # Create growth score
    print("Creating growth score...")
    result_df['growth'] = sum(result_df[f'z_{col}'] for col in growth_cols)
    
    # Create safety score
    print("Creating safety score...")
    safety_scores = []
    for col in safety_cols:
        if reverse_safety.get(col, False):
            # 反向指標：用負值
            safety_scores.append(-result_df[f'z_{col}'])
        else:
            safety_scores.append(result_df[f'z_{col}'])
    result_df['safety'] = sum(safety_scores)
    
    # Create combined score (equal weights)
    print("Creating combined score...")
    result_df['score'] = result_df['profitability'] + result_df['growth'] + result_df['safety']
    
    # 處理 inf 和 -inf 值
    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return result_df


def create_decile_groups(df, score_column='score'):
    """
    Create decile groups based on composite score by period.
    
    Args:
        df (pd.DataFrame): Input dataframe with standardized ratios and score
        score_column (str): Column name to use for sorting
        
    Returns:
        pd.DataFrame: Dataframe with decile group assignments
    """
    if df is None or df.empty:
        return None
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Create decile groups by period
    print(f"Creating decile groups by period based on {score_column}...")
    
    # Initialize the group column as numeric
    result_df['group'] = np.nan
    
    # Group by period and assign decile groups
    for period in result_df['年月'].unique():
        period_data = result_df[result_df['年月'] == period].copy()
        
        # Filter out rows with NaN scores
        valid_period_data = period_data.dropna(subset=[score_column])
        
        if len(valid_period_data) >= 10:  # Ensure enough data for 10 groups
            # Create decile based on score (10 groups, with 1 being highest score)
            try:
                # Calculate decile groups and map to integers 1-10 (reversed so 1 is highest score)
                valid_period_data['temp_group'] = pd.qcut(
                    valid_period_data[score_column], 
                    10, 
                    labels=False,  # Use numeric labels
                    duplicates='drop'
                )
                # Reverse order so 1 is highest score (10-group)
                valid_period_data['temp_group'] = 10 - valid_period_data['temp_group']
                
                # Update the main dataframe with the calculated groups
                for idx in valid_period_data.index:
                    result_df.loc[idx, 'group'] = valid_period_data.loc[idx, 'temp_group']
                    
            except ValueError as e:
                print(f"Warning: Could not create 10 groups for period {period}: {e}")
                # If we can't create exactly 10 groups, try with fewer bins
                try:
                    # Count unique values
                    unique_scores = valid_period_data[score_column].nunique()
                    if unique_scores >= 2:
                        # Use as many bins as unique values, max 10
                        n_bins = min(unique_scores, 10)
                        valid_period_data['temp_group'] = pd.qcut(
                            valid_period_data[score_column], 
                            n_bins, 
                            labels=False, 
                            duplicates='drop'
                        )
                        # Reverse order and rescale to 1-10 range
                        valid_period_data['temp_group'] = n_bins - valid_period_data['temp_group']
                        # Map to 1-10 scale
                        valid_period_data['temp_group'] = valid_period_data['temp_group'].map(
                            lambda x: int(1 + (x - 1) * (10 - 1) / (n_bins - 1)) if n_bins > 1 else 5
                        )
                        for idx in valid_period_data.index:
                            result_df.loc[idx, 'group'] = valid_period_data.loc[idx, 'temp_group']
                    else:
                        # All scores are the same, assign to middle group (5)
                        for idx in valid_period_data.index:
                            result_df.loc[idx, 'group'] = 5
                except Exception as inner_e:
                    print(f"Error creating groups for period {period}: {inner_e}")
                    # Assign to middle group as fallback
                    for idx in valid_period_data.index:
                        result_df.loc[idx, 'group'] = 5
        else:
            # Not enough data for 10 groups
            print(f"Warning: Period {period} has fewer than 10 stocks with valid scores")
            for idx in valid_period_data.index:
                result_df.loc[idx, 'group'] = 5
    
    # Ensure group is integer type for non-NaN values
    result_df.loc[result_df['group'].notna(), 'group'] = result_df.loc[result_df['group'].notna(), 'group'].astype(int)
    
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
    
    # Drop rows with NaN returns or groups, and filter for 年月 > 200600
    analysis_df = df[(df['年月'] > 200600)].dropna(subset=['rtn', 'group']).copy()
    
    print(f"Analyzing returns by group (年月 > 200600)...")
    print(f"Analysis data shape: {analysis_df.shape}")
    
    # Extract industry code (first 3 digits of TEJ產業_代碼)
    if 'TEJ產業_代碼' in analysis_df.columns:
        # 先轉換所有代碼為字串類型
        analysis_df['TEJ產業_代碼_str'] = analysis_df['TEJ產業_代碼'].astype(str)
        
        # 創建條件遮罩 (mask) 來識別M23開頭的代碼
        is_m23 = analysis_df['TEJ產業_代碼_str'].str.startswith('M23')
        
        # 對M23開頭的代碼取前4位，其他取前3位
        analysis_df['industry_code'] = np.where(
            is_m23,
            analysis_df['TEJ產業_代碼_str'].str[:4],  # 若為M23則取前4位
            analysis_df['TEJ產業_代碼_str'].str[:3]   # 其他取前3位
        )
        
        # 移除臨時欄位
        analysis_df.drop('TEJ產業_代碼_str', axis=1, inplace=True)
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


def save_sorted_data(df, output_path='database/sort4.csv'):
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
                          output_dir='database'):
    """
    Save analysis results to CSV files.
    
    Args:
        group_stats (pd.DataFrame): Group statistics dataframe
        hl_summary (pd.DataFrame): H-L summary dataframe
        hl_df (pd.DataFrame): H-L returns by period dataframe
        industry_effects (pd.DataFrame): Industry fixed-effects analysis
        output_dir (str): Directory to save output files
    """
    try:
        # Create directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save group statistics
        group_stats.to_csv(output_path / 'group_stats4.csv', index=False, encoding='utf-8')
        print(f"Group statistics saved to {output_path / 'group_stats4.csv'}")
        
        # Save H-L summary
        hl_summary.to_csv(output_path / 'hl_summary4.csv', index=False, encoding='utf-8')
        print(f"H-L summary saved to {output_path / 'hl_summary4.csv'}")
        
        # Save H-L returns by period if available
        if hl_df is not None:
            hl_df.to_csv(output_path / 'hl_returns4.csv', index=False, encoding='utf-8')
            print(f"H-L returns saved to {output_path / 'hl_returns4.csv'}")
            
        # Save industry fixed-effects analysis if available
        if industry_effects is not None and not industry_effects.empty:
            industry_effects.to_csv(output_path / 'industry_effects4.csv', index=False, encoding='utf-8')
            print(f"Industry fixed-effects analysis saved to {output_path / 'industry_effects4.csv'}")
    except Exception as e:
        print(f"Error saving analysis results: {e}")


def run_sort_process(input_file='database/make4.csv', output_file='database/sort4.csv'):
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
            # Create decile groups based on combined score
            group_df = create_decile_groups(std_df, score_column='score')
            
            if group_df is not None:
                # Save sorted data
                save_sorted_data(group_df, output_file)
                
                # Analyze returns by group with industry fixed-effects
                group_stats, hl_summary, hl_df, industry_effects = analyze_returns_by_group(group_df)
                
                # Save analysis results
                save_analysis_results(group_stats, hl_summary, hl_df, industry_effects)
                
                # Print analysis results
                print("\nGroup Statistics:")
                print(group_stats.to_string(index=False))
                
                print("\nHigh-Minus-Low (H-L) Portfolio Summary:")
                print(hl_summary.to_string(index=False))
                
                print("\nIndustry Fixed-Effects Analysis:")
                if industry_effects is not None and not industry_effects.empty:
                    print(industry_effects.to_string(index=False))
                
                # Create more meaningful factor analysis
                print("\nFactor Analysis:")
                
                # 1. 各因子與報酬的相關性
                correlation_df = std_df[['profitability', 'growth', 'safety', 'score', 'rtn']].corr()
                print("\nCorrelation with Returns:")
                print(correlation_df['rtn'].drop('rtn').round(4))
                
                # 2. 各期間的平均分數
                period_summary = std_df.groupby('年月')[['profitability', 'growth', 'safety', 'score']].mean()
                print("\nAverage Scores by Period (last 5 periods):")
                print(period_summary.tail().round(4))
                
                # 3. 缺失值統計
                print("\nMissing Value Counts:")
                missing_counts = std_df[['profitability', 'growth', 'safety', 'score', 'rtn']].isnull().sum()
                print(missing_counts)
                
                # 4. 分組報酬分析
                print("\nAverage Returns by Score Quintile:")
                std_df_valid = std_df[std_df['score'].notna() & std_df['rtn'].notna()]
                std_df_valid['score_quintile'] = pd.qcut(std_df_valid['score'], 5, labels=['Low', 'Q2', 'Q3', 'Q4', 'High'], duplicates='drop')
                quintile_returns = std_df_valid.groupby('score_quintile', observed=True)['rtn'].mean()
                print(quintile_returns.round(4))
                
                # 5. 比較 TEJ產業_代碼 的分類方式對產業效應的影響
                # 使用原始三碼分類
                if '產業分類' in group_df.columns and 'TEJ產業_代碼' in group_df.columns:
                    print("\n產業代碼與報酬相關性分析:")
                    # 每個產業代碼的平均報酬
                    ind_code_returns = group_df.dropna(subset=['rtn']).groupby('TEJ產業_代碼')['rtn'].agg(['mean', 'count'])
                    ind_code_returns = ind_code_returns[ind_code_returns['count'] >= 30].sort_values('mean', ascending=False)
                    print(f"\n平均報酬最高的前5個產業代碼:")
                    print(ind_code_returns.head().round(4))
                    print(f"\n平均報酬最低的前5個產業代碼:")
                    print(ind_code_returns.tail().round(4))
                
                return True
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sort and analyze financial data with profitability, growth and safety scores')
    parser.add_argument('--input', default='database/make4.csv', help='Input CSV file path')
    parser.add_argument('--output', default='database/sort4.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    success = run_sort_process(args.input, args.output)
    
    if success:
        print("Sort process completed successfully!")
    else:
        print("Sort process failed.")