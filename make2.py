#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Make Script
----------
This script creates a processed dataset with financial ratios from the backfilled data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_backfilled_data(file_path='database/first_backfill.csv'):
    """
    Load the backfilled CSV data.
    
    Args:
        file_path (str): Path to the backfilled CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Loaded backfilled data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading backfilled data: {e}")
        return None


def create_financial_ratios(df):
    """
    Create financial ratios from the backfilled data.
    
    Args:
        df (pd.DataFrame): Input dataframe with backfilled values
        
    Returns:
        pd.DataFrame: Dataframe with calculated financial ratios
    """
    if df is None or df.empty:
        return None
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check if all required columns exist
    required_columns = [
        "證券代碼", "年月", "收盤價(元)_月", "公司名稱", 
        "營業毛利", "資產總額", "ROE(A)－稅後", 
        "營業成本", "來自營運之現金流量"
    ]
    
    # 設定現金及約當現金欄位的正確名稱
    cash_column = "  現金及約當現金"  # 前面有兩個空格
    
    # 檢查所有必要欄位是否存在
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns or cash_column not in result_df.columns:
        print(f"Error: Missing required columns: {missing_columns + ([cash_column] if cash_column not in result_df.columns else [])}")
        return None
    
    # Calculate financial ratios
    print("Calculating financial ratios...")
    
    # gpoa = 營業毛利 / 資產總額
    result_df['gpoa'] = result_df['營業毛利'] / result_df['資產總額']
    
    # roe = ROE(A)-稅後
    result_df['roe'] = result_df['ROE(A)－稅後']
    
    # cfoa = 現金及約當現金 / 資產總額
    result_df['cfoa'] = result_df[cash_column] / result_df['資產總額']
    
    # gmar = 營業毛利 / (營業毛利 + 營業成本)
    result_df['gmar'] = result_df['營業毛利'] / (result_df['營業毛利'] + result_df['營業成本'])
    
    # acc = 營業毛利 - 來自營運之現金流量 (修正後的公式)
    result_df['acc'] = result_df['營業毛利'] - result_df['來自營運之現金流量']
    
    # Calculate return (rtn) based on next month's closing price
    print("Calculating returns...")
    # Sort by company and date
    result_df = result_df.sort_values(['證券代碼', '年月'])
    
    # Calculate returns for each company
    result_df['rtn'] = np.nan  # Initialize with NaN
    
    # Group by company and calculate returns
    for company_id in result_df['證券代碼'].unique():
        company_data = result_df[result_df['證券代碼'] == company_id].copy()
        
        # 確保年月是數值型態並按照年月排序
        company_data = company_data.sort_values('年月')
        
        # Shift closing price to get next month's price
        company_data['next_price'] = company_data['收盤價(元)_月'].shift(-1)
        
        # Calculate return
        company_data['rtn'] = (company_data['next_price'] - company_data['收盤價(元)_月']) / company_data['收盤價(元)_月']
        
        # Update the main dataframe
        result_df.loc[result_df['證券代碼'] == company_id, 'rtn'] = company_data['rtn']
        
    # 確保最後一個月份的資料 rtn 為 NaN
    for company_id in result_df['證券代碼'].unique():
        company_mask = result_df['證券代碼'] == company_id
        # 找出該公司的最大年月值
        max_year_month = result_df.loc[company_mask, '年月'].max()
        # 將該公司最大年月的 rtn 設為 NaN
        max_month_mask = company_mask & (result_df['年月'] == max_year_month)
        result_df.loc[max_month_mask, 'rtn'] = np.nan
    
    # Calculate delta values (growth rate compared to 60 periods ago)
    print("Calculating delta values...")
    delta_cols = ['gpoa', 'roe', 'cfoa', 'gmar', 'acc']  # Added acc to delta calculations
    
    for company_id in result_df['證券代碼'].unique():
        company_mask = result_df['證券代碼'] == company_id
        company_data = result_df[company_mask].copy()
        
        # 確保年月排序
        company_data = company_data.sort_values('年月')
        
        for col in delta_cols:
            # 計算延遲60期的值
            lagged_value = company_data[col].shift(60)
            
            # 計算成長率：(current - lagged) / lagged
            delta_col_name = f'delta of {col}'
            company_data[delta_col_name] = (company_data[col] - lagged_value) / lagged_value
            
            # 將計算結果更新到主 dataframe
            result_df.loc[company_mask, delta_col_name] = company_data[delta_col_name]
    
    # Replace infinite values with NaN
    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Select only the required columns
    required_output_columns = [
        "證券代碼", "年月", "收盤價(元)_月", "公司名稱", 
        "TEJ產業_代碼",  # Industry code
        "gpoa", "roe", "cfoa", "gmar", "acc",
        "delta of gpoa", "delta of roe", "delta of cfoa", "delta of gmar", "delta of acc",
        "rtn"
    ]
    final_df = result_df[required_output_columns].copy()
    
    # Check for any remaining NaN values in calculated ratios
    nan_counts = final_df.isna().sum()
    print(f"NaN counts in final dataset:\n{nan_counts}")
    
    # Fill any remaining NaN values in calculated ratios with median values by company
    for company_id in final_df['證券代碼'].unique():
        company_mask = final_df['證券代碼'] == company_id
        for ratio in ['gpoa', 'roe', 'cfoa', 'gmar', 'acc']:
            company_median = final_df.loc[company_mask, ratio].median()
            if not pd.isna(company_median):
                missing_mask = company_mask & final_df[ratio].isna()
                final_df.loc[missing_mask, ratio] = company_median
    
    # For any remaining NaN values, use global median
    for ratio in ['gpoa', 'roe', 'cfoa', 'gmar', 'acc']:
        global_median = final_df[ratio].median()
        final_df[ratio] = final_df[ratio].fillna(global_median)
    
    # Note: 不填補 delta 欄位和 rtn 的 NaN 值，因為這些在開始或結束時期本來就應該是 NaN
    
    print(f"Final dataframe shape: {final_df.shape}")
    return final_df


def save_processed_data(df, output_path='database/make2.csv'):
    """
    Save the processed dataframe to a new CSV file.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        output_path (str): Path to save the output CSV
    """
    try:
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def run_make_process(input_file='database/first_backfill.csv', output_file='database/make2.csv'):
    """
    Run the complete make process from loading to saving.
    
    Args:
        input_file (str): Input CSV file path
        output_file (str): Output CSV file path
        
    Returns:
        bool: Success status
    """
    print(f"Starting make process from {input_file}")
    
    # Load backfilled data
    backfilled_df = load_backfilled_data(input_file)
    
    if backfilled_df is not None:
        # Create financial ratios
        processed_df = create_financial_ratios(backfilled_df)
        
        if processed_df is not None:
            # Save processed data
            save_processed_data(processed_df, output_file)
            
            # Print statistics about the processed data
            print("\nMake Process Statistics:")
            print(f"Total rows: {len(processed_df)}")
            
            # Print summary statistics for the ratio columns
            ratio_cols = ['gpoa', 'roe', 'cfoa', 'gmar', 'acc', 
                         'delta of gpoa', 'delta of roe', 'delta of cfoa', 'delta of gmar', 'delta of acc',
                         'rtn']
            print("\nSummary statistics for financial ratios:")
            print(processed_df[ratio_cols].describe().round(4))
            
            return True
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create financial ratios from backfilled data')
    parser.add_argument('--input', default='database/first_backfill.csv', help='Input CSV file path')
    parser.add_argument('--output', default='database/make2.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    success = run_make_process(args.input, args.output)
    
    if success:
        print("Make process completed successfully!")
    else:
        print("Make process failed.")