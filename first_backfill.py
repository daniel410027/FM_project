#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
First Backfill Script
---------------------
This script performs backfilling operations on missing financial data.
It handles multiple imputation methods based on column characteristics.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(file_path='database/merged_data.csv'):
    """
    Load the CSV data with proper encoding and handling of numeric columns.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        # Load with automatic detection of numeric columns
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Print basic information about the loaded data
        print(f"Loaded data shape: {df.shape}")
        print(f"Missing values before backfill:\n{df.isna().sum()}")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def backfill_financial_data(df):
    """
    Apply various backfilling strategies for financial data.
    
    Args:
        df (pd.DataFrame): Input dataframe with missing values
        
    Returns:
        pd.DataFrame: Dataframe with backfilled values
    """
    if df is None or df.empty:
        return None
    
    # Create a copy to avoid modifying the original
    filled_df = df.copy()
    
    # Group by company and sort by year-month for ordered filling
    numeric_cols = filled_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Dictionary to store different fill methods by column type
    fill_methods = {}
    
    # Financial time-dependent columns - use forward fill first, then backward fill
    financial_cols = [
        '收盤價(元)_月', '市值(百萬元)', '本益比-TEJ', '股價淨值比-TEJ', 
        '現金及約當現金', '資產總額', '負債總額', '股東權益總額', 
        '營業毛利', 'ROA(A)稅後息前', 'ROE(A)－稅後', '每股盈餘', 
        '來自營運之現金流量', '營運槓桿度', '財務槓桿度', 
        '稅後淨利率', '營業成本', 'CAPM_Beta 一年'
    ]
    
    # Map the fill methods - adjust these based on financial domain knowledge
    for col in numeric_cols:
        if col in financial_cols:
            fill_methods[col] = 'financial'
        else:
            fill_methods[col] = 'general'
            
    # Process each company separately
    companies = filled_df['證券代碼'].unique()
    print(f"Processing {len(companies)} unique companies...")
    
    # Store the filled dataframes for each company
    filled_dfs = []
    
    for company_id in companies:
        company_data = filled_df[filled_df['證券代碼'] == company_id].copy()
        
        # Sort by year-month (assuming it's already in YYYYMM format)
        company_data = company_data.sort_values('年月')
        
        # First, handle the TEJ產業_代碼 column - fill with first non-NA value for this company
        if 'TEJ產業_代碼' in company_data.columns:
            first_valid_industry = company_data['TEJ產業_代碼'].dropna().iloc[0] if not company_data['TEJ產業_代碼'].dropna().empty else None
            if first_valid_industry is not None:
                company_data['TEJ產業_代碼'] = company_data['TEJ產業_代碼'].fillna(first_valid_industry)
        
        # Apply different fill methods based on column type
        for col in numeric_cols:
            if col in fill_methods:
                if fill_methods[col] == 'financial':
                    # For financial data: First linear interpolation, then forward fill and backward fill
                    # Linear interpolation works best for values between existing data points
                    company_data[col] = company_data[col].interpolate(method='linear')
                    # Then use forward fill and backward fill for remaining missing values
                    company_data[col] = company_data[col].ffill().bfill()
                elif fill_methods[col] == 'general':
                    # For general data: Use mean within the same company
                    col_mean = company_data[col].mean()
                    company_data[col] = company_data[col].fillna(col_mean)
                    
        filled_dfs.append(company_data)
    
    # Combine all the filled company dataframes
    result_df = pd.concat(filled_dfs)
    
    # For any remaining NaN values across companies, use industry average
    if 'TEJ產業_代碼' in result_df.columns:
        # Handle remaining missing TEJ產業_代碼 with most common value
        if result_df['TEJ產業_代碼'].isna().any():
            most_common_industry = result_df['TEJ產業_代碼'].mode()[0]
            result_df['TEJ產業_代碼'] = result_df['TEJ產業_代碼'].fillna(most_common_industry)
            
        industries = result_df['TEJ產業_代碼'].unique()
        
        for industry in industries:
            industry_mask = result_df['TEJ產業_代碼'] == industry
            industry_data = result_df[industry_mask]
            
            for col in numeric_cols:
                # Calculate industry average for each column
                industry_avg = industry_data[col].mean()
                
                # Fill remaining NaN values with industry average
                missing_mask = result_df[col].isna() & industry_mask
                result_df.loc[missing_mask, col] = industry_avg
    
    # Final cleaning - for any remaining NaN values, use global average
    for col in numeric_cols:
        global_avg = result_df[col].mean()
        result_df[col] = result_df[col].fillna(global_avg)
    
    print(f"Missing values after backfill:\n{result_df.isna().sum()}")
    return result_df


def save_backfilled_data(df, output_path='backfilled_merged_data.csv'):
    """
    Save the backfilled dataframe to a new CSV file.
    
    Args:
        df (pd.DataFrame): Backfilled dataframe
        output_path (str): Path to save the output CSV
    """
    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Backfilled data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def run_backfill_process(input_file='database/merged_data.csv', output_file='database/first_backfill.csv'):
    """
    Run the complete backfill process from loading to saving.
    
    Args:
        input_file (str): Input CSV file path
        output_file (str): Output CSV file path
    """
    print(f"Starting backfill process for {input_file}")
    df = load_data(input_file)
    
    if df is not None:
        filled_df = backfill_financial_data(df)
        if filled_df is not None:
            save_backfilled_data(filled_df, output_file)
            
            # Print statistics about the backfilling
            print("\nBackfill Statistics:")
            print(f"Total rows: {len(filled_df)}")
            print(f"Total columns: {len(filled_df.columns)}")
            return True
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill missing financial data in CSV')
    parser.add_argument('--input', default='database/merged_data.csv', help='Input CSV file path')
    parser.add_argument('--output', default='database/first_backfill.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    success = run_backfill_process(args.input, args.output)
    
    if success:
        print("Backfill process completed successfully!")
    else:
        print("Backfill process failed.")