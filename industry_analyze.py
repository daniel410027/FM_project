#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Script
----------
This script reads the processed financial data and creates a pie chart
of the TEJ產業_代碼 (TEJ Industry Code) distribution, categorized by the first 3 characters.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib as mpl

def load_processed_data(file_path='database/make.csv'):
    """
    Load the processed CSV data with financial ratios.
    
    Args:
        file_path (str): Path to the processed CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"Loaded processed data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None

def create_industry_pie_chart(df, output_dir='plots'):
    """
    Create a pie chart showing the distribution of TEJ產業_代碼,
    grouped by the first 3 characters of each code.
    
    Args:
        df (pd.DataFrame): Input dataframe with TEJ產業_代碼
        output_dir (str): Directory to save the output plot
        
    Returns:
        bool: Success status
    """
    if df is None or df.empty or 'TEJ產業_代碼' not in df.columns:
        print("Error: DataFrame is empty or missing 'TEJ產業_代碼' column")
        return False
    
    # Configure matplotlib for Chinese characters
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Extract the first 3 characters for categorization
    latest_date = df['年月'].max()
    latest_df = df[df['年月'] == latest_date].copy()
    
    # Create new column with first 3 chars of industry code
    latest_df['產業分類'] = latest_df['TEJ產業_代碼'].astype(str).apply(lambda x: x[:3] if len(str(x)) >= 3 else x)
    
    # Count by industry group
    industry_counts = latest_df['產業分類'].value_counts()
    
    # If there are too many categories, combine the smaller ones
    if len(industry_counts) > 10:
        # Keep top 9 industries, group the rest as "Other"
        top_industries = industry_counts.nlargest(9)
        other_count = industry_counts[9:].sum()
        
        # Create a new Series with "Other" category
        other_series = pd.Series({'其他': other_count})
        industry_counts = pd.concat([top_industries, other_series])
    
    # Create figure with larger size for better readability
    plt.figure(figsize=(12, 8))
    
    # Create a pie chart
    wedges, texts, autotexts = plt.pie(
        industry_counts, 
        labels=industry_counts.index,
        autopct='%1.1f%%',  # Show percentage with 1 decimal place
        startangle=90,
        shadow=True,
        explode=[0.05] * len(industry_counts),  # Slightly explode all slices
    )
    
    # Set font properties for better Chinese character display
    for text in texts + autotexts:
        text.set_fontsize(12)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    
    # Add title with appropriate font for Chinese characters
    plt.title('TEJ產業代碼分布 (前三碼分類)', fontsize=16)
    
    # Create directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    output_path = f"{output_dir}/industry_distribution_pie.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    print(f"Pie chart saved to {output_path}")
    
    # Also print the distribution table
    print("\n產業分布統計表 (前三碼分類):")
    industry_table = pd.DataFrame({
        '產業前三碼': industry_counts.index,
        '公司數量': industry_counts.values,
        '占比 (%)': (industry_counts.values / industry_counts.sum() * 100).round(2)
    })
    print(industry_table.to_string(index=False))
    
    return True

def run_test():
    """
    Run the complete test process.
    
    Returns:
        bool: Success status
    """
    print("Starting test process")
    
    # Load processed data
    processed_df = load_processed_data()
    
    if processed_df is not None:
        # Check if TEJ產業_代碼 column exists
        if 'TEJ產業_代碼' not in processed_df.columns:
            print("Error: 'TEJ產業_代碼' column not found in processed data")
            # Print available columns for debugging
            print("Available columns:", processed_df.columns.tolist())
            return False
        
        # Create industry pie chart
        success = create_industry_pie_chart(processed_df)
        
        # Print additional statistics
        if success:
            # Create a new column for the first 3 characters
            processed_df['產業前三碼'] = processed_df['TEJ產業_代碼'].astype(str).apply(
                lambda x: x[:3] if len(str(x)) >= 3 else x
            )
            
            # Number of unique industry categories
            unique_industries = processed_df['產業前三碼'].nunique()
            print(f"\n總共有 {unique_industries} 個不同的產業類別 (前三碼分類)")
            
            # Analyze company count by industry
            industry_company_counts = processed_df.groupby('產業前三碼')['證券代碼'].nunique().sort_values(ascending=False)
            print("\n各產業公司數量 (由高至低):")
            for industry, count in industry_company_counts.items():
                print(f"- {industry}: {count} 家公司")
            
            return True
        
    return False

if __name__ == "__main__":
    import sys
    
    # Allow custom input file path as command-line argument
    if len(sys.argv) > 1:
        success = run_test(sys.argv[1])
    else:
        success = run_test()
    
    if success:
        print("Test process completed successfully!")
    else:
        print("Test process failed.")