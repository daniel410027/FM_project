#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Safety Indicator Analysis Test Script
-----------------------------------
This script performs detailed analysis on safety indicators to diagnose issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


def load_data():
    """Load make3 and sort3 data for analysis."""
    try:
        make_df = pd.read_csv('database/make3.csv', encoding='utf-8')
        sort_df = pd.read_csv('database/sort3.csv', encoding='utf-8')
        print(f"Loaded make3 data: {make_df.shape}")
        print(f"Loaded sort3 data: {sort_df.shape}")
        return make_df, sort_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def analyze_safety_components(make_df, sort_df):
    """Analyze individual safety components."""
    print("\n=== SAFETY COMPONENT ANALYSIS ===")
    
    safety_cols = ['bab', 'lev', 'o', 'z', 'evol']
    standardized_cols = ['z_bab', 'z_lev', 'z_o', 'z_z', 'z_evol']
    
    # 1. Basic statistics for raw safety indicators
    print("\n1. Raw Safety Indicators Statistics:")
    raw_stats = make_df[safety_cols + ['rtn']].describe()
    print(raw_stats.round(4))
    
    # 2. Correlation with returns (raw values)
    print("\n2. Raw Safety Indicators Correlation with Returns:")
    raw_corr = make_df[safety_cols + ['rtn']].corr()['rtn'].drop('rtn')
    print(raw_corr.round(4))
    
    # 3. Standardized values correlation
    if all(col in sort_df.columns for col in standardized_cols):
        print("\n3. Standardized Safety Indicators Correlation with Returns:")
        std_corr = sort_df[standardized_cols + ['rtn']].corr()['rtn'].drop('rtn')
        print(std_corr.round(4))
    
    # 4. Check if reversals are working correctly
    print("\n4. Safety Score Components Analysis:")
    if 'safety' in sort_df.columns:
        # Calculate correlation of individual components with final safety score
        safety_components = []
        for col in standardized_cols:
            if col in sort_df.columns:
                component_corr = sort_df[[col, 'safety']].corr().iloc[0, 1]
                safety_components.append((col, component_corr))
        
        print("Component correlation with safety score:")
        for comp, corr in safety_components:
            print(f"  {comp}: {corr:.4f}")
    
    # 5. Distribution analysis
    print("\n5. Distribution Analysis:")
    for col in safety_cols:
        if col in make_df.columns:
            skew = make_df[col].skew()
            kurt = make_df[col].kurtosis()
            print(f"{col}: Skewness={skew:.4f}, Kurtosis={kurt:.4f}")


def analyze_period_effects(sort_df):
    """Analyze safety indicators over time periods."""
    print("\n=== PERIOD EFFECTS ANALYSIS ===")
    
    if 'safety' not in sort_df.columns:
        print("Safety column not found")
        return
    
    # Convert 年月 to datetime for better analysis
    sort_df['year'] = sort_df['年月'] // 100
    sort_df['month'] = sort_df['年月'] % 100
    
    # 1. Safety score by year
    print("\n1. Average Safety Score by Year:")
    yearly_safety = sort_df.groupby('year')['safety'].agg(['mean', 'std', 'count'])
    print(yearly_safety.tail(10).round(4))
    
    # 2. Safety vs Returns by year
    print("\n2. Safety-Return Correlation by Year:")
    yearly_corr = sort_df.groupby('year')[['safety', 'rtn']].corr().iloc[::2, -1]
    yearly_corr.index = yearly_corr.index.get_level_values(0)
    print(yearly_corr.tail(10).round(4))
    
    # 3. Component behavior over time
    print("\n3. Safety Components Over Time (Standardized):")
    standardized_cols = ['z_bab', 'z_lev', 'z_o', 'z_z', 'z_evol']
    available_cols = [col for col in standardized_cols if col in sort_df.columns]
    
    if available_cols:
        component_yearly = sort_df.groupby('year')[available_cols].mean()
        print(component_yearly.tail(5).round(4))


def test_reversal_logic(sort_df):
    """Test if the reversal logic is working correctly."""
    print("\n=== REVERSAL LOGIC TEST ===")
    
    # Check if reversed indicators have opposite signs
    reverse_mapping = {
        'z_o': True,    # Should be reversed
        'z_evol': True  # Should be reversed
    }
    
    print("\n1. Testing Reversal Implementation:")
    for col, should_reverse in reverse_mapping.items():
        if col in sort_df.columns and 'safety' in sort_df.columns:
            # Create temporary safety score with and without this component
            other_cols = ['z_bab', 'z_lev', 'z_z']
            other_cols = [c for c in other_cols if c in sort_df.columns and c != col]
            
            if other_cols:
                temp_safety_without = sort_df[other_cols].sum(axis=1)
                temp_safety_with = temp_safety_without + sort_df[col]
                temp_safety_with_reversed = temp_safety_without - sort_df[col]
                
                # Check correlation
                corr_without_reversal = temp_safety_with.corr(sort_df['rtn'])
                corr_with_reversal = temp_safety_with_reversed.corr(sort_df['rtn'])
                
                print(f"\n{col}:")
                print(f"  Correlation without reversal: {corr_without_reversal:.4f}")
                print(f"  Correlation with reversal: {corr_with_reversal:.4f}")
                print(f"  Should reverse: {should_reverse}")
                print(f"  Reversal improves correlation: {corr_with_reversal > corr_without_reversal}")


def analyze_by_industry(sort_df):
    """Analyze safety indicators by industry."""
    print("\n=== INDUSTRY ANALYSIS ===")
    
    if 'TEJ產業_代碼' not in sort_df.columns:
        print("Industry code not found")
        return
    
    # Extract industry code
    sort_df['industry'] = sort_df['TEJ產業_代碼'].astype(str).str[:3]
    
    # 1. Safety effectiveness by industry
    print("\n1. Safety-Return Correlation by Industry:")
    industry_corr = sort_df.groupby('industry').apply(
        lambda x: x[['safety', 'rtn']].corr().iloc[0, 1] if len(x) > 30 else np.nan
    )
    industry_corr = industry_corr.dropna().sort_values(ascending=False)
    print(industry_corr.head(10).round(4))
    print("...")
    print(industry_corr.tail(10).round(4))
    
    # 2. Average safety score by industry
    print("\n2. Average Safety Score by Industry:")
    industry_safety = sort_df.groupby('industry')['safety'].agg(['mean', 'count'])
    industry_safety = industry_safety[industry_safety['count'] > 100].sort_values('mean', ascending=False)
    print(industry_safety.head(10).round(4))


def suggest_improvements(make_df, sort_df):
    """Suggest improvements based on analysis."""
    print("\n=== IMPROVEMENT SUGGESTIONS ===")
    
    # Analyze each component
    suggestions = []
    
    # 1. Check correlations
    safety_cols = ['bab', 'lev', 'o', 'z', 'evol']
    raw_corr = make_df[safety_cols + ['rtn']].corr()['rtn'].drop('rtn')
    
    print("\n1. Component-wise Suggestions:")
    for col, corr in raw_corr.items():
        if col == 'bab':
            if corr < 0:
                suggestions.append(f"Consider NOT reversing {col} (currently negative correlation)")
        elif col == 'lev':
            if corr < 0:
                suggestions.append(f"Consider reversing {col} (negative correlation)")
        elif col == 'o':
            if corr > 0:
                suggestions.append(f"Confirm {col} is properly reversed (positive correlation)")
        elif col == 'z':
            if corr < 0:
                suggestions.append(f"Consider reversing {col} (negative correlation)")
        elif col == 'evol':
            if corr > 0:
                suggestions.append(f"Confirm {col} is properly reversed (positive correlation)")
    
    for suggestion in suggestions:
        print(f"  - {suggestion}")
    
    # 2. Overall safety effectiveness
    if 'safety' in sort_df.columns:
        safety_corr = sort_df[['safety', 'rtn']].corr().iloc[0, 1]
        print(f"\n2. Overall Safety Score Correlation: {safety_corr:.4f}")
        
        if safety_corr < 0:
            print("  - Safety score is negatively correlated with returns")
            print("  - Consider removing safety from the composite score")
            print("  - Or completely redesign the safety factor")
    
    # 3. Missing value impact
    print("\n3. Missing Value Impact:")
    missing_pct = sort_df[['z_bab', 'z_lev', 'z_o', 'z_z', 'z_evol']].isnull().sum() / len(sort_df) * 100
    for col, pct in missing_pct.items():
        if pct > 10:
            print(f"  - {col}: {pct:.1f}% missing (consider imputation or exclusion)")


def create_visualizations(make_df, sort_df):
    """Create visualizations for safety analysis."""
    output_dir = Path('safety_analysis_plots')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Correlation heatmap
    plt.figure(figsize=(10, 8))
    safety_cols = ['bab', 'lev', 'o', 'z', 'evol', 'rtn']
    available_cols = [col for col in safety_cols if col in make_df.columns]
    corr_matrix = make_df[available_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Safety Indicators Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'safety_correlation_heatmap.png')
    plt.close()
    
    # 2. Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(['bab', 'lev', 'o', 'z', 'evol']):
        if col in make_df.columns and i < len(axes):
            data = make_df[col].dropna()
            # Winsorize extreme values for better visualization
            lower = data.quantile(0.01)
            upper = data.quantile(0.99)
            data_winsorized = data.clip(lower=lower, upper=upper)
            
            axes[i].hist(data_winsorized, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col} Distribution (1-99 percentile)')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide extra subplot if any
    if len(['bab', 'lev', 'o', 'z', 'evol']) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'safety_distributions.png')
    plt.close()
    
    # 3. Time series of safety score
    if 'safety' in sort_df.columns:
        plt.figure(figsize=(12, 6))
        monthly_safety = sort_df.groupby('年月')['safety'].mean()
        monthly_safety.plot()
        plt.title('Average Safety Score Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel('Average Safety Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'safety_time_series.png')
        plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")


def generate_report(make_df, sort_df):
    """Generate a comprehensive report."""
    print("\n" + "="*50)
    print("SAFETY INDICATOR ANALYSIS REPORT")
    print("="*50)
    
    # Run all analyses
    analyze_safety_components(make_df, sort_df)
    analyze_period_effects(sort_df)
    test_reversal_logic(sort_df)
    analyze_by_industry(sort_df)
    suggest_improvements(make_df, sort_df)
    create_visualizations(make_df, sort_df)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETED")
    print("="*50)


def main():
    # Load data
    make_df, sort_df = load_data()
    
    if make_df is not None and sort_df is not None:
        generate_report(make_df, sort_df)
    else:
        print("Failed to load data. Please check file paths.")


if __name__ == "__main__":
    main()