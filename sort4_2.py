#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sort4 Script - 標準化金融比率、創建綜合得分並分析收益，包含產業M23細分分析
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path

def load_data(file_path='database/make4.csv'):
    """載入金融數據並過濾"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        return df[df['年月'] > 200600].copy()
    except Exception as e:
        print(f"載入數據錯誤: {e}")
        return None

def standardize_ratios(df):
    """按期間標準化財務比率"""
    if df is None or df.empty:
        return None
    
    result_df = df.copy()
    
    # 定義各組比率
    ratios = {
        'profitability': ['gpoa', 'roe', 'cfoa', 'gmar', 'acc'],
        'growth': ['delta of gpoa', 'delta of roe', 'delta of cfoa', 
                  'delta of gmar', 'delta of acc'],
        'safety': ['bab', 'lev', 'o', 'z', 'evol']
    }
    
    # 需要反向的安全指標
    reverse_safety = {'lev': True, 'bab': False, 'o': False, 'z': False, 'evol': False}
    
    # 檢查必要列
    all_cols = sum(ratios.values(), [])
    required_columns = ["證券代碼", "年月", "公司名稱", "rtn"] + all_cols
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        print(f"缺少必要列: {missing_columns}")
        return None
    
    # 初始化標準化列
    for col in all_cols:
        result_df[f'z_{col}'] = np.nan
    
    # 按期間標準化
    for period in result_df['年月'].unique():
        period_mask = result_df['年月'] == period
        
        for col in all_cols:
            values = result_df.loc[period_mask, col]
            is_growth = col in ratios['growth']
            
            # 處理增長指標的NaN值
            if is_growth:
                valid_values = values.dropna()
                if len(valid_values) == 0:
                    continue
                values_to_use = valid_values
            else:
                values_to_use = values
            
            mean = values_to_use.mean()
            std = values_to_use.std()
            
            if std > 0:
                z_scores = (values - mean) / std
                result_df.loc[period_mask, f'z_{col}'] = z_scores
            else:
                result_df.loc[period_mask, f'z_{col}'] = 0
    
    # 創建綜合得分
    result_df['profitability'] = sum(result_df[f'z_{col}'] for col in ratios['profitability'])
    result_df['growth'] = sum(result_df[f'z_{col}'] for col in ratios['growth'])
    
    # 安全得分 (考慮反向指標)
    safety_scores = []
    for col in ratios['safety']:
        if reverse_safety.get(col, False):
            safety_scores.append(-result_df[f'z_{col}'])
        else:
            safety_scores.append(result_df[f'z_{col}'])
    result_df['safety'] = sum(safety_scores)
    
    # 綜合得分
    result_df['score'] = result_df['profitability'] + result_df['growth'] + result_df['safety']
    
    # 處理無限值
    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return result_df

def create_decile_groups(df, score_column='score'):
    """基於綜合得分創建十分位組別"""
    if df is None or df.empty:
        return None
    
    result_df = df.copy()
    result_df['group'] = np.nan
    
    for period in result_df['年月'].unique():
        period_data = result_df[result_df['年月'] == period].copy()
        valid_data = period_data.dropna(subset=[score_column])
        
        if len(valid_data) >= 10:
            try:
                # 創建十分位 (1為最高分)
                valid_data['temp_group'] = pd.qcut(
                    valid_data[score_column], 10, labels=False, duplicates='drop')
                # 反轉順序使1為最高分
                valid_data['temp_group'] = 10 - valid_data['temp_group']
                
                # 更新主數據框
                result_df.loc[valid_data.index, 'group'] = valid_data['temp_group']
            except ValueError:
                # 處理無法創建10組的情況
                unique_scores = valid_data[score_column].nunique()
                if unique_scores >= 2:
                    n_bins = min(unique_scores, 10)
                    valid_data['temp_group'] = pd.qcut(
                        valid_data[score_column], n_bins, labels=False, duplicates='drop')
                    valid_data['temp_group'] = n_bins - valid_data['temp_group']
                    
                    # 映射到1-10範圍
                    scale_factor = (10 - 1) / (n_bins - 1) if n_bins > 1 else 0
                    valid_data['temp_group'] = valid_data['temp_group'].apply(
                        lambda x: int(1 + (x * scale_factor)))
                    
                    result_df.loc[valid_data.index, 'group'] = valid_data['temp_group']
                else:
                    # 所有分數相同，分配到中間組
                    result_df.loc[valid_data.index, 'group'] = 5
        else:
            # 數據不足10個股票
            result_df.loc[valid_data.index, 'group'] = 5
    
    # 確保組別為整數
    result_df.loc[result_df['group'].notna(), 'group'] = result_df.loc[result_df['group'].notna(), 'group'].astype(int)
    
    return result_df

def extract_industry_codes(df):
    """提取並處理產業代碼"""
    result_df = df.copy()
    
    # 提取 TEJ 產業代碼 (原始三碼和M23細分)
    if 'TEJ產業_代碼' in result_df.columns:
        # 原始三碼
        result_df['industry_code'] = result_df['TEJ產業_代碼'].astype(str).str[:3]
        
        # M23 產業細分 (前三碼 + 第四碼中的細分行業)
        # 例如：若代碼為"M2301"，則提取"M23-01"
        result_df['m23_code'] = result_df['TEJ產業_代碼'].astype(str).apply(
            lambda x: f"{x[:3]}-{x[3:5]}" if x.startswith('M23') and len(x) >= 5 else None)
    else:
        result_df['industry_code'] = '000'  # 預設值
        result_df['m23_code'] = None
    
    return result_df

def analyze_returns(df):
    """分析各組收益與產業固定效應"""
    if df is None or df.empty:
        return None, None, None, None, None
    
    # 移除NaN值
    analysis_df = df.dropna(subset=['rtn', 'group']).copy()
    
    # 計算各組統計數據
    group_stats = analysis_df.groupby('group')['rtn'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).reset_index()
    
    # 計算t統計量
    for idx, row in group_stats.iterrows():
        if row['count'] > 1:
            group_returns = analysis_df.loc[analysis_df['group'] == row['group'], 'rtn']
            t_result = stats.ttest_1samp(group_returns, 0)
            group_stats.loc[idx, 't_stat'] = t_result.statistic
            group_stats.loc[idx, 'p_value'] = t_result.pvalue
        else:
            group_stats.loc[idx, 't_stat'] = np.nan
            group_stats.loc[idx, 'p_value'] = np.nan
    
    # 高減低投資組合分析
    high_returns = analysis_df.loc[analysis_df['group'] == 1, ['年月', 'rtn', 'industry_code', 'm23_code']]
    low_returns = analysis_df.loc[analysis_df['group'] == 10, ['年月', 'rtn', 'industry_code', 'm23_code']]
    
    if high_returns.empty or low_returns.empty:
        # 數據不足時返回空結果
        return group_stats, pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame()
    
    # 計算整體高減低收益
    high_mean = high_returns.groupby('年月')['rtn'].mean().rename('high_rtn')
    low_mean = low_returns.groupby('年月')['rtn'].mean().rename('low_rtn')
    overall_hl = pd.merge(high_mean, low_mean, on='年月')
    overall_hl['hl_rtn'] = overall_hl['high_rtn'] - overall_hl['low_rtn']
    
    # 計算產業調整後高減低收益（使用原始三碼產業）
    high_by_ind = high_returns.groupby(['年月', 'industry_code'])['rtn'].mean().rename('high_rtn')
    low_by_ind = low_returns.groupby(['年月', 'industry_code'])['rtn'].mean().rename('low_rtn')
    ind_hl = pd.merge(high_by_ind, low_by_ind, on=['年月', 'industry_code'])
    ind_hl['hl_rtn'] = ind_hl['high_rtn'] - ind_hl['low_rtn']
    
    # 計算M23產業細分調整後高減低收益
    # 只處理有M23代碼的數據
    high_m23 = high_returns.dropna(subset=['m23_code'])
    low_m23 = low_returns.dropna(subset=['m23_code'])
    
    if not high_m23.empty and not low_m23.empty:
        high_by_m23 = high_m23.groupby(['年月', 'm23_code'])['rtn'].mean().rename('high_rtn')
        low_by_m23 = low_m23.groupby(['年月', 'm23_code'])['rtn'].mean().rename('low_rtn')
        m23_hl = pd.merge(high_by_m23, low_by_m23, on=['年月', 'm23_code'])
        m23_hl['hl_rtn'] = m23_hl['high_rtn'] - m23_hl['low_rtn']
        
        # 計算M23產業效應
        m23_effects = m23_hl.reset_index().groupby('m23_code')['hl_rtn'].agg([
            'count', 'mean', 'std'
        ]).reset_index()
        
        # 計算t統計量
        for idx, row in m23_effects.iterrows():
            if row['count'] > 1:
                m23_returns = m23_hl.loc[m23_hl.index.get_level_values('m23_code') == row['m23_code'], 'hl_rtn']
                t_result = stats.ttest_1samp(m23_returns, 0)
                m23_effects.loc[idx, 't_stat'] = t_result.statistic
                m23_effects.loc[idx, 'p_value'] = t_result.pvalue
            else:
                m23_effects.loc[idx, 't_stat'] = np.nan
                m23_effects.loc[idx, 'p_value'] = np.nan
    else:
        m23_hl = pd.DataFrame(columns=['年月', 'm23_code', 'high_rtn', 'low_rtn', 'hl_rtn'])
        m23_effects = pd.DataFrame(columns=['m23_code', 'count', 'mean', 'std', 't_stat', 'p_value'])
    
    # 計算產業效應
    industry_effects = ind_hl.reset_index().groupby('industry_code')['hl_rtn'].agg([
        'count', 'mean', 'std'
    ]).reset_index()
    
    # 計算t統計量
    for idx, row in industry_effects.iterrows():
        if row['count'] > 1:
            ind_returns = ind_hl.loc[ind_hl.index.get_level_values('industry_code') == row['industry_code'], 'hl_rtn']
            t_result = stats.ttest_1samp(ind_returns, 0)
            industry_effects.loc[idx, 't_stat'] = t_result.statistic
            industry_effects.loc[idx, 'p_value'] = t_result.pvalue
        else:
            industry_effects.loc[idx, 't_stat'] = np.nan
            industry_effects.loc[idx, 'p_value'] = np.nan
    
    # 創建高減低摘要
    hl_summary = pd.DataFrame({
        'portfolio': ['H-L (1-10)', 'Industry-Adjusted H-L', 'M23-Adjusted H-L'],
        'count': [len(overall_hl), len(ind_hl), len(m23_hl)],
        'mean': [
            overall_hl['hl_rtn'].mean(), 
            ind_hl['hl_rtn'].mean(), 
            m23_hl['hl_rtn'].mean() if not m23_hl.empty else np.nan
        ],
        'std': [
            overall_hl['hl_rtn'].std(), 
            ind_hl['hl_rtn'].std(),
            m23_hl['hl_rtn'].std() if not m23_hl.empty else np.nan
        ],
        'min': [
            overall_hl['hl_rtn'].min(), 
            ind_hl['hl_rtn'].min(),
            m23_hl['hl_rtn'].min() if not m23_hl.empty else np.nan
        ],
        'max': [
            overall_hl['hl_rtn'].max(), 
            ind_hl['hl_rtn'].max(),
            m23_hl['hl_rtn'].max() if not m23_hl.empty else np.nan
        ]
    })
    
    # 計算t統計量
    for idx, row in hl_summary.iterrows():
        if row['count'] > 1:
            if row['portfolio'] == 'H-L (1-10)':
                t_result = stats.ttest_1samp(overall_hl['hl_rtn'], 0)
            elif row['portfolio'] == 'Industry-Adjusted H-L':
                t_result = stats.ttest_1samp(ind_hl['hl_rtn'], 0)
            else:  # M23-Adjusted H-L
                if not m23_hl.empty:
                    t_result = stats.ttest_1samp(m23_hl['hl_rtn'], 0)
                else:
                    t_result = None
            
            if t_result is not None:
                hl_summary.loc[idx, 't_stat'] = t_result.statistic
                hl_summary.loc[idx, 'p_value'] = t_result.pvalue
            else:
                hl_summary.loc[idx, 't_stat'] = np.nan
                hl_summary.loc[idx, 'p_value'] = np.nan
        else:
            hl_summary.loc[idx, 't_stat'] = np.nan
            hl_summary.loc[idx, 'p_value'] = np.nan
    
    # 合併高減低收益數據
    overall_hl.reset_index(inplace=True)
    
    # 產業調整後平均收益（基本三碼產業）
    ind_adj_mean = ind_hl.reset_index().groupby('年月')['hl_rtn'].mean().reset_index()
    ind_adj_mean.rename(columns={'hl_rtn': 'ind_adj_hl_rtn'}, inplace=True)
    
    # M23產業調整後平均收益
    if not m23_hl.empty:
        m23_adj_mean = m23_hl.reset_index().groupby('年月')['hl_rtn'].mean().reset_index()
        m23_adj_mean.rename(columns={'hl_rtn': 'm23_adj_hl_rtn'}, inplace=True)
    else:
        m23_adj_mean = pd.DataFrame(columns=['年月', 'm23_adj_hl_rtn'])
    
    # 合併所有數據
    hl_df = pd.merge(
        overall_hl[['年月', 'hl_rtn']].rename(columns={'hl_rtn': 'overall_hl_rtn'}),
        ind_adj_mean,
        on='年月', how='outer'
    )
    
    hl_df = pd.merge(
        hl_df,
        m23_adj_mean,
        on='年月', how='outer'
    )
    
    return group_stats, hl_summary, hl_df, industry_effects, m23_effects

def save_data(df, path):
    """保存數據到CSV"""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8')
        return True
    except Exception as e:
        print(f"保存數據錯誤: {e}")
        return False

def analyze_m23_industry(df):
    """分析M23產業細分對績效的影響"""
    if df is None or df.empty or 'm23_code' not in df.columns:
        return pd.DataFrame()
    
    # 僅使用有M23代碼的數據
    m23_df = df.dropna(subset=['m23_code', 'rtn']).copy()
    
    if m23_df.empty:
        return pd.DataFrame()
    
    # 計算各M23產業細分的平均收益
    m23_stats = m23_df.groupby('m23_code').agg({
        'rtn': ['count', 'mean', 'std', 'min', 'max'],
        'score': 'mean'
    })
    
    # 展平多級索引
    m23_stats.columns = ['_'.join(col).strip() for col in m23_stats.columns.values]
    m23_stats.reset_index(inplace=True)
    
    # 計算t統計量
    for idx, row in m23_stats.iterrows():
        if row['rtn_count'] > 1:
            m23_returns = m23_df.loc[m23_df['m23_code'] == row['m23_code'], 'rtn']
            t_result = stats.ttest_1samp(m23_returns, 0)
            m23_stats.loc[idx, 't_stat'] = t_result.statistic
            m23_stats.loc[idx, 'p_value'] = t_result.pvalue
        else:
            m23_stats.loc[idx, 't_stat'] = np.nan
            m23_stats.loc[idx, 'p_value'] = np.nan
    
    # 排序結果
    m23_stats_sorted = m23_stats.sort_values('rtn_mean', ascending=False)
    
    return m23_stats_sorted

def run_sort_process(input_file='database/make4.csv', output_file='database/sort4.csv'):
    """執行完整分析流程"""
    
    # 載入數據
    make_df = load_data(input_file)
    if make_df is None:
        return False
    
    # 標準化比率
    std_df = standardize_ratios(make_df)
    if std_df is None:
        return False
    
    # 創建十分位組別
    group_df = create_decile_groups(std_df)
    if group_df is None:
        return False
    
    # 提取產業代碼（包括M23細分）
    group_df = extract_industry_codes(group_df)
    
    # 保存分組數據
    if not save_data(group_df, output_file):
        return False
    
    # 分析收益
    group_stats, hl_summary, hl_df, industry_effects, m23_effects = analyze_returns(group_df)
    
    # 分析M23產業細分
    m23_industry_stats = analyze_m23_industry(group_df)
    
    # 保存分析結果
    output_dir = Path(output_file).parent
    save_data(group_stats, output_dir / 'group_stats4.csv')
    save_data(hl_summary, output_dir / 'hl_summary4.csv')
    
    if hl_df is not None:
        save_data(hl_df, output_dir / 'hl_returns4.csv')
    
    if industry_effects is not None and not industry_effects.empty:
        save_data(industry_effects, output_dir / 'industry_effects4.csv')
    
    if m23_effects is not None and not m23_effects.empty:
        save_data(m23_effects, output_dir / 'm23_effects4.csv')
    
    if not m23_industry_stats.empty:
        save_data(m23_industry_stats, output_dir / 'm23_industry_stats4.csv')
    
    # 簡單因子分析
    correlation = std_df[['profitability', 'growth', 'safety', 'score', 'rtn']].corr()['rtn'].drop('rtn')
    print(f"因子與收益相關性:\n{correlation.round(4)}")
    
    # 打印M23產業細分分析結果
    if not m23_industry_stats.empty:
        print("\nM23產業細分分析 (前5名):")
        print(m23_industry_stats.head()[['m23_code', 'rtn_count', 'rtn_mean', 't_stat', 'p_value']].round(4))
    
    # 對比M23產業調整前後的高減低收益
    if hl_summary is not None and not hl_summary.empty:
        print("\n高減低投資組合收益比較:")
        print(hl_summary[['portfolio', 'count', 'mean', 't_stat', 'p_value']].round(4))
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='排序與分析財務數據 (含M23產業細分)')
    parser.add_argument('--input', default='database/make4.csv', help='輸入CSV路徑')
    parser.add_argument('--output', default='database/sort4.csv', help='輸出CSV路徑')
    
    args = parser.parse_args()
    
    if run_sort_process(args.input, args.output):
        print("排序過程成功完成!")
    else:
        print("排序過程失敗.")