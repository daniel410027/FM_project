#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sort_Diverse Script
-----------------
這個腳本根據市值將公司分為前 50% 和後 50% 兩組，然後針對每一組
分別標準化財務比率、創建綜合得分、排序股票到十分位數，
並使用產業固定效應分析每組的回報。
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path


def load_make_data(file_path='database/make4.csv'):
    """
    載入處理過的財務數據。
    
    Args:
        file_path (str): 處理過的 CSV 檔案路徑
        
    Returns:
        pd.DataFrame: 載入的資料框
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"載入財務數據，資料形狀: {df.shape}")
        
        # 篩選出只包含 200600 之後的期間資料，以確保資料完整性
        df_filtered = df[df['年月'] > 200600].copy()
        print(f"篩選後的資料形狀 (年月 > 200600): {df_filtered.shape}")
        
        return df_filtered
    except Exception as e:
        print(f"載入財務數據時出錯: {e}")
        return None


def split_by_market_cap(df, market_cap_col='市值(百萬元)'):
    """
    根據市值將資料分成兩組：前 50% 和後 50%。
    
    Args:
        df (pd.DataFrame): 輸入的資料框
        market_cap_col (str): 市值欄位名稱
        
    Returns:
        tuple: (前 50% 市值公司資料, 後 50% 市值公司資料)
    """
    if df is None or df.empty or market_cap_col not in df.columns:
        print(f"錯誤: 缺少市值欄位 {market_cap_col}")
        return None, None
    
    result_df = df.copy()
    
    # 確認市值欄位類型為數值型
    result_df[market_cap_col] = pd.to_numeric(result_df[market_cap_col], errors='coerce')
    
    # 初始化市值分組欄位
    result_df['market_cap_group'] = np.nan
    
    # 按期間分組進行市值分組
    print("根據市值按期間分組中...")
    
    for period in result_df['年月'].unique():
        period_mask = result_df['年月'] == period
        period_data = result_df.loc[period_mask].copy()
        
        # 排除市值為 NaN 的行
        valid_period_data = period_data.dropna(subset=[market_cap_col])
        
        if len(valid_period_data) >= 2:  # 確保有足夠的數據進行分組
            try:
                # 計算市值中位數
                median_market_cap = valid_period_data[market_cap_col].median()
                
                # 標記高於中位數的為 1 (前 50%)，低於的為 0 (後 50%)
                for idx in valid_period_data.index:
                    if valid_period_data.loc[idx, market_cap_col] >= median_market_cap:
                        result_df.loc[idx, 'market_cap_group'] = 1  # 前 50%
                    else:
                        result_df.loc[idx, 'market_cap_group'] = 0  # 後 50%
            except Exception as e:
                print(f"期間 {period} 市值分組錯誤: {e}")
        else:
            print(f"警告: 期間 {period} 的有效市值資料少於 2 筆")
    
    # 將市值分組轉換為整數類型
    result_df.loc[result_df['market_cap_group'].notna(), 'market_cap_group'] = \
        result_df.loc[result_df['market_cap_group'].notna(), 'market_cap_group'].astype(int)
    
    # 根據市值分組拆分資料
    high_cap_df = result_df[result_df['market_cap_group'] == 1].copy()
    low_cap_df = result_df[result_df['market_cap_group'] == 0].copy()
    
    print(f"高市值組 (前 50%) 資料數量: {len(high_cap_df)}")
    print(f"低市值組 (後 50%) 資料數量: {len(low_cap_df)}")
    
    return high_cap_df, low_cap_df


def standardize_ratios_by_period(df):
    """
    按期間標準化財務比率及其增量。
    
    Args:
        df (pd.DataFrame): 輸入的資料框，包含財務比率
        
    Returns:
        pd.DataFrame: 包含標準化比率的資料框
    """
    if df is None or df.empty:
        return None
    
    # 創建副本以避免修改原始資料
    result_df = df.copy()
    
    # 需要標準化的收益率比率欄位
    profitability_cols = ['gpoa', 'roe', 'cfoa', 'gmar', 'acc']
    
    # 需要標準化的增長比率欄位
    growth_cols = ['delta of gpoa', 'delta of roe', 'delta of cfoa', 
                   'delta of gmar', 'delta of acc']
    
    # 需要標準化的安全性比率欄位
    safety_cols = ['bab', 'lev', 'o', 'z', 'evol']
    
    # 定義需要反向處理的安全性指標
    reverse_safety = {
        'bab': False,  # -0.0041 負相關，不反向
        'lev': False,   # -0.0005 負相關，需要反向
        'o': False,    # -0.0009 負相關，但反向後更差，不反向
        'z': False,    # -0.0009 負相關，分布問題嚴重，不反向
        'evol': False  # +0.0033 正相關，不反向
    }
    
    # 檢查所有必要的欄位是否存在
    required_columns = ["證券代碼", "年月", "公司名稱", "rtn"] + profitability_cols + growth_cols + safety_cols
    
    # 檢查所有必要的欄位
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        print(f"錯誤: 缺少必要的欄位: {missing_columns}")
        return None
    
    # 創建標準化欄位
    print("按期間標準化財務比率中...")
    
    # 初始化所有標準化欄位
    all_cols = profitability_cols + growth_cols + safety_cols
    for col in all_cols:
        result_df[f'z_{col}'] = np.nan
    
    # 按期間分組標準化
    for period in result_df['年月'].unique():
        period_mask = result_df['年月'] == period
        
        # 標準化所有比率欄位
        for col in all_cols:
            # 獲取當前期間的數值
            values = result_df.loc[period_mask, col]
            
            # 對於增量欄位，適當處理 NaN 值
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
                # 對於非增量欄位
                mean = values.mean()
                std = values.std()
                
                if std > 0:
                    z_scores = (values - mean) / std
                    result_df.loc[period_mask, f'z_{col}'] = z_scores
                else:
                    result_df.loc[period_mask, f'z_{col}'] = 0
    
    # 創建收益率得分
    print("創建收益率得分...")
    result_df['profitability'] = sum(result_df[f'z_{col}'] for col in profitability_cols)
    
    # 創建增長得分
    print("創建增長得分...")
    result_df['growth'] = sum(result_df[f'z_{col}'] for col in growth_cols)
    
    # 創建安全性得分
    print("創建安全性得分...")
    safety_scores = []
    for col in safety_cols:
        if reverse_safety.get(col, False):
            # 反向指標：使用負值
            safety_scores.append(-result_df[f'z_{col}'])
        else:
            safety_scores.append(result_df[f'z_{col}'])
    result_df['safety'] = sum(safety_scores)
    
    # 創建綜合得分（等權重）
    print("創建綜合得分...")
    result_df['score'] = result_df['profitability'] + result_df['growth'] + result_df['safety']
    
    # 處理 inf 和 -inf 值
    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return result_df


def create_decile_groups(df, score_column='score'):
    """
    根據綜合得分按期間創建十分位數分組。
    
    Args:
        df (pd.DataFrame): 包含標準化比率和得分的輸入資料框
        score_column (str): 用於排序的欄位名稱
        
    Returns:
        pd.DataFrame: 包含十分位數分組的資料框
    """
    if df is None or df.empty:
        return None
    
    # 創建副本以避免修改原始資料
    result_df = df.copy()
    
    # 按期間創建十分位數分組
    print(f"按期間根據 {score_column} 創建十分位數分組中...")
    
    # 初始化分組欄位為數值型
    result_df['group'] = np.nan
    
    # 按期間分組分配十分位數
    for period in result_df['年月'].unique():
        period_data = result_df[result_df['年月'] == period].copy()
        
        # 排除得分為 NaN 的行
        valid_period_data = period_data.dropna(subset=[score_column])
        
        if len(valid_period_data) >= 10:  # 確保有足夠的數據進行分成 10 組
            try:
                # 計算十分位數分組，映射到整數 1-10（反轉，使 1 為最高得分）
                valid_period_data['temp_group'] = pd.qcut(
                    valid_period_data[score_column], 
                    10, 
                    labels=False,  # 使用數值標籤
                    duplicates='drop'
                )
                # 反轉順序，使 1 為最高得分（10-group）
                valid_period_data['temp_group'] = 10 - valid_period_data['temp_group']
                
                # 使用計算出的分組更新主資料框
                for idx in valid_period_data.index:
                    result_df.loc[idx, 'group'] = valid_period_data.loc[idx, 'temp_group']
                    
            except ValueError as e:
                print(f"警告: 無法為期間 {period} 創建 10 個分組: {e}")
                # 如果無法創建精確的 10 個分組，嘗試使用較少的組數
                try:
                    # 計算唯一值的數量
                    unique_scores = valid_period_data[score_column].nunique()
                    if unique_scores >= 2:
                        # 使用與唯一值數量相同的組數，最多 10 個
                        n_bins = min(unique_scores, 10)
                        valid_period_data['temp_group'] = pd.qcut(
                            valid_period_data[score_column], 
                            n_bins, 
                            labels=False, 
                            duplicates='drop'
                        )
                        # 反轉順序並重新縮放到 1-10 範圍
                        valid_period_data['temp_group'] = n_bins - valid_period_data['temp_group']
                        # 映射到 1-10 尺度
                        valid_period_data['temp_group'] = valid_period_data['temp_group'].map(
                            lambda x: int(1 + (x - 1) * (10 - 1) / (n_bins - 1)) if n_bins > 1 else 5
                        )
                        for idx in valid_period_data.index:
                            result_df.loc[idx, 'group'] = valid_period_data.loc[idx, 'temp_group']
                    else:
                        # 所有得分都相同，分配到中間組（5）
                        for idx in valid_period_data.index:
                            result_df.loc[idx, 'group'] = 5
                except Exception as inner_e:
                    print(f"為期間 {period} 創建分組時出錯: {inner_e}")
                    # 作為備用方案，分配到中間組
                    for idx in valid_period_data.index:
                        result_df.loc[idx, 'group'] = 5
        else:
            # 數據不足以分成 10 組
            print(f"警告: 期間 {period} 的有效得分資料少於 10 支股票")
            for idx in valid_period_data.index:
                result_df.loc[idx, 'group'] = 5
    
    # 確保分組對於非 NaN 值為整數類型
    result_df.loc[result_df['group'].notna(), 'group'] = result_df.loc[result_df['group'].notna(), 'group'].astype(int)
    
    return result_df


def analyze_returns_by_group(df, group_label):
    """
    使用產業固定效應按十分位數分組分析回報。
    
    Args:
        df (pd.DataFrame): 包含十分位數分組的輸入資料框
        group_label (str): 分組標籤（用於輸出檔案名稱）
        
    Returns:
        tuple: (摘要資料, 差異資料, t 檢定結果, 產業效應資料)
    """
    if df is None or df.empty:
        return None, None, None, None
    
    # 刪除回報或分組為 NaN 的行，並篩選出 年月 > 200600 的資料
    analysis_df = df[(df['年月'] > 200600)].dropna(subset=['rtn', 'group']).copy()
    
    print(f"分析 {group_label} 組按分組的回報（年月 > 200600）...")
    print(f"分析資料形狀: {analysis_df.shape}")
    
    # 提取產業代碼（TEJ產業_代碼的前 3 位）
    if 'TEJ產業_代碼' in analysis_df.columns:
        analysis_df['industry_code'] = analysis_df['TEJ產業_代碼'].astype(str).str[:3]
    else:
        print("警告: 未找到 'TEJ產業_代碼' 欄位。使用預設產業代碼。")
        analysis_df['industry_code'] = '000'  # 預設值
    
    # 按十分位數分組計算回報統計數據
    group_stats = analysis_df.groupby('group')['rtn'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).reset_index()
    
    # 計算均值回報不同於零的 t 統計量
    group_stats['t_stat'] = group_stats.apply(
        lambda x: stats.ttest_1samp(
            analysis_df.loc[analysis_df['group'] == x['group'], 'rtn'], 0
        ).statistic if x['count'] > 1 else np.nan,
        axis=1
    )
    
    # 計算 p 值
    group_stats['p_value'] = group_stats.apply(
        lambda x: stats.ttest_1samp(
            analysis_df.loc[analysis_df['group'] == x['group'], 'rtn'], 0
        ).pvalue if x['count'] > 1 else np.nan,
        axis=1
    )
    
    # 按分組排序
    group_stats = group_stats.sort_values('group')
    
    # 計算高減低（H-L）投資組合回報
    # 分組 1（最高得分）減分組 10（最低得分）
    high_returns = analysis_df.loc[analysis_df['group'] == 1, ['年月', 'rtn', 'industry_code']]
    low_returns = analysis_df.loc[analysis_df['group'] == 10, ['年月', 'rtn', 'industry_code']]
    
    # 初始化產業效應資料框
    industry_effects = pd.DataFrame()
    
    if not high_returns.empty and not low_returns.empty:
        # 計算每個期間的總體高減低回報（不考慮產業匹配）
        high_df = high_returns.copy().rename(columns={'rtn': 'high_rtn'})
        low_df = low_returns.copy().rename(columns={'rtn': 'low_rtn'})
        
        # 按期間合併高低回報
        overall_hl_df = pd.merge(
            high_df.groupby('年月')['high_rtn'].mean().reset_index(),
            low_df.groupby('年月')['low_rtn'].mean().reset_index(),
            on='年月', how='inner'
        )
        
        # 計算總體 H-L 回報
        overall_hl_df['hl_rtn'] = overall_hl_df['high_rtn'] - overall_hl_df['low_rtn']
        
        # 現在計算產業調整後的 H-L 回報
        # 按期間和產業分組的高低回報
        high_by_ind = high_returns.groupby(['年月', 'industry_code'])['rtn'].mean().reset_index()
        high_by_ind = high_by_ind.rename(columns={'rtn': 'high_rtn'})
        
        low_by_ind = low_returns.groupby(['年月', 'industry_code'])['rtn'].mean().reset_index()
        low_by_ind = low_by_ind.rename(columns={'rtn': 'low_rtn'})
        
        # 按期間和產業合併高低回報
        ind_hl_df = pd.merge(high_by_ind, low_by_ind, on=['年月', 'industry_code'], how='inner')
        
        # 按產業計算 H-L 回報
        ind_hl_df['hl_rtn'] = ind_hl_df['high_rtn'] - ind_hl_df['low_rtn']
        
        # 計算產業特定的 H-L 統計數據
        industry_effects = ind_hl_df.groupby('industry_code')['hl_rtn'].agg([
            'count', 'mean', 'std'
        ]).reset_index()
        
        # 計算每個產業的 t 統計量
        industry_effects['t_stat'] = industry_effects.apply(
            lambda x: stats.ttest_1samp(
                ind_hl_df.loc[ind_hl_df['industry_code'] == x['industry_code'], 'hl_rtn'], 0
            ).statistic if x['count'] > 1 else np.nan,
            axis=1
        )
        
        # 計算每個產業的 p 值
        industry_effects['p_value'] = industry_effects.apply(
            lambda x: stats.ttest_1samp(
                ind_hl_df.loc[ind_hl_df['industry_code'] == x['industry_code'], 'hl_rtn'], 0
            ).pvalue if x['count'] > 1 else np.nan,
            axis=1
        )
        
        # 按產業代碼排序
        industry_effects = industry_effects.sort_values('industry_code')
        
        # 計算所有產業的平均值（產業調整後的 H-L）
        ind_adj_hl_mean = ind_hl_df['hl_rtn'].mean()
        ind_adj_hl_std = ind_hl_df['hl_rtn'].std()
        ind_adj_hl_count = len(ind_hl_df)
        ind_adj_hl_min = ind_hl_df['hl_rtn'].min()
        ind_adj_hl_max = ind_hl_df['hl_rtn'].max()
        
        # 計算產業調整後 H-L 回報的 t 統計量
        ind_adj_hl_tstat, ind_adj_hl_pvalue = stats.ttest_1samp(ind_hl_df['hl_rtn'], 0) if ind_adj_hl_count > 1 else (np.nan, np.nan)
        
        # 計算總體（非產業調整）H-L 的統計數據
        overall_hl_count = len(overall_hl_df)
        overall_hl_mean = overall_hl_df['hl_rtn'].mean()
        overall_hl_std = overall_hl_df['hl_rtn'].std()
        overall_hl_min = overall_hl_df['hl_rtn'].min()
        overall_hl_max = overall_hl_df['hl_rtn'].max()
        
        # 計算總體 H-L 回報的 t 統計量
        overall_hl_tstat, overall_hl_pvalue = stats.ttest_1samp(overall_hl_df['hl_rtn'], 0) if overall_hl_count > 1 else (np.nan, np.nan)
        
        # 創建同時包含總體和產業調整結果的 H-L 摘要資料框
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
        
        # 返回 hl_df 作為兩者的組合，用於詳細分析
        hl_df = pd.merge(
            overall_hl_df.rename(columns={'hl_rtn': 'overall_hl_rtn'}),
            ind_hl_df.groupby('年月')['hl_rtn'].mean().reset_index().rename(columns={'hl_rtn': 'ind_adj_hl_rtn'}),
            on='年月', how='outer'
        )
    else:
        # 如果數據不足，創建空摘要
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


def save_sorted_data(df, output_path, group_label):
    """
    將排序後的資料框儲存到新的 CSV 檔案。
    
    Args:
        df (pd.DataFrame): 排序後的資料框
        output_path (str): 輸出目錄路徑
        group_label (str): 分組標籤（用於檔案名稱）
    """
    try:
        # 如果目錄不存在則創建
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / f"sort_diverse_{group_label}.csv"
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"排序後的資料已儲存到 {file_path}")
    except Exception as e:
        print(f"儲存資料時出錯: {e}")


def save_analysis_results(group_stats, hl_summary, hl_df=None, industry_effects=None,
                          output_dir='database', group_label=''):
    """
    將分析結果儲存到 CSV 檔案。
    
    Args:
        group_stats (pd.DataFrame): 分組統計資料框
        hl_summary (pd.DataFrame): H-L 摘要資料框
        hl_df (pd.DataFrame): 按期間的 H-L 回報資料框
        industry_effects (pd.DataFrame): 產業固定效應分析
        output_dir (str): 輸出目錄
        group_label (str): 分組標籤（用於檔案名稱）
    """
    try:
        # 如果目錄不存在則創建
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 儲存分組統計
        group_stats.to_csv(output_path / f"group_stats_diverse_{group_label}.csv", index=False, encoding='utf-8')
        print(f"分組統計已儲存到 {output_path / f'group_stats_diverse_{group_label}.csv'}")
        
        # 儲存 H-L 摘要
        hl_summary.to_csv(output_path / f"hl_summary_diverse_{group_label}.csv", index=False, encoding='utf-8')
        print(f"H-L 摘要已儲存到 {output_path / f'hl_summary_diverse_{group_label}.csv'}")
        
        # 如果有按期間的 H-L 回報則儲存
        if hl_df is not None:
            hl_df.to_csv(output_path / f"hl_returns_diverse_{group_label}.csv", index=False, encoding='utf-8')
            print(f"H-L 回報已儲存到 {output_path / f'hl_returns_diverse_{group_label}.csv'}")
            
        # 如果有產業固定效應分析則儲存
        if industry_effects is not None and not industry_effects.empty:
            industry_effects.to_csv(output_path / f"industry_effects_diverse_{group_label}.csv", index=False, encoding='utf-8')
            print(f"產業固定效應分析已儲存到 {output_path / f'industry_effects_diverse_{group_label}.csv'}")
    except Exception as e:
        print(f"儲存分析結果時出錯: {e}")


def analyze_factor_performance(df, group_label):
    """
    分析因子表現並印出結果。
    
    Args:
        df (pd.DataFrame): 包含標準化因子和分數的資料框
        group_label (str): 分組標籤（用於輸出）
    """
    print(f"\n{group_label} 分組的因子分析:")
    
    # 1. 各因子與回報的相關性
    correlation_df = df[['profitability', 'growth', 'safety', 'score', 'rtn']].corr()
    print("\n與回報的相關性:")
    print(correlation_df['rtn'].drop('rtn').round(4))
    
    # 2. 各期間的平均分數
    period_summary = df.groupby('年月')[['profitability', 'growth', 'safety', 'score']].mean()
    print(f"\n{group_label} 組按期間的平均分數（最後 5 個期間）:")
    print(period_summary.tail().round(4))
    
    # 3. 缺失值統計
    print(f"\n{group_label} 組的缺失值數量:")
    missing_counts = df[['profitability', 'growth', 'safety', 'score', 'rtn']].isnull().sum()
    print(missing_counts)
    
    # 4. 分組回報分析
    print(f"\n{group_label} 組按分數五分位數的平均回報:")
    df_valid = df[df['score'].notna() & df['rtn'].notna()]
    df_valid['score_quintile'] = pd.qcut(df_valid['score'], 5, labels=['Low', 'Q2', 'Q3', 'Q4', 'High'], duplicates='drop')
    quintile_returns = df_valid.groupby('score_quintile', observed=True)['rtn'].mean()
    print(quintile_returns.round(4))
    
    # 5. 各因子五分位數回報分析
    for factor in ['profitability', 'growth', 'safety']:
        print(f"\n{group_label} 組按 {factor} 五分位數的平均回報:")
        factor_df = df[df[factor].notna() & df['rtn'].notna()]
        if len(factor_df) > 5:
            factor_df[f'{factor}_quintile'] = pd.qcut(factor_df[factor], 5, 
                                                     labels=['Low', 'Q2', 'Q3', 'Q4', 'High'], 
                                                     duplicates='drop')
            factor_quintile_returns = factor_df.groupby(f'{factor}_quintile', observed=True)['rtn'].mean()
            print(factor_quintile_returns.round(4))
        else:
            print(f"數據不足以分析 {factor} 的五分位數")


def run_sort_diverse_process(input_file='database/make4.csv', output_dir='database'):
    """
    運行完整的多樣化排序過程，從載入到儲存。
    
    Args:
        input_file (str): 輸入 CSV 檔案路徑
        output_dir (str): 輸出目錄路徑
        
    Returns:
        bool: 成功狀態
    """
    print(f"從 {input_file} 開始排序多樣化流程")
    
    # 載入資料
    make_df = load_make_data(input_file)
    
    if make_df is None:
        print("無法載入資料，流程中止")
        return False
    
    # 檢查是否存在市值欄位
    if '市值(百萬元)' not in make_df.columns:
        print("錯誤: 缺少市值欄位（市值(百萬元)）")
        print(f"可用欄位: {make_df.columns.tolist()}")
        return False
    
    # 根據市值拆分資料為兩組
    high_cap_df, low_cap_df = split_by_market_cap(make_df, market_cap_col='市值(百萬元)')
    
    if high_cap_df is None or low_cap_df is None:
        print("市值分組失敗，流程中止")
        return False
    
    # 處理高市值組
    print("\n處理高市值組（前 50%）...")
    high_cap_std_df = standardize_ratios_by_period(high_cap_df)
    
    if high_cap_std_df is not None:
        high_cap_group_df = create_decile_groups(high_cap_std_df, score_column='score')
        
        if high_cap_group_df is not None:
            # 儲存高市值組排序資料
            save_sorted_data(high_cap_group_df, output_dir, "high_cap")
            
            # 分析高市值組按分組的回報
            high_group_stats, high_hl_summary, high_hl_df, high_industry_effects = analyze_returns_by_group(
                high_cap_group_df, "high_cap")
            
            # 儲存高市值組分析結果
            save_analysis_results(high_group_stats, high_hl_summary, high_hl_df, high_industry_effects,
                               output_dir, "high_cap")
            
            # 分析高市值組的因子表現
            analyze_factor_performance(high_cap_std_df, "高市值")
    
    # 處理低市值組
    print("\n處理低市值組（後 50%）...")
    low_cap_std_df = standardize_ratios_by_period(low_cap_df)
    
    if low_cap_std_df is not None:
        low_cap_group_df = create_decile_groups(low_cap_std_df, score_column='score')
        
        if low_cap_group_df is not None:
            # 儲存低市值組排序資料
            save_sorted_data(low_cap_group_df, output_dir, "low_cap")
            
            # 分析低市值組按分組的回報
            low_group_stats, low_hl_summary, low_hl_df, low_industry_effects = analyze_returns_by_group(
                low_cap_group_df, "low_cap")
            
            # 儲存低市值組分析結果
            save_analysis_results(low_group_stats, low_hl_summary, low_hl_df, low_industry_effects,
                               output_dir, "low_cap")
            
            # 分析低市值組的因子表現
            analyze_factor_performance(low_cap_std_df, "低市值")
    
    # 高市值與低市值組的比較分析
    if high_cap_std_df is not None and low_cap_std_df is not None:
        print("\n高市值與低市值組的比較分析:")
        
        # 1. 因子相關性比較
        high_corr = high_cap_std_df[['profitability', 'growth', 'safety', 'score', 'rtn']].corr()['rtn'].drop('rtn')
        low_corr = low_cap_std_df[['profitability', 'growth', 'safety', 'score', 'rtn']].corr()['rtn'].drop('rtn')
        
        corr_comparison = pd.DataFrame({
            'factor': high_corr.index,
            'high_cap_corr': high_corr.values,
            'low_cap_corr': low_corr.values,
            'difference': high_corr.values - low_corr.values
        })
        
        print("\n因子與回報相關性比較:")
        print(corr_comparison.round(4))
        
        # 2. 平均回報比較
        if 'high_group_stats' in locals() and 'low_group_stats' in locals():
            high_returns = high_group_stats.set_index('group')['mean']
            low_returns = low_group_stats.set_index('group')['mean']
            
            merged_returns = pd.DataFrame({
                'high_cap_return': high_returns,
                'low_cap_return': low_returns,
                'difference': high_returns - low_returns
            }).reset_index()
            
            print("\n按分組的平均回報比較:")
            print(merged_returns.round(4))
            
            # 3. H-L 投資組合比較
            if 'high_hl_summary' in locals() and 'low_hl_summary' in locals():
                high_hl = high_hl_summary.set_index('portfolio')
                low_hl = low_hl_summary.set_index('portfolio')
                
                print("\nH-L 投資組合比較:")
                print(f"高市值組: {high_hl.loc['H-L (1-10)', 'mean']:.4f} (t={high_hl.loc['H-L (1-10)', 't_stat']:.2f}, p={high_hl.loc['H-L (1-10)', 'p_value']:.4f})")
                print(f"低市值組: {low_hl.loc['H-L (1-10)', 'mean']:.4f} (t={low_hl.loc['H-L (1-10)', 't_stat']:.2f}, p={low_hl.loc['H-L (1-10)', 'p_value']:.4f})")
                print(f"差異: {high_hl.loc['H-L (1-10)', 'mean'] - low_hl.loc['H-L (1-10)', 'mean']:.4f}")
        
        # 儲存比較分析結果
        comparison_file = Path(output_dir) / "market_cap_comparison.csv"
        corr_comparison.to_csv(comparison_file, index=False, encoding='utf-8')
        print(f"\n市值比較分析已儲存到 {comparison_file}")
    
    print("\n多樣化排序過程完成！")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='根據市值將公司分為前 50% 和後 50% 兩組，分別分析財務因子')
    parser.add_argument('--input', default='database/make4.csv', help='輸入 CSV 檔案路徑')
    parser.add_argument('--output', default='database', help='輸出目錄路徑')
    
    args = parser.parse_args()
    
    success = run_sort_diverse_process(args.input, args.output)
    
    if success:
        print("多樣化排序過程成功完成！")
    else:
        print("多樣化排序過程失敗。")