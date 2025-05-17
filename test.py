#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
測試安全指標的反向設定
-----------
此腳本測試安全指標 (bab, lev, o, z, evol) 原始值和反向值與股票報酬率的相關性，
以確定哪些指標應該被反向處理。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path='database/make4.csv'):
    """載入財務資料"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"載入財務資料，形狀：{df.shape}")
        
        # 過濾數據以僅包含 200600 之後的數據
        df_filtered = df[df['年月'] > 200600].copy()
        print(f"過濾後資料形狀（年月 > 200600）：{df_filtered.shape}")
        
        return df_filtered
    except Exception as e:
        print(f"載入財務資料時發生錯誤：{e}")
        return None


def test_safety_indicators_correlation(df):
    """測試安全指標與報酬的相關性，比較原始值和反向值"""
    if df is None or df.empty:
        return None, None
    
    # 安全指標列
    safety_cols = ['bab', 'lev', 'o', 'z', 'evol']
    
    # 檢查是否所有必要的列都存在
    required_columns = safety_cols + ["rtn"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"錯誤：缺少必要的列：{missing_columns}")
        return None, None
    
    # 建立每個安全指標的原始值和反向值
    results = []
    correlation_plots = {}
    
    for col in safety_cols:
        # 計算原始值的相關性
        orig_corr = df[[col, 'rtn']].corr().iloc[0, 1]
        
        # 計算反向值的相關性（乘以-1）
        df[f'{col}_reversed'] = -df[col]
        reversed_corr = df[[f'{col}_reversed', 'rtn']].corr().iloc[0, 1]
        
        # 儲存結果
        results.append({
            'indicator': col,
            'original_correlation': orig_corr,
            'reversed_correlation': reversed_corr,
            'better_if_reversed': abs(reversed_corr) > abs(orig_corr),
            'recommendation': 'Reverse' if abs(reversed_corr) > abs(orig_corr) else 'Keep Original'
        })
        
        # 為每個指標建立散點圖資料
        correlation_plots[col] = {
            'original': (df[col], df['rtn'], orig_corr),
            'reversed': (df[f'{col}_reversed'], df['rtn'], reversed_corr)
        }
    
    # 轉換為DataFrame並返回
    results_df = pd.DataFrame(results)
    return results_df, correlation_plots


def plot_correlations(correlation_plots, output_dir='figures'):
    """繪製相關性散點圖"""
    try:
        # 建立輸出目錄（如果不存在）
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 設定圖形風格
        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 12))
        
        for i, (indicator, data) in enumerate(correlation_plots.items()):
            # 原始值的散點圖
            ax1 = plt.subplot(5, 2, 2*i+1)
            x, y, corr = data['original']
            ax1.scatter(x, y, alpha=0.3)
            ax1.set_title(f'{indicator} vs Return (Original, corr={corr:.4f})')
            ax1.set_xlabel(indicator)
            ax1.set_ylabel('Return')
            
            # 反向值的散點圖
            ax2 = plt.subplot(5, 2, 2*i+2)
            x, y, corr = data['reversed']
            ax2.scatter(x, y, alpha=0.3, color='orange')
            ax2.set_title(f'{indicator} vs Return (Reversed, corr={corr:.4f})')
            ax2.set_xlabel(f'{indicator} (Reversed)')
            ax2.set_ylabel('Return')
        
        plt.tight_layout()
        plt.savefig(output_path / 'safety_indicators_correlation.png', dpi=300)
        plt.close()
        
        print(f"相關性散點圖已儲存至 {output_path / 'safety_indicators_correlation.png'}")
    except Exception as e:
        print(f"繪製相關性散點圖時發生錯誤：{e}")


def calculate_standardized_scores(df):
    """計算原始和反向標準化後的安全分數，並分析與報酬的相關性"""
    if df is None or df.empty:
        return None
    
    # 安全指標列
    safety_cols = ['bab', 'lev', 'o', 'z', 'evol']
    
    # 建立標準化後的指標
    result_df = df.copy()
    
    # 為每個指標建立標準化的列
    for col in safety_cols:
        result_df[f'z_{col}'] = (result_df[col] - result_df[col].mean()) / result_df[col].std()
    
    # 測試不同的反向組合
    combinations = []
    
    # 測試所有可能的反向設定（32種可能性）
    for bab_rev in [False, True]:
        for lev_rev in [False, True]:
            for o_rev in [False, True]:
                for z_rev in [False, True]:
                    for evol_rev in [False, True]:
                        # 建立反向設定
                        reverse_safety = {
                            'bab': bab_rev,
                            'lev': lev_rev,
                            'o': o_rev,
                            'z': z_rev,
                            'evol': evol_rev
                        }
                        
                        # 建立安全分數
                        safety_scores = []
                        for col in safety_cols:
                            if reverse_safety.get(col, False):
                                # 反向指標：用負值
                                safety_scores.append(-result_df[f'z_{col}'])
                            else:
                                safety_scores.append(result_df[f'z_{col}'])
                        
                        # 計算安全分數
                        safety_score = sum(safety_scores)
                        
                        # 計算與報酬的相關性
                        corr = safety_score.corr(result_df['rtn'])
                        
                        # 儲存結果
                        combinations.append({
                            'bab_rev': bab_rev,
                            'lev_rev': lev_rev,
                            'o_rev': o_rev,
                            'z_rev': z_rev,
                            'evol_rev': evol_rev,
                            'correlation': corr
                        })
    
    # 轉換為DataFrame
    combinations_df = pd.DataFrame(combinations)
    
    # 根據相關性排序
    combinations_df = combinations_df.sort_values('correlation', ascending=False)
    
    return combinations_df


def run_tests(input_file='database/make4.csv'):
    """執行所有測試"""
    print(f"開始從 {input_file} 測試安全指標")
    
    # 載入資料
    data_df = load_data(input_file)
    
    if data_df is not None:
        # 測試安全指標與報酬的相關性
        results_df, correlation_plots = test_safety_indicators_correlation(data_df)
        
        if results_df is not None:
            print("\n原始與反向相關性比較:")
            print(results_df.to_string(index=False))
            
            # 繪製相關性散點圖
            plot_correlations(correlation_plots)
            
            # 計算標準化分數的組合
            combinations_df = calculate_standardized_scores(data_df)
            
            if combinations_df is not None:
                print("\n前10組最佳反向設定組合（按相關性排序）:")
                print(combinations_df.head(10).to_string(index=False))
                
                print("\n當前程式使用的反向設定:")
                current_setting = {
                    'bab_rev': False,
                    'lev_rev': True,
                    'o_rev': False, 
                    'z_rev': False,
                    'evol_rev': False
                }
                
                # 尋找當前設定在組合中的排名
                current_rank = combinations_df[
                    (combinations_df['bab_rev'] == current_setting['bab_rev']) &
                    (combinations_df['lev_rev'] == current_setting['lev_rev']) &
                    (combinations_df['o_rev'] == current_setting['o_rev']) &
                    (combinations_df['z_rev'] == current_setting['z_rev']) &
                    (combinations_df['evol_rev'] == current_setting['evol_rev'])
                ]
                
                if not current_rank.empty:
                    rank_index = combinations_df.index.get_loc(current_rank.index[0]) + 1
                    print(f"當前設定的排名: {rank_index}/{len(combinations_df)}")
                    print(f"當前設定的相關性: {current_rank['correlation'].values[0]:.6f}")
                else:
                    print("當前設定不在測試組合中")
                
                return True
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='測試安全指標的反向設定')
    parser.add_argument('--input', default='database/make4.csv', help='輸入CSV檔案路徑')
    
    args = parser.parse_args()
    
    success = run_tests(args.input)
    
    if success:
        print("測試完成！")
    else:
        print("測試失敗。")