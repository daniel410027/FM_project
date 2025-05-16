#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修改版腳本
----------
本腳本讀取處理過的財務數據並創建以下內容：
1. 基於TEJ產業代碼前三碼的分布圓餅圖
2. 基於TEJ產業代碼前三碼的分布圓餅圖，但針對M23類別使用前四碼進行細分
所有圓餅圖都會被保存到指定目錄。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib as mpl

def load_processed_data(file_path='database/make.csv'):
    """
    載入處理過的CSV財務數據。
    
    Args:
        file_path (str): 處理後CSV文件的路徑
        
    Returns:
        pd.DataFrame: 載入的數據框
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"載入處理後數據形狀: {df.shape}")
        return df
    except Exception as e:
        print(f"載入處理後數據時出錯: {e}")
        return None

def create_industry_pie_chart(df, output_dir='plots', special_treatment=False):
    """
    創建顯示TEJ產業代碼分布的圓餅圖，
    基於每個代碼的前三個字符分組。
    
    Args:
        df (pd.DataFrame): 包含TEJ產業_代碼的輸入數據框
        output_dir (str): 保存輸出圖表的目錄
        special_treatment (bool): 如果為True，則對M23類別使用四碼分類
        
    Returns:
        bool: 成功狀態
    """
    if df is None or df.empty or 'TEJ產業_代碼' not in df.columns:
        print("錯誤: 數據框為空或缺少'TEJ產業_代碼'列")
        return False
    
    # 配置matplotlib以支持中文字符
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取最新日期的數據
    latest_date = df['年月'].max()
    latest_df = df[df['年月'] == latest_date].copy()
    
    if special_treatment:
        # 創建新列，對M23使用前四碼，其他使用前三碼
        latest_df['產業分類'] = latest_df['TEJ產業_代碼'].astype(str).apply(
            lambda x: x[:4] if (len(str(x)) >= 4 and str(x).startswith('M23')) else x[:3] if len(str(x)) >= 3 else x
        )
        filename_suffix = 'special_m23'
        title_suffix = '(M23使用四碼分類)'
    else:
        # 創建新列，統一使用前三碼
        latest_df['產業分類'] = latest_df['TEJ產業_代碼'].astype(str).apply(
            lambda x: x[:3] if len(str(x)) >= 3 else x
        )
        filename_suffix = 'standard'
        title_suffix = '(標準三碼分類)'
    
    # 按產業組計數
    industry_counts = latest_df['產業分類'].value_counts()
    
    # 創建更大尺寸的圖形以提高可讀性
    plt.figure(figsize=(14, 10))
    
    # 創建圓餅圖
    wedges, texts, autotexts = plt.pie(
        industry_counts, 
        labels=industry_counts.index,
        autopct='%1.1f%%',  # 顯示1位小數的百分比
        startangle=90,
        shadow=False,  # 移除陰影效果
        explode=[0.05] * len(industry_counts),  # 略微分離所有切片
        pctdistance=0.6,  # 將百分比標籤移近圓心
        labeldistance=1.1,  # 將標籤移遠一點
    )
    
    # 設置字體屬性以更好地顯示中文字符
    for text in texts:
        text.set_fontsize(11)
    
    # 設置百分比標籤的屬性
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    
    # 使用調整函數來避免標籤重疊
    plt.tight_layout()
    
    # 相等的縱橫比確保餅圖呈現為圓形
    plt.axis('equal')
    
    # 添加標題，並使用適合中文字符的字體
    plt.title(f'TEJ產業代碼分布 {title_suffix}', fontsize=16)
    
    # 如果目錄不存在則創建
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存圖片
    output_path = f"{output_dir}/industry_distribution_pie_{filename_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # 顯示圖表
    plt.tight_layout()
    plt.close()  # 關閉圖表以避免顯示
    
    print(f"圓餅圖已保存到 {output_path}")
    
    # 同時打印分布表
    print(f"\n產業分布統計表 {title_suffix}:")
    industry_table = pd.DataFrame({
        '產業代碼': industry_counts.index,
        '公司數量': industry_counts.values,
        '占比 (%)': (industry_counts.values / industry_counts.sum() * 100).round(2)
    })
    print(industry_table.to_string(index=False))
    
    return industry_table

def run_test(file_path='database/make.csv'):
    """
    運行完整的測試流程。
    
    Args:
        file_path (str): 輸入文件路徑
        
    Returns:
        bool: 成功狀態
    """
    print("開始測試流程")
    
    # 載入處理後的數據
    processed_df = load_processed_data(file_path)
    
    if processed_df is not None:
        # 檢查TEJ產業_代碼列是否存在
        if 'TEJ產業_代碼' not in processed_df.columns:
            print("錯誤: 處理後的數據中未找到'TEJ產業_代碼'列")
            # 打印可用列以便調試
            print("可用列:", processed_df.columns.tolist())
            return False
        
        # 創建兩個版本的產業圓餅圖
        print("\n創建標準版本的產業圓餅圖 (三碼分類)")
        standard_table = create_industry_pie_chart(processed_df, special_treatment=False)
        
        print("\n創建特殊版本的產業圓餅圖 (M23使用四碼分類)")
        special_table = create_industry_pie_chart(processed_df, special_treatment=True)
        
        # 打印附加統計信息
        if standard_table is not None and special_table is not None:
            # 標準分類的獨特產業類別數
            standard_categories = len(standard_table)
            print(f"\n標準三碼分類總共有 {standard_categories} 個不同的產業類別")
            
            # 特殊分類的獨特產業類別數
            special_categories = len(special_table)
            print(f"\n使用M23四碼分類總共有 {special_categories} 個不同的產業類別")
            
            # 分析M23類別的改進
            if 'M23' in standard_table['產業代碼'].values:
                m23_standard = standard_table[standard_table['產業代碼'] == 'M23']['公司數量'].iloc[0]
                print(f"\nM23在標準分類中包含 {m23_standard} 家公司")
                
                # 計算M23在特殊分類中的細分類別
                m23_subcategories = special_table[special_table['產業代碼'].str.startswith('M23')]
                if not m23_subcategories.empty:
                    print(f"M23在特殊分類中細分為 {len(m23_subcategories)} 個子類別:")
                    for _, row in m23_subcategories.iterrows():
                        print(f"- {row['產業代碼']}: {row['公司數量']} 家公司 ({row['占比 (%)']}%)")
            
            return True
        
    return False

if __name__ == "__main__":
    import sys
    
    # 允許通過命令行參數自定義輸入文件路徑
    if len(sys.argv) > 1:
        success = run_test(sys.argv[1])
    else:
        success = run_test()
    
    if success:
        print("測試流程成功完成!")
    else:
        print("測試流程失敗。")