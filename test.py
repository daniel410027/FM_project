#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Script
----------
檢查數據檔案欄位名稱並打印相關信息。
"""

import pandas as pd
import numpy as np
import os.path

def check_file_exists(file_path):
    """檢查檔案是否存在"""
    exists = os.path.isfile(file_path)
    print(f"檔案 '{file_path}' 存在: {exists}")
    return exists

def check_columns(file_path):
    """檢查CSV檔案的欄位名稱"""
    try:
        # 嘗試讀取檔案
        df = pd.read_csv(file_path, encoding='utf-8', nrows=5)  # 只讀取前5行提高效率
        
        # 打印基本信息
        print(f"\n檔案: {file_path}")
        print(f"行數 (前5行): {len(df)}")
        print(f"欄位數: {len(df.columns)}")
        
        # 打印所有欄位名稱
        print("\n全部欄位名稱 (包含前後空格):")
        for i, column in enumerate(df.columns):
            print(f"{i+1}. '{column}'")
        
        # 檢查特定欄位是否存在
        required_columns = [
            "證券代碼", "年月", "收盤價(元)_月", "公司名稱", 
            "營業毛利", "資產總額", "ROE(A)－稅後", "現金及約當現金", 
            "營業成本", "來自營運之現金流量", "稅後淨利率"
        ]
        
        print("\n所需欄位狀態:")
        for col in required_columns:
            exists = col in df.columns
            if exists:
                print(f"'{col}' - 存在")
            else:
                # 檢查是否是空格問題
                similar_cols = [c for c in df.columns if c.strip() == col.strip()]
                if similar_cols:
                    print(f"'{col}' - 不存在，但找到類似欄位: {similar_cols}")
                else:
                    print(f"'{col}' - 不存在")
        
        # 檢查數據類型
        print("\n欄位數據類型:")
        for col in df.columns:
            print(f"'{col}' - {df[col].dtype}")
            
        return True
    except Exception as e:
        print(f"錯誤: {e}")
        return False

def main():
    """主函數"""
    # 檢查資料夾是否存在
    print("檢查資料夾...")
    if not os.path.isdir("database"):
        print("警告: 'database' 資料夾不存在")
    
    # 檢查檔案
    files_to_check = [
        "database/merged_data.csv",
        "database/first_backfill.csv"
    ]
    
    for file_path in files_to_check:
        if check_file_exists(file_path):
            check_columns(file_path)
        print("-" * 60)
        
if __name__ == "__main__":
    main()