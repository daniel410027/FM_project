import pandas as pd
import os


# 處理 fnd.csv 文件，創建 fnd2.csv
def process_fnd_file():
    try:
        # 讀取原始 fnd.csv 文件
        df = pd.read_csv("database/fnd.csv", encoding='utf-16')
        
        # 獲取第一列名稱
        col_name = df.columns[0]
        
        # 將數據按 tab 分割成多列
        split_data = df[col_name].str.split('\t', expand=True)
        
        # 處理證券代碼和公司名稱
        stock_info = split_data[0].str.split(' ', n=1, expand=True)
        
        # 創建新的數據框
        new_df = pd.DataFrame()
        new_df['證券代碼'] = stock_info[0]
        new_df['公司名稱'] = stock_info[1] if 1 in stock_info.columns else ""
        new_df['年月'] = split_data[1].str.strip()
        
        # 添加其他財務相關列
        headers = col_name.split('\t')
        for i in range(2, min(split_data.shape[1], len(headers))):
            new_df[headers[i]] = split_data[i]
        
        # 保存為 fnd2.csv，使用 utf-8 編碼
        output_file = "database/fnd2.csv"
        new_df.to_csv(output_file, encoding='utf-8', index=False)
        print(f"成功處理 fnd.csv 並保存到 {output_file}")
        return True
    except Exception as e:
        print(f"處理 fnd.csv 時發生錯誤: {e}")
        return False

# 處理 price.csv 文件
def process_price_file():
    try:
        # 讀取原始 price.csv 文件
        df = pd.read_csv("database/price.csv", encoding='utf-16')
        
        # 獲取第一列名稱
        col_name = df.columns[0]
        
        # 將數據按 tab 分割成多列
        split_data = df[col_name].str.split('\t', expand=True)
        
        # 處理證券代碼和公司名稱
        stock_info = split_data[0].str.split(' ', n=1, expand=True)
        
        # 創建新的數據框
        new_df = pd.DataFrame()
        new_df['證券代碼'] = stock_info[0]
        new_df['公司名稱'] = stock_info[1] if 1 in stock_info.columns else ""
        new_df['年月'] = split_data[1].str.strip()
        
        # 添加其他價格相關列
        headers = col_name.split('\t')
        for i in range(2, min(split_data.shape[1], len(headers))):
            new_df[headers[i]] = split_data[i]
        
        # 保存為新的 price2.csv，使用 utf-8 編碼
        output_file = "database/price2.csv"
        new_df.to_csv(output_file, encoding='utf-8', index=False)
        print(f"成功處理 price.csv 並保存到 {output_file}")
        return True
    except Exception as e:
        print(f"處理 price.csv 時發生錯誤: {e}")
        return False

# 處理 beta.csv 文件
def process_beta_file():
    try:
        # 讀取原始 beta.csv 文件
        df = pd.read_csv("database/beta.csv", encoding='utf-16')
        
        # 獲取第一列名稱
        col_name = df.columns[0]
        
        # 將數據按 tab 分割成多列
        split_data = df[col_name].str.split('\t', expand=True)
        
        # 處理證券代碼和公司名稱
        stock_info = split_data[0].str.split(' ', n=1, expand=True)
        
        # 創建新的數據框
        new_df = pd.DataFrame()
        new_df['證券代碼'] = stock_info[0]
        new_df['公司名稱'] = stock_info[1] if 1 in stock_info.columns else ""
        
        # 將八位數的年月日轉換為六位數的年月
        date_str = split_data[1].str.strip()
        new_df['年月'] = date_str.str[:6]  # 取前六位數字
        
        # 添加其他beta相關列
        headers = col_name.split('\t')
        for i in range(2, min(split_data.shape[1], len(headers))):
            new_df[headers[i]] = split_data[i]
        
        # 保存為新的 beta2.csv，使用 utf-8 編碼
        output_file = "database/beta2.csv"
        new_df.to_csv(output_file, encoding='utf-8', index=False)
        print(f"成功處理 beta.csv 並保存到 {output_file}")
        return True
    except Exception as e:
        print(f"處理 beta.csv 時發生錯誤: {e}")
        return False

# 執行轉換
fnd_success = process_fnd_file()
price_success = process_price_file()
beta_success = process_beta_file()

if fnd_success and price_success and beta_success:
    print("三個文件都成功轉換為 VSCode 可讀的格式。")
    print("您可以在 'database/' 目錄下找到這些文件。")
else:
    print("轉換過程中出現問題，請檢查錯誤訊息。")