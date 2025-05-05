import pandas as pd

# 讀取已處理過的 CSV 文件
df1 = pd.read_csv("database/fnd2.csv", encoding='utf-8')  # 財務數據
df2 = pd.read_csv("database/price2.csv", encoding='utf-8')  # 價格數據

# 檢查資料框的列名
print("資料框 1 的列名：", df1.columns.tolist())
print("資料框 2 的列名：", df2.columns.tolist())

# 合併前檢查缺失值和重複值
print(f"\n資料框 1 (fnd2.csv) 中'證券代碼'的唯一值數量: {df1['證券代碼'].nunique()}")
print(f"資料框 1 (fnd2.csv) 中'年月'的唯一值數量: {df1['年月'].nunique()}")
print(f"資料框 2 (price2.csv) 中'證券代碼'的唯一值數量: {df2['證券代碼'].nunique()}")
print(f"資料框 2 (price2.csv) 中'年月'的唯一值數量: {df2['年月'].nunique()}")

# 使用左連接進行合併，以保留 price2.csv 中的所有行
merged_df = pd.merge(
    df2,  # 以價格數據為基礎
    df1,  # 將財務數據合併進來
    on=['證券代碼', '年月'],
    how='left',  # 使用左連接保留 df2 的所有行
    suffixes=('_價格', '_財務')
)

# 處理 '公司名稱' 列的衝突 (兩個資料框都有這一欄)
if '公司名稱_價格' in merged_df.columns and '公司名稱_財務' in merged_df.columns:
    # 優先選擇非空的公司名稱
    merged_df['公司名稱'] = merged_df['公司名稱_價格'].fillna(merged_df['公司名稱_財務'])
    # 刪除重複的列
    merged_df.drop(['公司名稱_價格', '公司名稱_財務'], axis=1, inplace=True)

# 儲存合併後的資料框
merged_df.to_csv("database/merged_data.csv", encoding='utf-8', index=False)

# 輸出合併相關資訊
print(f"\n原始 df1 形狀 (財務數據): {df1.shape}")
print(f"原始 df2 形狀 (價格數據): {df2.shape}")
print(f"合併後資料框形狀: {merged_df.shape}")
print("\n合併後資料框的前 5 行:")
print(merged_df.head())

# 檢查有多少行包含缺失的財務數據
missing_financial_data = merged_df[merged_df['TEJ產業_代碼'].isna()].shape[0]
print(f"\n合併後缺少財務數據的行數: {missing_financial_data}")
print(f"缺少財務數據的百分比: {missing_financial_data / len(merged_df) * 100:.2f}%")