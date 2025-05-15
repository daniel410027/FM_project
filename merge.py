import pandas as pd

# 讀取已處理過的 CSV 文件
df1 = pd.read_csv("database/fnd2.csv", encoding='utf-8')  # 財務數據
df2 = pd.read_csv("database/price2.csv", encoding='utf-8')  # 價格數據
df3 = pd.read_csv("database/beta2.csv", encoding='utf-8')  # beta數據

# 檢查資料框的列名
print("資料框 1 的列名：", df1.columns.tolist())
print("資料框 2 的列名：", df2.columns.tolist())
print("資料框 3 的列名：", df3.columns.tolist())

# 處理 beta2.csv 的年月欄位，只保留前六位
# 先將年月轉換為字串（如果不是的話），然後取前6位
df3['年月'] = df3['年月'].astype(str).str[:6]

# 確保其他資料框的年月欄位也是字串格式，以利後續合併
df1['年月'] = df1['年月'].astype(str)
df2['年月'] = df2['年月'].astype(str)

# 合併前檢查缺失值和重複值
print(f"\n資料框 1 (fnd2.csv) 中'證券代碼'的唯一值數量: {df1['證券代碼'].nunique()}")
print(f"資料框 1 (fnd2.csv) 中'年月'的唯一值數量: {df1['年月'].nunique()}")
print(f"資料框 2 (price2.csv) 中'證券代碼'的唯一值數量: {df2['證券代碼'].nunique()}")
print(f"資料框 2 (price2.csv) 中'年月'的唯一值數量: {df2['年月'].nunique()}")
print(f"資料框 3 (beta2.csv) 中'證券代碼'的唯一值數量: {df3['證券代碼'].nunique()}")
print(f"資料框 3 (beta2.csv) 中'年月'的唯一值數量: {df3['年月'].nunique()}")

# 顯示 beta2.csv 的年月範例（驗證是否正確截取）
print(f"\nbeta2.csv 的年月範例: {df3['年月'].unique()[:5]}")

# 第一步：先合併價格數據和財務數據
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

# 第二步：再合併 beta 數據
merged_df = pd.merge(
    merged_df,
    df3[df3.columns[3:]].join(df3[['證券代碼', '年月']]),  # 只選擇 beta 相關的列，避免再次合併公司名稱
    on=['證券代碼', '年月'],
    how='left',
    suffixes=('', '_CAPM_Beta')  # 修正 suffix 格式
)

# 儲存合併後的資料框
merged_df.to_csv("database/merged_data.csv", encoding='utf-8', index=False)

# 輸出合併相關資訊
print(f"\n原始 df1 形狀 (財務數據): {df1.shape}")
print(f"原始 df2 形狀 (價格數據): {df2.shape}")
print(f"原始 df3 形狀 (beta數據): {df3.shape}")
print(f"合併後資料框形狀: {merged_df.shape}")
print("\n合併後資料框的前 5 行:")
print(merged_df.head())

# 檢查有多少行包含缺失的財務數據
missing_financial_data = merged_df[merged_df['TEJ產業_代碼'].isna()].shape[0]
print(f"\n合併後缺少財務數據的行數: {missing_financial_data}")
print(f"缺少財務數據的百分比: {missing_financial_data / len(merged_df) * 100:.2f}%")

# 檢查有多少行包含缺失的 beta 數據（假設有一個代表性的 beta 欄位）
beta_columns = [col for col in df3.columns if col not in ['證券代碼', '公司名稱', '年月']]
if beta_columns:
    missing_beta_data = merged_df[merged_df[beta_columns[0]].isna()].shape[0]
    print(f"\n合併後缺少 beta 數據的行數: {missing_beta_data}")
    print(f"缺少 beta 數據的百分比: {missing_beta_data / len(merged_df) * 100:.2f}%")