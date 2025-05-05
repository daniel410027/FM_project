import pandas as pd
import os

# 設定來源與輸出檔案參數
input_filename = "merged_data.csv"
input_path = f"database/{input_filename}"
output_prefix = os.path.splitext(input_filename)[0]  # e.g., 'merged_data'

# 讀取資料
df = pd.read_csv(input_path, encoding='utf-8')

# 計算每個欄位在不同年月的覆蓋率（非缺失值比例）
coverage = df.groupby("年月").agg(lambda x: x.notna().mean())

# 將 index 年月 轉為欄位
coverage.reset_index(inplace=True)

# 組合輸出檔名
output_filename = f"{output_prefix}_coverage_by_ym.csv"
output_path = f"database/{output_filename}"

# 儲存結果
coverage.to_csv(output_path, encoding='utf-8', index=False)

print(f"覆蓋率結果已儲存至: {output_path}")


# 輸出前幾行
print("各年月欄位資料覆蓋率（前五行）:")
print(coverage.head())

# 額外：找出某欄在某些年月覆蓋率過低
threshold = 0.5
low_coverage = coverage.set_index("年月").lt(threshold)
if low_coverage.any().any():
    print("\n以下欄位在某些年月的資料覆蓋率低於 50%：")
    for col in low_coverage.columns:
        low_months = low_coverage.index[low_coverage[col]].tolist()
        if low_months:
            print(f"- {col}: {low_months}")
else:
    print("\n所有欄位的資料覆蓋率皆在 50% 以上。")
