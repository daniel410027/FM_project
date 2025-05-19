import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import platform
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# === 設置中文顯示 ===
# 根據操作系統設置不同的字型
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Heiti TC', 'Apple LiGothic', 'Arial Unicode MS']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'Arial Unicode MS']

plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# === 讀取資料 ===
try:
    df = pd.read_csv("database/sort4.csv")
    print("成功讀取資料")
except Exception as e:
    print(f"讀取資料時發生錯誤: {e}")
    exit(1)

# 資料處理：先篩選掉NaN值，包括 rtn 欄位
df = df.dropna(subset=['group', 'rtn', 'TEJ產業_代碼'])
df['年月'] = df['年月'].astype(int)
df['group'] = df['group'].astype(int)

# 將 TEJ產業_代碼 轉為類別型 (category)，而非字串型，可能更適合後續處理
df['TEJ產業_代碼'] = df['TEJ產業_代碼'].astype('category')

# 檢查資料型態
print(f"資料型態檢查:\n年月: {df['年月'].dtype}\ngroup: {df['group'].dtype}\nTEJ產業_代碼: {df['TEJ產業_代碼'].dtype}\nrtn: {df['rtn'].dtype}")

# === 添加：控制產業固定效果 ===
# 創建一個空的 DataFrame 來存儲調整後的報酬率
industry_adjusted_returns = pd.DataFrame()

# 對每個年月分別進行產業固定效果的調整
for ym in df['年月'].unique():
    # 篩選特定年月的數據
    month_data = df[df['年月'] == ym].copy()
    
    # 創建產業虛擬變數 (One-hot encoding)
    try:
        industry_dummies = pd.get_dummies(month_data['TEJ產業_代碼'], prefix='ind', drop_first=True)
    except Exception as e:
        print(f"產業虛擬變數創建錯誤，年月 {ym}: {e}")
        continue
        
    # 確認虛擬變數是否為空
    if industry_dummies.empty:
        print(f"警告: 年月 {ym} 的產業虛擬變數為空")
        month_data['adj_rtn'] = month_data['rtn']
        industry_adjusted_returns = pd.concat([industry_adjusted_returns, month_data[['證券代碼', '年月', 'group', 'TEJ產業_代碼', 'adj_rtn']]])
        continue
    
    # 合併資料
    reg_data = pd.concat([month_data[['證券代碼', 'rtn', '年月']], industry_dummies], axis=1)
    
    # 檢查並處理NaN值
    if reg_data.isnull().values.any():
        print(f"警告: 年月 {ym} 的回歸資料含有NaN值，進行清理")
        reg_data = reg_data.dropna()
    
    # 如果該月份的資料太少，直接使用原始報酬率
    if len(reg_data) <= len(industry_dummies.columns) + 2:
        print(f"警告: 年月 {ym} 的樣本數不足進行回歸分析")
        month_data['adj_rtn'] = month_data['rtn']
        industry_adjusted_returns = pd.concat([industry_adjusted_returns, month_data[['證券代碼', '年月', 'group', 'TEJ產業_代碼', 'adj_rtn']]])
        continue
    
    try:
        # 構建回歸模型，控制產業固定效果
        # 準備 X (產業虛擬變數) 和 y (報酬率)，明確轉換為浮點數類型
        X = sm.add_constant(industry_dummies.astype(float))
        y = reg_data['rtn'].astype(float)
        
        # 執行回歸
        model = sm.OLS(y, X)
        results = model.fit()
        
        # 計算殘差 (實際報酬 - 產業預期報酬) 作為調整後的報酬
        month_data['adj_rtn'] = y - results.predict(X)
        
        # 添加平均報酬，以保持整體報酬水平
        month_data['adj_rtn'] = month_data['adj_rtn'] + month_data['rtn'].mean()
        
    except Exception as e:
        print(f"年月 {ym} 進行回歸分析時發生錯誤: {e}")
        month_data['adj_rtn'] = month_data['rtn']  # 發生錯誤時使用原始報酬
    
    # 將當月調整後的資料添加到結果中
    industry_adjusted_returns = pd.concat([industry_adjusted_returns, month_data[['證券代碼', '年月', 'group', 'TEJ產業_代碼', 'adj_rtn']]])

print("產業調整完成")

# 重新排序
industry_adjusted_returns = industry_adjusted_returns.sort_values(by=['group', '證券代碼', '年月'])

# === 計算各 group 等權報酬（使用調整後的報酬率）===
adj_group_monthly_return = (
    industry_adjusted_returns.groupby(['group', '年月'])['adj_rtn']
    .mean()
    .reset_index()
    .rename(columns={'adj_rtn': 'group_ret'})
)

# 確保年月正確排序
adj_group_monthly_return = adj_group_monthly_return.sort_values(by=['group', '年月'])

# === 計算 H-L (Group 1 - Group 10) 投資組合 ===
# 先創建 pivot table 來方便計算 H-L
adj_pivot_returns = adj_group_monthly_return.pivot(index='年月', columns='group', values='group_ret')

# 確認 group 1 和 group 10 存在
min_group = adj_pivot_returns.columns.min()
max_group = adj_pivot_returns.columns.max()

print(f"使用 Group {min_group} (最低) 和 Group {max_group} (最高) 計算 H-L 投資組合")

# 計算 H-L (Group 1 - Group 10 或最低與最高組)
adj_pivot_returns['H-L'] = adj_pivot_returns[min_group] - adj_pivot_returns[max_group]

# 重新轉換為長格式
adj_hl_returns = adj_pivot_returns['H-L'].reset_index()
adj_hl_returns['group'] = 'H-L'
adj_hl_returns = adj_hl_returns.rename(columns={'H-L': 'group_ret'})

# 合併回原始數據
adj_group_monthly_return = pd.concat([
    adj_group_monthly_return,
    adj_hl_returns[['年月', 'group', 'group_ret']]
])

# 將年月轉換為日期對象以便於繪圖
try:
    adj_group_monthly_return['date'] = pd.to_datetime(adj_group_monthly_return['年月'].astype(str), format='%Y%m')
except Exception as e:
    print(f"日期轉換錯誤: {e}")
    # 嘗試不同的轉換方式
    adj_group_monthly_return['date'] = pd.to_datetime(adj_group_monthly_return['年月'].astype(str).str.zfill(6), format='%Y%m')

# 初始化累積報酬列
adj_group_monthly_return['cum_return'] = 0.0

# 對每個組分別計算累積回報
for g in adj_group_monthly_return['group'].unique():
    mask = adj_group_monthly_return['group'] == g
    returns = adj_group_monthly_return.loc[mask, 'group_ret']
    # 檢查並處理 returns 中的 NaN 值
    returns = returns.fillna(0)  # 將 NaN 替換為 0，或選擇其他處理方式
    adj_group_monthly_return.loc[mask, 'cum_return'] = (1 + returns).cumprod()

print("累積報酬計算完成")

# === 繪製調整後的累積報酬圖 ===
try:
    plt.figure(figsize=(14, 8))

    # 使用不同線型和顏色
    groups = sorted([g for g in adj_group_monthly_return['group'].unique() if g != 'H-L'])
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
    line_styles = ['-', '--', '-.', ':']
    line_style_cycle = [style for style in line_styles for _ in range(3)]
    if len(line_style_cycle) < len(groups):
        line_style_cycle.extend(line_styles[:len(groups)-len(line_style_cycle)])

    # 先繪製所有分組
    for i, g in enumerate(groups):
        data = adj_group_monthly_return[adj_group_monthly_return['group'] == g]
        plt.plot(data['date'], data['cum_return'], 
                label=f'Group {g}', 
                color=colors[i], 
                linestyle=line_style_cycle[i],
                linewidth=1.5)

    # 特別突出 H-L 投資組合
    hl_data = adj_group_monthly_return[adj_group_monthly_return['group'] == 'H-L']
    plt.plot(hl_data['date'], hl_data['cum_return'], 
            label='H-L (G1-G10)', 
            color='red', 
            linestyle='-',
            linewidth=2.5)

    # 設定日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))  # 每兩年顯示一次
    plt.gcf().autofmt_xdate()  # 自動格式化x軸日期標籤

    # 美化圖表
    plt.title('控制產業固定效果後各 Group 模擬累積報酬（含 H-L 投資組合）', fontsize=16)
    plt.xlabel('年月', fontsize=14)
    plt.ylabel('累積報酬（模擬淨值）', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 儲存並顯示
    plt.savefig("database/industry_adj_group_cum_return_with_hl.png", dpi=300)
    plt.show()
    
    print("繪製總累積報酬圖表完成")
except Exception as e:
    print(f"繪製圖表時發生錯誤: {e}")

# === 額外創建 H-L 報酬率的獨立圖表 ===
try:
    plt.figure(figsize=(14, 6))
    plt.plot(hl_data['date'], hl_data['cum_return'], 
            color='red', 
            linewidth=2)
    plt.title('控制產業固定效果後 H-L (Group 1 - Group 10) 投資組合累積報酬', fontsize=16)
    plt.xlabel('年月', fontsize=14)
    plt.ylabel('累積報酬', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.gcf().autofmt_xdate()

    # 計算 H-L 投資組合的統計數據
    # 先檢查並清理 NaN 值
    hl_monthly_returns = hl_data['group_ret'].values
    hl_monthly_returns = hl_monthly_returns[~np.isnan(hl_monthly_returns)]  # 移除 NaN 值

    # 如果沒有足夠的數據進行計算，設置默認值
    if len(hl_monthly_returns) > 0:
        hl_annualized_return = ((hl_data['cum_return'].iloc[-1]) ** (12 / len(hl_data)) - 1) * 100
        hl_volatility = np.std(hl_monthly_returns) * np.sqrt(12) * 100
        hl_sharpe = hl_annualized_return / hl_volatility if hl_volatility != 0 else 0
        hl_win_rate = np.sum(hl_monthly_returns > 0) / len(hl_monthly_returns) * 100
    else:
        hl_annualized_return = 0
        hl_volatility = 0
        hl_sharpe = 0
        hl_win_rate = 0

    # 在圖表上添加統計數據
    stats_text = f'年化報酬: {hl_annualized_return:.2f}%\n' \
                f'年化波動率: {hl_volatility:.2f}%\n' \
                f'夏普比率: {hl_sharpe:.2f}\n' \
                f'勝率: {hl_win_rate:.2f}%'
                
    plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                fontsize=12)

    plt.tight_layout()
    plt.savefig("database/industry_adj_hl_portfolio_return.png", dpi=300)
    plt.show()
    
    print("繪製 H-L 投資組合圖表完成")
except Exception as e:
    print(f"繪製 H-L 圖表時發生錯誤: {e}")

# === 輸出調整前後的比較報告 ===
# 原始數據重新處理（確保與調整後的數據處理邏輯一致）
df_orig = df.copy()

# 以現有 rtn 欄位為月報酬率，計算各 group 等權報酬
orig_group_monthly_return = (
    df_orig.groupby(['group', '年月'])['rtn']
    .mean()
    .reset_index()
    .rename(columns={'rtn': 'group_ret'})
)

# 確保年月正確排序
orig_group_monthly_return = orig_group_monthly_return.sort_values(by=['group', '年月'])

# 創建 pivot table 來方便計算 H-L
orig_pivot_returns = orig_group_monthly_return.pivot(index='年月', columns='group', values='group_ret')

# 確認 group 1 和 group 10 存在
min_group = orig_pivot_returns.columns.min()
max_group = orig_pivot_returns.columns.max()

# 計算 H-L
orig_pivot_returns['H-L'] = orig_pivot_returns[min_group] - orig_pivot_returns[max_group]

# 轉換為長格式
orig_hl_returns = orig_pivot_returns['H-L'].reset_index()
orig_hl_returns['group'] = 'H-L'
orig_hl_returns = orig_hl_returns.rename(columns={'H-L': 'group_ret'})

# 合併回原始數據
orig_group_monthly_return = pd.concat([
    orig_group_monthly_return,
    orig_hl_returns[['年月', 'group', 'group_ret']]
])

# 將年月轉換為日期對象
try:
    orig_group_monthly_return['date'] = pd.to_datetime(orig_group_monthly_return['年月'].astype(str), format='%Y%m')
except Exception as e:
    print(f"原始數據日期轉換錯誤: {e}")
    # 嘗試不同的轉換方式
    orig_group_monthly_return['date'] = pd.to_datetime(orig_group_monthly_return['年月'].astype(str).str.zfill(6), format='%Y%m')

# 初始化累積報酬列
orig_group_monthly_return['cum_return'] = 0.0

# 對每個組分別計算累積回報
for g in orig_group_monthly_return['group'].unique():
    mask = orig_group_monthly_return['group'] == g
    returns = orig_group_monthly_return.loc[mask, 'group_ret']
    returns = returns.fillna(0)
    orig_group_monthly_return.loc[mask, 'cum_return'] = (1 + returns).cumprod()

# 繪製調整前後 H-L 投資組合比較圖
try:
    plt.figure(figsize=(14, 6))

    # 獲取兩種方法的 H-L 數據
    orig_hl_data = orig_group_monthly_return[orig_group_monthly_return['group'] == 'H-L']
    adj_hl_data = adj_group_monthly_return[adj_group_monthly_return['group'] == 'H-L']

    # 繪製兩條線
    plt.plot(orig_hl_data['date'], orig_hl_data['cum_return'], 
            label='原始 H-L', 
            color='blue', 
            linestyle='-', 
            linewidth=2)

    plt.plot(adj_hl_data['date'], adj_hl_data['cum_return'], 
            label='產業調整後 H-L', 
            color='red', 
            linestyle='--', 
            linewidth=2)

    # 美化圖表
    plt.title('原始 vs 控制產業固定效果後 H-L 投資組合累積報酬比較', fontsize=16)
    plt.xlabel('年月', fontsize=14)
    plt.ylabel('累積報酬', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.gcf().autofmt_xdate()

    # 儲存並顯示
    plt.tight_layout()
    plt.savefig("database/hl_comparison.png", dpi=300)
    plt.show()
    
    print("繪製比較圖表完成")
except Exception as e:
    print(f"繪製比較圖表時發生錯誤: {e}")

# 計算並輸出兩種方法的 H-L 投資組合統計比較
print("\nH-L 投資組合統計比較 (原始 vs 產業調整後):")

# 原始 H-L 統計
orig_hl_returns = orig_hl_data['group_ret'].values
orig_hl_returns = orig_hl_returns[~np.isnan(orig_hl_returns)]

if len(orig_hl_returns) > 0:
    orig_annualized_return = ((orig_hl_data['cum_return'].iloc[-1]) ** (12 / len(orig_hl_data)) - 1) * 100
    orig_volatility = np.std(orig_hl_returns) * np.sqrt(12) * 100
    orig_sharpe = orig_annualized_return / orig_volatility if orig_volatility != 0 else 0
    orig_win_rate = np.sum(orig_hl_returns > 0) / len(orig_hl_returns) * 100
    
    print("\n原始 H-L 投資組合:")
    print(f"年化報酬率: {orig_annualized_return:.4f}%")
    print(f"年化波動率: {orig_volatility:.4f}%")
    print(f"夏普比率: {orig_sharpe:.4f}")
    print(f"勝率: {orig_win_rate:.2f}%")
    print(f"最終累積報酬: {orig_hl_data['cum_return'].iloc[-1]:.4f}")

# 調整後 H-L 統計
adj_hl_returns = adj_hl_data['group_ret'].values
adj_hl_returns = adj_hl_returns[~np.isnan(adj_hl_returns)]

if len(adj_hl_returns) > 0:
    adj_annualized_return = ((adj_hl_data['cum_return'].iloc[-1]) ** (12 / len(adj_hl_data)) - 1) * 100
    adj_volatility = np.std(adj_hl_returns) * np.sqrt(12) * 100
    adj_sharpe = adj_annualized_return / adj_volatility if adj_volatility != 0 else 0
    adj_win_rate = np.sum(adj_hl_returns > 0) / len(adj_hl_returns) * 100
    
    print("\n產業調整後 H-L 投資組合:")
    print(f"年化報酬率: {adj_annualized_return:.4f}%")
    print(f"年化波動率: {adj_volatility:.4f}%")
    print(f"夏普比率: {adj_sharpe:.4f}")
    print(f"勝率: {adj_win_rate:.2f}%")
    print(f"最終累積報酬: {adj_hl_data['cum_return'].iloc[-1]:.4f}")

# 比較改善幅度
print("\n產業調整效果:")
if len(orig_hl_returns) > 0 and len(adj_hl_returns) > 0:
    return_improvement = adj_annualized_return - orig_annualized_return
    sharpe_improvement = adj_sharpe - orig_sharpe
    winrate_improvement = adj_win_rate - orig_win_rate
    
    print(f"年化報酬率改善: {return_improvement:.4f}%")
    print(f"夏普比率改善: {sharpe_improvement:.4f}")
    print(f"勝率改善: {winrate_improvement:.2f}%")
    
    # 檢查產業調整是否有效
    if sharpe_improvement > 0:
        print("\n結論: 產業固定效果控制有效提升策略表現")
    else:
        print("\n結論: 產業固定效果控制未能有效提升策略表現")