import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import platform

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
df = pd.read_csv("database/sort4.csv")

# 資料處理：先篩選掉NaN值，包括 rtn 欄位
df = df.dropna(subset=['group', 'rtn'])
df['年月'] = df['年月'].astype(int)
df['group'] = df['group'].astype(int)

# 按照正確順序排序
df = df.sort_values(by=['group', '證券代碼', '年月', 'TEJ產業_代碼'])

# 以現有 rtn 欄位為月報酬率，計算各 group 等權報酬
group_monthly_return = (
    df.groupby(['group', '年月'])['rtn']
    .mean()
    .reset_index()
    .rename(columns={'rtn': 'group_ret'})
)

# 確保年月正確排序
group_monthly_return = group_monthly_return.sort_values(by=['group', '年月'])

# === 計算 H-L (Group 1 - Group 10) 投資組合 ===
# 先創建 pivot table 來方便計算 H-L
pivot_returns = group_monthly_return.pivot(index='年月', columns='group', values='group_ret')

# 確認 group 1 和 group 10 存在
min_group = pivot_returns.columns.min()
max_group = pivot_returns.columns.max()

# 計算 H-L (Group 1 - Group 10 或最低與最高組)
pivot_returns['H-L'] = pivot_returns[min_group] - pivot_returns[max_group]

# 重新轉換為長格式
hl_returns = pivot_returns['H-L'].reset_index()
hl_returns['group'] = 'H-L'
hl_returns = hl_returns.rename(columns={'H-L': 'group_ret'})

# 合併回原始數據
group_monthly_return = pd.concat([
    group_monthly_return,
    hl_returns[['年月', 'group', 'group_ret']]
])

# 將年月轉換為日期對象以便於繪圖
group_monthly_return['date'] = pd.to_datetime(group_monthly_return['年月'].astype(str), format='%Y%m')

# 初始化累積報酬列
group_monthly_return['cum_return'] = 0.0

# 對每個組分別計算累積回報
for g in group_monthly_return['group'].unique():
    mask = group_monthly_return['group'] == g
    returns = group_monthly_return.loc[mask, 'group_ret']
    # 檢查並處理 returns 中的 NaN 值
    returns = returns.fillna(0)  # 將 NaN 替換為 0，或選擇其他處理方式
    group_monthly_return.loc[mask, 'cum_return'] = (1 + returns).cumprod()

# === 畫圖 ===
plt.figure(figsize=(14, 8))

# 使用不同線型和顏色
groups = sorted([g for g in group_monthly_return['group'].unique() if g != 'H-L'])
colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
line_styles = ['-', '--', '-.', ':']
line_style_cycle = [style for style in line_styles for _ in range(3)]
if len(line_style_cycle) < len(groups):
    line_style_cycle.extend(line_styles[:len(groups)-len(line_style_cycle)])

# 先繪製所有分組
for i, g in enumerate(groups):
    data = group_monthly_return[group_monthly_return['group'] == g]
    plt.plot(data['date'], data['cum_return'], 
             label=f'Group {g}', 
             color=colors[i], 
             linestyle=line_style_cycle[i],
             linewidth=1.5)

# 特別突出 H-L 投資組合
hl_data = group_monthly_return[group_monthly_return['group'] == 'H-L']
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
plt.title('各 Group 模擬累積報酬（含 H-L 投資組合）', fontsize=16)
plt.xlabel('年月', fontsize=14)
plt.ylabel('累積報酬（模擬淨值）', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 標記重要時間點
important_dates = {}

for date_str, label in important_dates.items():
    date = pd.to_datetime(date_str)
    plt.axvline(x=date, linestyle=':', color='gray', alpha=0.7)
    plt.text(date, plt.ylim()[1]*0.9, label, rotation=90)

# 儲存並顯示
plt.savefig("database/group_cum_return_with_hl.png", dpi=300)
plt.show()

# === 額外創建 H-L 報酬率的獨立圖表 ===
plt.figure(figsize=(14, 6))
plt.plot(hl_data['date'], hl_data['cum_return'], 
         color='red', 
         linewidth=2)
plt.title('H-L (Group 1 - Group 10) 投資組合累積報酬', fontsize=16)
plt.xlabel('年月', fontsize=14)
plt.ylabel('累積報酬', fontsize=14)
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
plt.gcf().autofmt_xdate()

# 標記相同的重要時間點
for date_str, label in important_dates.items():
    date = pd.to_datetime(date_str)
    plt.axvline(x=date, linestyle=':', color='gray', alpha=0.7)
    plt.text(date, plt.ylim()[1]*0.9, label, rotation=90)

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
plt.savefig("database/hl_portfolio_return.png", dpi=300)
plt.show()

# 輸出 H-L 投資組合的月度報酬摘要統計
print("\nH-L 投資組合月度報酬統計:")
if len(hl_monthly_returns) > 0:
    print(f"平均月報酬率: {np.mean(hl_monthly_returns) * 100:.4f}%")
    print(f"最大月報酬率: {np.max(hl_monthly_returns) * 100:.4f}%")
    print(f"最小月報酬率: {np.min(hl_monthly_returns) * 100:.4f}%")
    print(f"勝率: {hl_win_rate:.2f}%")
else:
    print("無足夠有效數據進行統計")