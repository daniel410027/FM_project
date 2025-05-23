# FM_project

time horizon : 2000/01/01-2024/12/31 (論文從1957-2016, 考量台股市場成熟性,決定使用24年內時間資料,真實理由只是覺得湊個整數挺不錯的)
target : 上市, 普通股(不含TDR+KY), 400th market value in 2000/01/01-2024/12/31(從1995/01/01開始撈資料避免資料截斷問題), 要記得撈財報發布日期
IFRS data :
	現金及約當現金
	資產總額
	負債總額
	股東權益總額
	營業毛利
	ROA(A)稅後息前
	ROE(A)-稅後
	每股盈餘
	來自營運之現金流量
	營運槓桿度
	財務槓桿度
	稅後淨利率
	營業成本

股價 data :
	收盤價(元)_月
	市值(百萬元)
	本益比-TEJ
	股價淨值比criteria :
    profitability
    growth
    safety

股價報酬(日)-beta值(一年)

formula & definition

size breakpoint :80th percentile by country

gpoa : gross profit over assets
roe : return on equity
roa : return on assets
cfoa : cash flow over assets
gmar : gross margin
acc : minus accruals (the fraction earnings composed of cash)

![1746350875729](image/README/1746350875729.png)

delta : five-year growth
bab : companies with low beta(betting against beta)
lev : companies with low leverage
o : low bankruptcy risk
z : low bankruptcy risk
evol : low roe volatility

profitability = z(Zgpoa + Zroe + Zcfoa + Zgmar + Zacc) (formula 2)
growth = z(Zdelta of gpoa + Zdelta of roe + Zdelta of cfoa + Zdelta of gmar + Zdelta of acc) (formula 3)
safety = z(Zbab + Zlev + Zo + Zz + Zevol) (formula 4)
quality = z(profitability + growth + safety) (formula 5)

assigned stock in each country to ten quality-sorted portfolios, value-weighted, refreshed every calender month, rebalance every calender month to maintain value weights

QMJ portfolios = 0.5(small quality + big quality) - 0.5(small junk + big junk) (formula 7)

control variable : industry, country, firm-fixed effect(???)

p(i, t) = a + b * quality + controls + epsilon, p(i, t) = log(MB), MB = current market equity in JUNE / book equity
rt = alpha + beta(for mkt) * MKT + beta(for smb) * SMB + beta(for umd) * UMD + epsilon (formula 10)
後半段結果我懶得看

程式流程
encoding : 更改編碼方法，使vscode就能打開預覽器，將fnd與price做成fnd2跟price2
merge : 資料合併，對齊，將fnd2與price2做成merge
first_backfill : 填補缺失值，由於我就爛，所以填法有待改進，先對文字類資料groupby公司名稱往前往後回填，對數字類資料groupby公司名稱線性補值，再groupby產業代號groupmean回填，存出first_backfill
接下來，寫新的make.py，利用database/first_backfill.csv製作出database/make.csv
裡面只需要以下欄位 : ["證券代碼", "年月", "收盤價(元)* 月", "公司名稱", "gpoa", "roe", "cfoa", "gmar", "acc", "rtn"], gpoa為"營業毛利" / "資產總額", roe為"ROE(A)-稅後", cfoa為"現金及約當現金" / "資產總額", gmar為"營業毛利" / ("營業毛利" + "營業成本"), acc為"來自營運之現金流量" / ("稅後淨利率" * ("營業毛利" + "營業成本"))
接下來，寫sort.py，對不同的"年月"的["gpoa", "roe", "cfoa", "gmar", "acc"]標準化，製作新的欄位"score"為那五個分數加總，然後對不同"年月"依照分數分成10組，之後要看這十組的rtn的summary，還有差別，還有H-L跟t-statistic
存出 : sort.csv, group_stats.csv, hl_summary.csv, hl_returns.csv，不過跟論文不同的是我的第一組是最高分的

![1746514527462](image/README/1746514527462.png)

控制產業因素(note : 非子產業)後結果
![1746516468796](image/README/1746516468796.png)

程式支線任務
coverage_analyze : merge之後，分析資料覆蓋率，存出merged_data_coverage_by_ym
industry_analyze : 分析make資料產業分類
總共有 21 個不同的產業類別 (前三碼分類)
各產業公司數量 (由高至低):
M23: 417 家公司
M15: 68 家公司
M17: 65 家公司
M25: 62 家公司
M99: 46 家公司
M14: 44 家公司
M28: 34 家公司
M20: 34 家公司
M26: 24 家公司
M13: 23 家公司
M12: 21 家公司
M27: 17 家公司
M22: 15 家公司
M29: 13 家公司
M21: 11 家公司
M11: 10 家公司
M16: 9 家公司
M19: 7 家公司
M18: 5 家公司
M30: 3 家公司
M31: 1 家公司

5/14更新
利用database/first_backfill.csv製作出database/make2.csv 裡面只需要以下欄位 : ["證券代碼", "年月", ""收盤價(元) 月", "公司名稱", "gpoa", "roe", "cfoa", "gmar", "acc", "delta of gpoa", "delta of roe", "delta of cfoa", "delta of gmar", "rtn"], gpoa為"營業毛利" / "資產總額", roe為"ROE(A)-稅後", cfoa為"現金及約當現金" / "資產總額", gmar為"營業毛利" / ("營業毛利" + "營業成本"), acc為"營業毛利"-"來自營運之現金流量"，delta of x為那隻證券代碼的x與延遲六十筆資料之前的x相比的成長率
接下來，寫sort2.py，對不同的"年月"的["gpoa", "roe", "cfoa", "gmar", "acc"]標準化，製作新的欄位"profitability"為那五個分數加總，對不同"年月"的['delta of gpoa', 'delta of roe', 'delta of cfoa', 'delta of gmar', 'delta of acc']標準化，製作新的欄位"growth"為那五個分數加總，然後對不同"年月"依照分數分成10組，之後要看這十組的rtn的summary，還有差別，還有H-L跟t-statistic
因為資料有delta，所以統計時只要"年月">200000的資料就好(年月格式為yyyymm的六位數字)

5/15更新
新增bab資料，需要重撈資料讚啦
程式流程
encoding : 更改編碼方法，使vscode就能打開預覽器，將fnd與price與beta做成fnd2跟price2跟beta2(注意，beta原始是年月日，這裡也改成年月)
merge : 資料合併，對齊，將fnd2與price2做成merge
first_backfill : 填補缺失值，由於我就爛，所以填法有待改進，先對文字類資料groupby公司名稱往前往後回填，對數字類資料groupby公司名稱線性補值，再groupby產業代號groupmean回填，存出first_backfill

利用database/first_backfill.csv製作出database/make3.csv 裡面只需要以下欄位 : ["證券代碼", "年月", ""收盤價(元) 月", "公司名稱", "gpoa", "roe", "cfoa", "gmar", "acc", "delta of gpoa", "delta of roe", "delta of cfoa", "delta of gmar", "bab", "lev", "o", "z", "evol", "rtn"], gpoa為"營業毛利" / "資產總額", roe為"ROE(A)-稅後", cfoa為"現金及約當現金" / "資產總額", gmar為"營業毛利" / ("營業毛利" + "營業成本"), acc為"營業毛利"-"來自營運之現金流量"，delta of x為那隻證券代碼的x與延遲六十筆資料之前的x相比的成長率，bab為"CAPM_Beta 一年" * -1，lev為"股東權益總額" / "資產總額"，o為-1 * (6.03 * "負債總額" / "資產總額" - 1.43 * "來自營運之現金流量" / "資產總額 + 0.076 * "CL" / "CA" - 2.37 * "稅後淨利率" - 1.83 * "來自營運之現金流量" / "負債總額")，z為1.2 * "來自營運之現金流量" / "資產總額" + 3.3 * "稅後淨利率" + 0.6 * "市值(百萬元)" / "負債總額" + 1 * "營業毛利" / "資產總額"，evol是那隻證券代碼的過去60筆資料的"ROE(A)－稅後"標準差

note O分數原始超難做，我是砍掉那些dummy variable跟log(TA/GNP)，實際理由是好麻煩我不想重做資料，比較想要玩其他酷酷的東西，準備給昭文的說法是我們的模型只有台灣股市，所以log(TA/GNP)這項拔掉，然後之後打算做機器學習的東西，那些dummy有可能影響結果，所以砍掉
我實際使用的公式只有O_Score=6.03⋅TLTA−1.43⋅WCTA−2.37⋅NITA−1.83⋅FUTL
好煩之後要記得重撈CL/CA/保留盈餘

![download](image/README/download.jpgdownload.jpg)
safety = z(Zbab + Zlev + Zo + Zz + Zevol) (formula 4)

接下來，寫sort2.py，對不同的"年月"的["gpoa", "roe", "cfoa", "gmar", "acc"]標準化，製作新的欄位"profitability"為那五個分數加總，對不同"年月"的['delta of gpoa', 'delta of roe', 'delta of cfoa', 'delta of gmar', 'delta of acc']標準化，製作新的欄位"growth"為那五個分數加總，對不同"年月"的["bab", "lev", "o", "z", "evol"]標準化，製作新的欄位"safety"為那五個分數的加總然後對不同"年月"依照分數分成10組，之後要看這十組的rtn的summary，還有差別，還有H-L跟t-statistic
因為資料有delta，所以統計時只要"年月">200000的資料就好(年月格式為yyyymm的六位數字)
存出 : sort.csv, group_stats.csv, hl_summary.csv, hl_returns.csv，不過跟論文不同的是我的第一組是最高分的

總結主要修改

修正 Safety 指標方向：反向 o 和 evol
改善資料過濾：從 200000 改為 200600
調整因子權重：根據相關性給予不同權重
處理極端值：使用 Winsorize 方法
修正日期轉換錯誤

5/16
支線任務 : 製作公司分布圓餅圖

5/17
重撈fnd資料，避免前面的不可重作，撈取結果存為fnd3，後面修改encoding後，其他路徑不用改變
利用database/first_backfill.csv製作出database/make4.csv 裡面只需要以下欄位 : ["證券代碼", "年月", ""收盤價(元) 月", "公司名稱", "gpoa", "roe", "cfoa", "gmar", "acc", "delta of gpoa", "delta of roe", "delta of cfoa", "delta of gmar", "bab", "lev", "o", "z", "evol", "rtn"], gpoa為"營業毛利" / "資產總額", roe為"ROE(A)-稅後", cfoa為"現金及 "資產總額", gmar為"營業毛利" / ("營業毛利" + "營業成本"), acc為"營業毛利"-"來自營運之現金流量"，delta of x為那隻證券代碼的x與延遲六十筆資料之前的x相比的成長率，bab為"CAPM_Beta 一年" * -1，lev為"股東權益總額" / "資產總額"，o為-1 * (6.03 * "負債總額" / "資產總額" - 1.43 * "來自營運之現金流量" / "資產總額 + 0.076 * "流動資產" / "流動負債" - 2.37 * "稅後淨利率" - 1.83 * "來自營運之現金流量" / "負債總額")，z為1.2 * "來自營運之現金流量" / "資產總額" + 3.3 * "稅後淨利率" + 0.6 * "市值(百萬元)" / "負債總額" + 1 * "營業毛利" / "資產總額"，evol是那隻證券代碼的過去60筆資料的"ROE(A)－稅後"標準差

考慮在M23產業中使用四碼分類
reverse_safety = { 'bab': False, *# -0.0041 負相關，不反向* 'lev': True, *# -0.0005 負相關，需要反向* 'o'False, *# -0.0009 負相關，但反向後更差，不反向* 'z': False, *# -0.0009 負相關，分布問題嚴重，不反向* 'evol': False *# +0.0033 正相關，不反向* }要重新檢查看看
還要將公司依照市值區分為前50%以及後50%，個別分析前面的分析的效果

5/18
生重新看前面的屎山code重新理清架構
支線任務 : diverse_graph.py : 生成五爪圖
