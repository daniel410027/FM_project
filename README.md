# FM_project
time horizon : 2000/01/01-2024/12/31 (論文從1957-2016, 考量台股市場成熟性,決定使用24年內時間資料,真實理由只是覺得湊個整數挺不錯的)
target : 上市, 普通股(不含TDR+KY), 400th market value in 2000/01/01
data : 
    adjusted close price, marketvalue, 本益比-TEJ, 股價淨值比-TEJ
criteria : 
    profitability
    growth
    safety


formula & definition

size breakpoint :80th percentile by country

gpoa : gross profit over assets
roe : return on equity
roa : return on assets
cfoa : cash flow over assets
gmar : gross margin
acc : minus accruals (the fraction earnings composed of cash) 

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
