# Daily strategies for a sterling ISA (2017-01-31 to 2026-11-19)

## The funds checked (WisdomTree and Leverage Shares, London-listed, in pence)

WisdomTree's 3x index funds trade every day since 2012. Leverage Shares' 3x sector funds list in London from December 2021, but Yahoo has no prices for them from December 2022 to April 2024, so they enter in April 2024. Signals: the US ETF at the US close for US markets (QQQ, SPY, SMH, GDX), the fund's own London prices for the FTSE 100, DAX and EURO STOXX 50; every trade at the next London open.

| Fund | Tracks | Issuer | Used from | Median GBP traded a day (last year) | Split / consolidation fixed | Untraded days repriced (backtest years) |
|---|---|---|---|---|---|---|
| LQQ3 | Nasdaq-100 3x | WisdomTree | before 2017 | 8,296,679 | - | 107 (2017: 17, 2018: 89, 2020: 1) |
| 3LUS | S&P 500 3x | WisdomTree | before 2017 | 561,335 | - | 121 (2017: 24, 2018: 94, 2020: 3) |
| 3UKL | FTSE 100 3x | WisdomTree | before 2017 | 631,601 | 2026-05-11 x0.05 | 1 (2018: 1) |
| 3LDE | DAX 3x | WisdomTree | before 2017 | 54,192 | 2026-05-11 x0.05 | 32 (2017: 6, 2018: 19, 2019: 2, 2020: 3, 2024: 1, 2025: 1) |
| 3LEU | EURO STOXX 50 3x | WisdomTree | before 2017 | 20,414 | 2026-05-11 x0.04 | 176 (2017: 32, 2018: 123, 2019: 4, 2020: 15, 2025: 2) |
| 3SMH | Semiconductors (SMH) 3x | Leverage Shares | 2024-04-16 | 100,639 | - | 1 (2025: 1) |
| 3GDX | Gold miners (GDX) 3x | Leverage Shares | 2024-04-16 | 307,535 | - | 1 (2024: 1) |
| 3UKS | FTSE 100 -3x | WisdomTree | before 2017 | 1,325,681 | - | 1 (2018: 1) |

Left out: 3XLF: barely trades (about GBP 300 a day; no price range on 26% of days); 3XLE: barely trades (about GBP 8,000 a day; no price range on 17% of days); 3ULS: S&P 500 -3x: its December 2022 consolidation ratio cannot be pinned down from the prices; 3LSI / 3LOI: silver and oil: one day of pence prices, or broken history from late 2020.

## Answer

Benchmark: the S&P 500 in pounds, 13.6% a year (16.3% in 2017-21, 10.9% in 2022-26). 390 settings on these funds; 1 beat it by 15% or more in both 2017-21 and 2022-26 with at least 60% winning trades.

**Recommended: US indices (2), one slot each: IBS < 0.1; sell close > 5-day avg.** Half the ISA in LQQ3 (Nasdaq-100 3x) and half in 3LUS (S&P 500 3x): buy a fund at the London open after its index (QQQ for LQQ3, SPY for 3LUS) closes in the bottom 10% of its day's range, sell at the London open after the index closes above its 5-day average, cash in between. It was picked after seeing the results, but 5 of the 6 IBS < 0.1 settings on these two funds beat the S&P in both halves and the rest were level with it (within 0.5%). The European index funds and the short funds made results worse; Leverage Shares' semiconductor and gold miner funds (prices only from April 2024) did not improve it.

Rows: the recommendation; the rule fixed in advance from the US nine-fund test (backtest/DAILY.md), before any London price was seen (the honest out-of-sample test of that idea); the setting with the best 2017-21 return among those that won at least 60% of their 2017-21 trades; then the settings that beat the S&P by 15% or more in both halves:

| Setting | CAGR | vs S&P 500 | 2017-21 vs S&P | 2022-26 vs S&P | Win rate | Trades per year | Avg hold (sessions) | Max drawdown | Worst year |
|---|---|---|---|---|---|---|---|---|---|
| US indices (2), one slot each: IBS < 0.1; sell close > 5-day avg | 30.8% | +17.2% | +22.0% | +12.8% | 70.0% | 37 | 3 | -32.9% | -4.6% |
| indices + semis and gold miners (7), one slot each: IBS < 0.1; sell close > prior high | 12.0% | -1.6% | -7.7% | +4.7% | 60.9% | 132 | 3 | -34.7% | -14.5% |
| US indices (2), one slot each: IBS < 0.2; sell RSI(2) > 70 | 27.9% | +14.4% | +30.3% | +0.7% | 67.8% | 52 | 4 | -49.1% | -26.2% |
| US indices (2), one slot each: IBS < 0.1; sell RSI(2) > 70 | 34.0% | +20.5% | +26.0% | +15.3% | 69.6% | 34 | 4 | -42.6% | -17.9% |

By fund set and number of places:

| Funds and places | Settings | Median CAGR | Median vs S&P | Median 2017-21 vs S&P | Median 2022-26 vs S&P | Median win rate | Median max drawdown |
|---|---|---|---|---|---|---|---|
| US indices (2), one slot each | 30 | 21.0% | +7.5% | +5.0% | +4.7% | 69.9% | -41.7% |
| US indices + semis and gold miners (4), one slot each | 30 | 18.2% | +4.7% | -5.3% | +14.0% | 69.3% | -27.4% |
| US indices + semis and gold miners (4), up to 2 at a time | 30 | 23.7% | +10.2% | +3.0% | +14.2% | 68.5% | -46.4% |
| US indices + semis and gold miners (4), up to 3 at a time | 30 | 20.8% | +7.3% | -1.0% | +13.7% | 68.9% | -36.9% |
| WisdomTree indices (5), one slot each | 30 | 8.7% | -4.8% | -7.3% | -1.5% | 63.9% | -43.9% |
| WisdomTree indices (5), up to 2 at a time | 30 | 10.3% | -3.3% | -2.7% | -0.9% | 63.7% | -62.1% |
| WisdomTree indices (5), up to 3 at a time | 30 | 11.4% | -2.2% | -4.5% | -1.3% | 63.6% | -59.0% |
| indices + semis and gold miners (7), one slot each | 30 | 10.6% | -2.9% | -9.4% | +5.1% | 64.6% | -32.3% |
| indices + semis and gold miners (7), up to 2 at a time | 30 | 11.4% | -2.2% | -1.8% | +1.9% | 63.9% | -62.8% |
| indices + semis and gold miners (7), up to 3 at a time | 30 | 13.6% | +0.0% | -2.1% | +3.9% | 64.1% | -58.1% |
| 7 longs + FTSE 100 short (8), one slot each | 30 | 8.7% | -4.9% | -10.4% | +1.8% | 63.2% | -31.5% |
| 7 longs + FTSE 100 short (8), up to 2 at a time | 30 | 5.6% | -7.9% | -6.8% | -8.8% | 62.2% | -62.9% |
| 7 longs + FTSE 100 short (8), up to 3 at a time | 30 | 9.4% | -4.2% | -4.6% | -1.9% | 62.9% | -56.4% |

Adding the FTSE 100 3x short fund (bought after short rises while the FTSE is below its 200-day average) changed the yearly return by -3.2% (median over 90 settings; 0% improved) and the max drawdown by -0.5%.

Calendar years:

| Year | Recommended | Rule fixed in advance (7 funds) | S&P 500 in pounds |
|---|---|---|---|
| 2017 | +10.8% | +11.1% | +11.5% |
| 2018 | -4.6% | -14.5% | +0.4% |
| 2019 | +35.8% | +12.0% | +25.8% |
| 2020 | +117.8% | +20.7% | +14.7% |
| 2021 | +57.2% | +16.8% | +30.1% |
| 2022 | +13.6% | -4.7% | -8.6% |
| 2023 | +28.7% | +11.0% | +19.3% |
| 2024 | +54.7% | +14.1% | +27.2% |
| 2025 | +3.1% | +37.2% | +9.4% |
| 2026 | +20.7% | +22.1% | +9.0% |

Where the recommendation made its money (share of summed trade returns, trades): LQQ3 58% (185), 3LUS 42% (178).
Where the rule fixed in advance made its money (share of summed trade returns, trades): LQQ3 36% (171), 3LUS 25% (170), 3GDX 19% (53), 3SMH 12% (42), 3UKL 8% (266), 3LDE 3% (339), 3LEU -4% (249).

Worst trades of the recommendation: 3LUS 2022-06-10 to 2022-06-22 -21.7%, LQQ3 2022-06-10 to 2022-06-22 -20.7%, LQQ3 2018-12-17 to 2018-12-27 -19.6%, 3LUS 2018-01-30 to 2018-02-13 -19.5%, LQQ3 2025-02-24 to 2025-03-06 -18.6%.

## Without the biggest winners

2020 (the crash and the rebound) was the recommendation's best year. Its daily returns taken out, the rest chained together (all lines over the same years):

| Rule | All years | Without 2020 | Without its own best year | Max drawdown | Max drawdown without 2020 |
|---|---|---|---|---|---|
| Recommended | 30.8% | 23.4% | 23.4% (no 2020) | -33% | -33% |
| S&P 500 in pounds | 13.6% | 13.4% | 11.8% (no 2021) | -25% | -22% |
| Your current rule (LQQ3 above its 200-day avg) | 29.1% | 22.5% | 21.3% (no 2023) | -56% | -55% |
| LQQ3 buy and hold | 30.9% | 23.2% | 19.5% (no 2023) | -79% | -79% |

2017-21 without 2020: 23.2% a year against 16.7% for the S&P 500 in pounds (2022-26 unchanged). Trades in 2020: 43 of 363; win rate without them 69%.

With its best trades taken out (the money stays in cash for those sessions): 5 best: 25.6% a year (+12.0% against the S&P); 10 best: 21.9% a year (+8.3% against the S&P); 20 best: 16.3% a year (+2.7% against the S&P). The ten best: 3LUS 2020-03-23 +22%, LQQ3 2022-11-10 +20%, LQQ3 2020-04-21 +17%, 3LUS 2020-04-08 +15%, 3LUS 2020-04-21 +14%, 3LUS 2022-11-10 +14%, 3LUS 2020-05-13 +13%, LQQ3 2018-02-22 +12%, LQQ3 2024-05-01 +12%, LQQ3 2020-11-10 +11%.


## Two funds only: LQQ3 and 3LUS (the earlier test)

Generated by `python -m backtest.isa` on 2026-11-19. See the module docstring for the rules. Real London prices of LQQ3 (3x Nasdaq-100) and 3LUS (3x S&P 500) in pence; signals at the US close, trades at the next London open, 10 bps a side, sterling interest on idle cash. Benchmark: the S&P 500 in pounds, 13.6% a year (16.3% in 2017-21, 10.9% in 2022-26). Targets: at least 60% winning trades and 15% a year above the S&P 500.

208 settings tested; 17 reached both targets over the whole period and 1 in both 2017-21 and 2022-26.

By portfolio:

| Portfolio | Settings | Median CAGR | Median vs S&P | Median 2022-26 vs S&P | Median win rate | Median max drawdown | Beat S&P by 15%+ in both halves |
|---|---|---|---|---|---|---|---|
| LQQ3 alone | 52 | 19.5% | +6.0% | +3.5% | 69.3% | -41.7% | 0 |
| 3LUS alone | 52 | 9.0% | -4.6% | +3.1% | 70.2% | -48.2% | 0 |
| both funds, half each | 52 | 14.6% | +1.0% | +4.1% | 69.9% | -41.4% | 1 |
| one at a time, all in, more oversold first | 52 | 12.9% | -0.6% | +0.4% | 69.5% | -46.2% | 0 |

The pick chosen on 2017-21 only (best return among settings that won at least 60% of their 2017-21 trades) first, then the settings that beat the S&P by 15% or more in both halves:

| Setting | CAGR | vs S&P 500 | 2017-21 vs S&P | 2022-26 vs S&P | Win rate | Trades per year | Avg hold (sessions) | Max drawdown | Worst year |
|---|---|---|---|---|---|---|---|---|---|
| LQQ3 alone: IBS < 0.2, above 200-day; sell RSI(2) > 70 | 33.4% | +19.8% | +56.4% | -8.0% | 68.1% | 21 | 4 | -34.4% | -14.8% |
| both funds, half each: IBS < 0.1; sell RSI(2) > 70 | 34.0% | +20.5% | +26.0% | +15.3% | 69.6% | 34 | 4 | -42.6% | -17.9% |

For comparison:

| Rule | CAGR | vs S&P 500 | 2017-21 vs S&P | 2022-26 vs S&P | Win rate | Trades per year | Avg hold (sessions) | Max drawdown | Worst year |
|---|---|---|---|---|---|---|---|---|---|
| Your current ISA rule: LQQ3 while QQQ is above its 200-day avg | 29.1% | +15.5% | +28.9% | +3.8% | 27.3% | 2 | 94 | -56.2% | -43.5% |
| Both funds, half each, while their index is above its 200-day avg | 23.9% | +10.3% | +21.4% | +0.6% | 31.9% | 5 | 85 | -50.2% | -42.0% |

## The steadiest family: buy when the index closes in the bottom 10% of its day's range (IBS < 0.1)

Every IBS < 0.1 setting, best first. Picked out after seeing the results, so treat the 2022-26 column as the realistic expectation rather than the whole-period one:

| Setting | CAGR | vs S&P 500 | 2017-21 vs S&P | 2022-26 vs S&P | Win rate | Trades per year | Avg hold (sessions) | Max drawdown | Worst year |
|---|---|---|---|---|---|---|---|---|---|
| LQQ3 alone: IBS < 0.1; sell RSI(2) > 70 | 40.9% | +27.4% | +42.5% | +14.1% | 68.0% | 17 | 4 | -45.2% | -10.8% |
| LQQ3 alone: IBS < 0.1, above 200-day; sell RSI(2) > 70 | 36.5% | +22.9% | +46.3% | +3.5% | 69.6% | 14 | 4 | -37.8% | -16.6% |
| LQQ3 alone: IBS < 0.1; sell close > 5-day avg | 35.7% | +22.1% | +34.9% | +10.8% | 69.2% | 19 | 3 | -38.6% | -5.3% |
| LQQ3 alone: IBS < 0.1; sell close > prior high | 34.9% | +21.4% | +33.7% | +10.5% | 70.8% | 17 | 4 | -41.4% | -13.1% |
| both funds, half each: IBS < 0.1; sell RSI(2) > 70 | 34.0% | +20.5% | +26.0% | +15.3% | 69.6% | 34 | 4 | -42.6% | -17.9% |
| LQQ3 alone: IBS < 0.1, above 200-day; sell close > 5-day avg | 31.7% | +18.2% | +34.6% | +4.1% | 70.0% | 15 | 3 | -30.3% | -16.6% |
| both funds, half each: IBS < 0.1; sell close > 5-day avg | 30.8% | +17.2% | +22.0% | +12.8% | 70.0% | 37 | 3 | -32.9% | -4.6% |
| LQQ3 alone: IBS < 0.1, above 200-day; sell close > prior high | 30.7% | +17.1% | +33.9% | +2.7% | 71.7% | 14 | 4 | -36.9% | -16.6% |
| one at a time, all in, more oversold first: IBS < 0.1; sell RSI(2) > 70 | 30.3% | +16.7% | +19.3% | +14.3% | 68.8% | 21 | 4 | -45.6% | -32.0% |
| both funds, half each: IBS < 0.1; sell close > prior high | 29.7% | +16.1% | +24.1% | +8.9% | 70.4% | 35 | 3 | -38.4% | -26.3% |
| both funds, half each: IBS < 0.1, above 200-day; sell RSI(2) > 70 | 27.1% | +13.6% | +22.9% | +5.2% | 70.6% | 27 | 4 | -29.4% | -9.8% |
| one at a time, all in, more oversold first: IBS < 0.1; sell close > prior high | 26.8% | +13.3% | +23.4% | +4.2% | 70.1% | 22 | 4 | -43.6% | -35.5% |
| one at a time, all in, more oversold first: IBS < 0.1, above 200-day; sell RSI(2) > 70 | 26.2% | +12.6% | +22.3% | +4.0% | 69.6% | 17 | 4 | -37.5% | -13.8% |
| 3LUS alone: IBS < 0.1; sell RSI(2) > 70 | 26.1% | +12.6% | +9.8% | +15.3% | 71.1% | 17 | 4 | -48.6% | -39.3% |
| one at a time, all in, more oversold first: IBS < 0.1; sell close > 5-day avg | 25.6% | +12.0% | +17.0% | +7.4% | 69.0% | 24 | 3 | -44.8% | -27.4% |
| 3LUS alone: IBS < 0.1; sell close > 5-day avg | 24.8% | +11.2% | +8.7% | +13.6% | 70.8% | 18 | 3 | -38.8% | -21.0% |
| 3LUS alone: IBS < 0.1; sell close > prior high | 23.6% | +10.1% | +14.0% | +6.4% | 70.0% | 17 | 3 | -44.5% | -38.6% |
| one at a time, all in, more oversold first: IBS < 0.1, above 200-day; sell close > prior high | 22.8% | +9.2% | +22.8% | -2.6% | 71.4% | 18 | 4 | -34.4% | -13.8% |
| both funds, half each: IBS < 0.1, above 200-day; sell close > prior high | 22.1% | +8.6% | +18.2% | -0.0% | 70.7% | 28 | 4 | -29.5% | -9.8% |
| both funds, half each: IBS < 0.1, above 200-day; sell close > 5-day avg | 22.1% | +8.5% | +12.6% | +4.7% | 69.9% | 30 | 3 | -23.2% | -9.8% |
| one at a time, all in, more oversold first: IBS < 0.1, above 200-day; sell close > 5-day avg | 20.8% | +7.2% | +15.7% | -0.4% | 69.5% | 19 | 3 | -32.7% | -13.8% |
| 3LUS alone: IBS < 0.1, above 200-day; sell RSI(2) > 70 | 17.4% | +3.8% | +1.4% | +6.1% | 71.8% | 13 | 4 | -36.4% | -18.5% |
| 3LUS alone: IBS < 0.1, above 200-day; sell close > prior high | 13.2% | -0.3% | +2.8% | -3.3% | 69.7% | 13 | 4 | -33.9% | -21.8% |
| 3LUS alone: IBS < 0.1, above 200-day; sell close > 5-day avg | 12.2% | -1.4% | -7.4% | +4.8% | 69.8% | 14 | 3 | -40.8% | -24.6% |

## Calendar years

| Year | LQQ3 alone: IBS < 0.2, above 200-day; sell RSI(2) > 70 | both funds, half each: IBS < 0.1; sell RSI(2) > 70 | S&P 500 in pounds |
|---|---|---|---|
| 2017 | +44.2% | +15.1% | +11.5% |
| 2018 | +32.0% | -17.9% | +0.4% |
| 2019 | +36.1% | +46.1% | +25.8% |
| 2020 | +278.7% | +148.1% | +14.7% |
| 2021 | +49.3% | +65.3% | +30.1% |
| 2022 | -14.8% | +18.3% | -8.6% |
| 2023 | +25.1% | +40.0% | +19.3% |
| 2024 | +27.9% | +68.9% | +27.2% |
| 2025 | -9.1% | -5.4% | +9.4% |
| 2026 | -7.5% | +17.4% | +9.0% |

Worst trades of the 2017-21 pick: LQQ3 2025-02-24 to 2025-03-18 -28.4%, LQQ3 2018-03-14 to 2018-04-05 -20.9%, LQQ3 2026-10-07 to 2026-10-22 -18.9%, LQQ3 2022-01-14 to 2022-01-31 -18.0%, LQQ3 2023-10-16 to 2023-11-01 -13.6%.

## Would 3x short funds help? (tested on the US funds)

London prices of the short funds (e.g. 3ULS, the S&P 500 3x short) are not approved for this repository yet, so the question is answered on the US 3x funds, in dollars at the next US open: long TQQQ / UPRO after dips while the index is above its 200-day average, and short via SQQQ / SPXU after short rises (the mirror rules) while it is below. Two places, half the account each.

Across the 26 rule pairs, adding the short side changed the yearly return by -2.4% (median; 23% of pairs improved), the 2022-26 return by -3.8%, and the max drawdown by -10.8%. The short side alone: median 0.1% a year with 53% winning trades.

| Rule | Sides | CAGR | 2017-21 | 2022-26 | Win rate | Max drawdown |
|---|---|---|---|---|---|---|
| 3 lower closes, sell RSI(2) > 70 | long dips + short rips | 7.2% | 13.3% | 1.5% | 68.8% | -50.7% |
| 3 lower closes, sell RSI(2) > 70 | long dips only | 11.3% | 11.2% | 11.3% | 71.2% | -30.7% |
| 3 lower closes, sell close > 5-day avg | long dips + short rips | 8.7% | 12.1% | 5.4% | 66.4% | -29.8% |
| 3 lower closes, sell close > 5-day avg | long dips only | 10.4% | 8.6% | 12.2% | 69.9% | -32.5% |
| 3 lower closes, sell close > prior high | long dips + short rips | 8.0% | 10.4% | 5.7% | 66.4% | -40.0% |
| 3 lower closes, sell close > prior high | long dips only | 8.2% | 7.0% | 9.5% | 69.1% | -34.5% |
| 5-day low close, sell 5-day high close | long dips + short rips | 9.3% | 13.7% | 5.0% | 69.3% | -56.6% |
| 5-day low close, sell 5-day high close | long dips only | 17.1% | 20.1% | 14.2% | 72.3% | -34.1% |
| 5-day low close, sell RSI(2) > 70 | long dips + short rips | 10.2% | 19.3% | 1.9% | 69.1% | -56.6% |
| 5-day low close, sell RSI(2) > 70 | long dips only | 21.3% | 24.8% | 17.9% | 74.8% | -34.0% |
| 5-day low close, sell close > 5-day avg | long dips + short rips | 13.0% | 16.6% | 9.5% | 65.9% | -39.2% |
| 5-day low close, sell close > 5-day avg | long dips only | 19.0% | 20.9% | 17.1% | 70.3% | -35.2% |
| 5-day low close, sell close > prior high | long dips + short rips | 17.1% | 26.4% | 8.6% | 67.8% | -41.0% |
| 5-day low close, sell close > prior high | long dips only | 22.0% | 30.0% | 14.5% | 72.5% | -28.5% |
| 7-day low close, sell 7-day high close | long dips + short rips | 4.7% | 2.9% | 6.5% | 70.4% | -68.0% |
| 7-day low close, sell 7-day high close | long dips only | 15.2% | 23.0% | 8.0% | 72.6% | -29.9% |
| 7-day low close, sell RSI(2) > 70 | long dips + short rips | 14.3% | 20.4% | 8.5% | 68.9% | -50.3% |
| 7-day low close, sell RSI(2) > 70 | long dips only | 20.3% | 24.0% | 16.7% | 72.9% | -27.9% |
| 7-day low close, sell close > 5-day avg | long dips + short rips | 15.4% | 18.6% | 12.3% | 65.7% | -33.4% |
| 7-day low close, sell close > 5-day avg | long dips only | 17.1% | 20.9% | 13.4% | 69.5% | -26.4% |
| 7-day low close, sell close > prior high | long dips + short rips | 17.8% | 22.2% | 13.5% | 67.6% | -36.6% |
| 7-day low close, sell close > prior high | long dips only | 17.8% | 23.4% | 12.4% | 72.0% | -28.0% |
| IBS < 0.1, sell RSI(2) > 70 | long dips + short rips | 26.6% | 38.6% | 15.5% | 69.1% | -50.5% |
| IBS < 0.1, sell RSI(2) > 70 | long dips only | 28.8% | 45.0% | 14.3% | 74.0% | -29.7% |
| IBS < 0.1, sell close > 5-day avg | long dips + short rips | 30.0% | 42.9% | 18.2% | 66.8% | -33.2% |
| IBS < 0.1, sell close > 5-day avg | long dips only | 26.9% | 40.7% | 14.3% | 71.9% | -27.5% |
| IBS < 0.1, sell close > prior high | long dips + short rips | 29.8% | 36.1% | 23.7% | 70.5% | -38.5% |
| IBS < 0.1, sell close > prior high | long dips only | 26.3% | 43.1% | 11.4% | 75.6% | -27.9% |
| IBS < 0.2, sell RSI(2) > 70 | long dips + short rips | 22.0% | 40.7% | 5.7% | 68.2% | -49.2% |
| IBS < 0.2, sell RSI(2) > 70 | long dips only | 28.5% | 50.7% | 9.4% | 71.9% | -31.8% |
| IBS < 0.2, sell close > 5-day avg | long dips + short rips | 29.7% | 46.7% | 14.6% | 65.3% | -36.4% |
| IBS < 0.2, sell close > 5-day avg | long dips only | 31.4% | 51.9% | 13.5% | 69.1% | -29.8% |
| IBS < 0.2, sell close > prior high | long dips + short rips | 20.7% | 31.7% | 10.6% | 67.7% | -41.0% |
| IBS < 0.2, sell close > prior high | long dips only | 23.3% | 43.1% | 6.1% | 71.7% | -30.0% |
| RSI(2) < 10, sell RSI(2) > 70 | long dips + short rips | 6.1% | 10.1% | 2.2% | 70.7% | -50.4% |
| RSI(2) < 10, sell RSI(2) > 70 | long dips only | 9.1% | 8.6% | 9.5% | 71.4% | -32.8% |
| RSI(2) < 10, sell close > 5-day avg | long dips + short rips | 8.2% | 8.7% | 7.7% | 66.5% | -33.9% |
| RSI(2) < 10, sell close > 5-day avg | long dips only | 8.8% | 5.7% | 12.0% | 68.2% | -35.3% |
| RSI(2) < 10, sell close > prior high | long dips + short rips | 6.9% | 9.2% | 4.6% | 68.4% | -39.3% |
| RSI(2) < 10, sell close > prior high | long dips only | 5.9% | 5.1% | 6.6% | 69.6% | -35.2% |
| RSI(2) < 20, sell RSI(2) > 70 | long dips + short rips | 16.8% | 25.4% | 8.8% | 71.5% | -53.6% |
| RSI(2) < 20, sell RSI(2) > 70 | long dips only | 24.1% | 27.7% | 20.6% | 76.0% | -30.5% |
| RSI(2) < 20, sell close > 5-day avg | long dips + short rips | 17.2% | 20.9% | 13.6% | 67.8% | -41.5% |
| RSI(2) < 20, sell close > 5-day avg | long dips only | 20.8% | 22.0% | 19.7% | 71.9% | -33.8% |
| RSI(2) < 20, sell close > prior high | long dips + short rips | 19.9% | 25.8% | 14.2% | 70.0% | -44.8% |
| RSI(2) < 20, sell close > prior high | long dips only | 22.4% | 26.4% | 18.4% | 74.3% | -28.5% |
| RSI(2) < 5, sell RSI(2) > 70 | long dips + short rips | 9.5% | 8.1% | 11.0% | 73.8% | -41.3% |
| RSI(2) < 5, sell RSI(2) > 70 | long dips only | 9.2% | 7.8% | 10.6% | 75.0% | -26.7% |
| RSI(2) < 5, sell close > 5-day avg | long dips + short rips | 10.4% | 7.9% | 13.0% | 70.4% | -23.2% |
| RSI(2) < 5, sell close > 5-day avg | long dips only | 8.9% | 7.2% | 10.6% | 71.9% | -23.2% |
| RSI(2) < 5, sell close > prior high | long dips + short rips | 7.1% | 5.7% | 8.6% | 72.6% | -30.4% |
| RSI(2) < 5, sell close > prior high | long dips only | 5.6% | 4.7% | 6.6% | 73.6% | -26.7% |

Trend rules (hold the 3x fund above the 200-day average; below it hold cash, or the 3x short fund):

| Rule | CAGR | 2017-21 | 2022-26 | Win rate | Max drawdown |
|---|---|---|---|---|---|
| trend: TQQQ above 200-day, else cash | 32.2% | 50.7% | 15.9% | 27.3% | -57.7% |
| trend: TQQQ above 200-day, else SQQQ | 5.5% | 14.1% | -2.5% | 20.9% | -89.1% |
| trend: UPRO above 200-day, else cash | 19.2% | 30.5% | 8.8% | 40.0% | -51.2% |
| trend: UPRO above 200-day, else SPXU | -4.2% | -0.4% | -7.9% | 24.0% | -87.1% |

## Caveats

- 208 settings were tried; whole-period winners are flattered by selection. The pick was chosen on 2017-21 only, so its 2022-26 column is the honest test.
- 3x funds can lose most of their value in a crash, and nothing can be sold before the next London open after a US-close signal; the London open of a 3x fund can also have a wide spread.
- The short-side test uses US funds at the US open; London short funds trade at the London open, which the long-side results show can cost several points a year.
- Prices on days nothing traded: Yahoo often repeats the last price (about a third of 2018 for LQQ3 and 3LUS) and sometimes shows a false one (3LUS, 3LDE and 3LEU +15-27% on 26-27 June 2017, back the next day). Those days are valued at the price implied by the index since the last real price (fill_stale_prints; within about 2% a day of real prices where both exist), so trades happen on schedule as with a market maker's quote. Trading at the stale or false prices instead would move single years by more than ten points, so 2017-18 are the least certain years.
