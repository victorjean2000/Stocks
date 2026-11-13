"""Paper trading of two strategies from backtest/strategies.py, each in its own Alpaca paper account.

    python -m trading status            # target, holdings and open orders for each account
    python -m trading run --dry-run     # the orders a run would place, without placing them
    python -m trading run               # cancel this tool's open orders, then place today's

Strategies (rules identical to the backtest):
  momentum  10 of the 500 largest US stocks by 6-1 month return, equal weight, rebalanced once
            after each month end; a holding is kept while it ranks in the top 30 (since 2026-11-13;
            before that it was sold on leaving the top 10), and empty places go to the best-ranked others
  lqq3      TQQQ (3x Nasdaq-100, the dollar stand-in for London-listed LQQ3) while QQQ closes above its
            200-day average, otherwise BIL (T-bills)

Accounts: momentum uses ALPACA_MOMENTUM_KEY_ID / ALPACA_MOMENTUM_SECRET_KEY, or else the main
ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY; lqq3 uses ALPACA_LQQ3_KEY_ID / ALPACA_LQQ3_SECRET_KEY.
Only the paper trading API is ever called. Decisions use closing prices; run after the close and
the market orders queue for the next open (run during market hours and they fill right away).
"""
