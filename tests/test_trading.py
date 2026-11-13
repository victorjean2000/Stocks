"""Paper trader: order planning, the once-per-decision rebalance, and account safety. No network."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading import broker, run as runner, targets
from trading.plan import CASH_BUFFER, Holding, plan_orders
from trading.targets import KEEP_RANK, Target, buffered_top, month_end_decision


def test_only_the_paper_api():
    assert broker.PAPER_URL == "https://paper-api.alpaca.markets"
    assert broker.Paper("k", "s").base == broker.PAPER_URL


def test_first_rebalance_buys_equal_weights_with_a_cash_buffer():
    w = {s: 0.1 for s in "ABCDEFGHIJ"}
    orders = plan_orders(w, {}, 100_000, 100_000, full=True, fractionable={s: True for s in w})
    assert len(orders) == 10 and {o.side for o in orders} == {"buy"}
    assert sum(o.notional for o in orders) == pytest.approx(100_000 * (1 - CASH_BUFFER), abs=1)


def test_rebalance_sells_non_targets_trims_and_tops_up_never_oversells():
    hold = {"OLD": Holding("OLD", "3.5", 7_000, 2_000.0), "A": Holding("A", "100", 70_000, 700.0),
            "B": Holding("B", "10", 20_000, 2_000.0)}
    orders = plan_orders({"A": 0.5, "B": 0.5}, hold, 100_000, 3_000, full=True, fractionable={"A": True, "B": True})
    by = {(o.symbol, o.side): o for o in orders}
    assert by[("OLD", "sell")].qty == "3.5"  # exact quantity, full exit
    assert float(by[("A", "sell")].qty) == pytest.approx((70_000 - 49_750) / 700, rel=1e-5)
    assert by[("B", "buy")].notional == pytest.approx(49_750 - 20_000, abs=0.01)
    assert orders[0].side == "sell" and orders[-1].side == "buy"  # sells first


def test_keep_matched_does_not_trade_drift_but_invests_idle_cash():
    frac = {"A": True, "B": True}
    hold = {"A": Holding("A", "10", 60_000, 6_000.0), "B": Holding("B", "10", 38_000, 3_800.0)}
    assert plan_orders({"A": 0.5, "B": 0.5}, hold, 100_000, 2_000, full=False, fractionable=frac) == []
    hold["B"] = Holding("B", "10", 34_000, 3_400.0)  # 6% idle cash, e.g. after dividends
    orders = plan_orders({"A": 0.5, "B": 0.5}, hold, 100_000, 6_000, full=False, fractionable=frac)
    assert [o.why for o in orders] == ["idle cash", "idle cash"]
    assert sum(o.notional for o in orders) == pytest.approx(6_000 - 500, abs=0.02)


def test_trend_switch_sells_one_fund_and_buys_the_other():
    hold = {"TQQQ": Holding("TQQQ", "812.123456", 99_400, 122.4)}
    orders = plan_orders({"BIL": 1.0}, hold, 100_000, 600, full=False, fractionable={"BIL": True})
    assert [(o.side, o.symbol) for o in orders] == [("sell", "TQQQ"), ("buy", "BIL")]
    assert orders[0].qty == "812.123456" and orders[1].notional == pytest.approx(99_500, abs=0.01)


def test_whole_shares_when_not_fractionable():
    orders = plan_orders({"X": 1.0}, {}, 10_000, 10_000, full=True, fractionable={"X": False}, prices={"X": 333.0})
    assert orders[0].qty == "29" and orders[0].notional is None


def test_month_end_decision_uses_the_trading_calendar():
    days = pd.DatetimeIndex(["2024-03-27", "2024-03-28", "2024-04-01", "2024-04-02"])  # 29 March: Good Friday
    assert month_end_decision(days, pd.Timestamp("2024-04-02")) == pd.Timestamp("2024-03-28")


class FakePaper:
    """Fills market orders at once at fixed prices; enough of the paper API for the runner."""

    def __init__(self, acct_id="acct-1", cash=100_000.0, prices=None):
        self.id, self.cash, self.px = acct_id, cash, prices or {}
        self.pos: dict[str, float] = {}
        self.closed: list[dict] = []
        self.submitted: list[dict] = []

    def account(self):
        eq = self.cash + sum(q * self.px[s] for s, q in self.pos.items())
        return {"id": self.id, "status": "ACTIVE", "equity": str(eq), "cash": str(self.cash),
                "trading_blocked": False, "account_blocked": False}

    def positions(self):
        return [{"symbol": s, "qty": str(q), "market_value": str(q * self.px[s]), "current_price": str(self.px[s])}
                for s, q in self.pos.items() if q > 0]

    def orders(self, status="open", after=None, limit=500):
        return [] if status == "open" else list(self.closed)

    def asset(self, s):
        return {"fractionable": True}

    def latest_prices(self, symbols):
        return {s: self.px[s] for s in symbols}

    def clock(self):
        return {"is_open": True}

    def cancel(self, oid):
        raise AssertionError("nothing to cancel")

    def submit(self, symbol, side, cid, qty=None, notional=None):
        q = float(qty) if qty is not None else notional / self.px[symbol]
        if side == "sell":
            assert q <= self.pos.get(symbol, 0) + 1e-9, "oversold"
            q = -q
        self.pos[symbol] = self.pos.get(symbol, 0) + q
        self.cash -= q * self.px[symbol]
        o = {"client_order_id": cid, "filled_qty": str(abs(q)), "status": "filled", "symbol": symbol, "side": side}
        self.closed.append(o)
        self.submitted.append(o)
        return o


def _target(weights, once=True):
    d = pd.Timestamp("2026-10-30")
    return Target("momentum", weights, d, pd.Timestamp("2026-11-03"), f"momentum-{d:%Y%m%d}", once)


def test_monthly_rebalance_happens_once_per_decision(monkeypatch):
    fake = FakePaper(prices={"A": 100.0, "B": 50.0})
    monkeypatch.setitem(runner.TARGETS, "momentum", lambda p: _target({"A": 0.5, "B": 0.5}))
    first = runner.run_one("momentum", fake, dry_run=False)
    assert first["mode"] == "rebalance" and first["placed"] == 2
    assert all(o["client_order_id"].startswith("momentum-20261030-rebal-") for o in fake.submitted)
    fake.px["A"] = 130.0  # drift: no trades until the next month end
    second = runner.run_one("momentum", fake, dry_run=False)
    assert second["mode"] == "keep matched" and second["orders"] == []


def test_dry_run_places_nothing(monkeypatch):
    fake = FakePaper(prices={"A": 100.0})
    monkeypatch.setitem(runner.TARGETS, "momentum", lambda p: _target({"A": 1.0}))
    res = runner.run_one("momentum", fake, dry_run=True)
    assert res["placed"] == "dry run" and len(res["orders"]) == 1 and fake.submitted == []


def test_two_strategies_never_share_an_account(monkeypatch):
    shared = FakePaper(acct_id="same", prices={"A": 100.0, "TQQQ": 100.0})
    monkeypatch.setattr(runner, "account_for", lambda n: shared)
    monkeypatch.setitem(runner.TARGETS, "momentum", lambda p: _target({"A": 1.0}))
    monkeypatch.setitem(runner.TARGETS, "lqq3", lambda p: _target({"TQQQ": 1.0}, once=False))
    res = runner.run(["momentum", "lqq3"])
    assert "error" not in str(res[0].get("status", ""))
    assert "same paper account" in res[1]["status"]


def test_a_strategy_refuses_an_account_holding_another_strategys_orders(monkeypatch):
    lqq3_account = FakePaper(acct_id="lqq3-acct", prices={"A": 100.0, "BIL": 91.0})
    lqq3_account.closed.append({"client_order_id": "lqq3-20261105-fix-BIL-buy-0d773670", "filled_qty": "1",
                                "status": "filled", "symbol": "BIL", "side": "buy"})
    monkeypatch.setattr(runner, "account_for", lambda n: lqq3_account)  # momentum keys point at the lqq3 account
    monkeypatch.setitem(runner.TARGETS, "momentum", lambda p: _target({"A": 1.0}))
    res = runner.run(["momentum"])
    assert "it is the lqq3 account" in res[0]["status"] and lqq3_account.submitted == []


# ------------------------------------------------------------------ momentum: keep while in the top 30

def test_buffer_keeps_holdings_inside_the_cut_off_and_fills_with_the_best():
    score = -np.arange(40.0)  # column j ranks #(j + 1)
    ok = np.ones(40, bool)
    held = {2, 14, 28, 30, 44}  # ranks 3, 15, 29, 31 and a stock not in the pool any more
    picks = buffered_top(score, ok, 10, 30, held)
    assert picks[:3] == [2, 14, 28]  # kept, best first; rank 31 and the missing stock are sold
    assert picks[3:] == [0, 1, 3, 4, 5, 6, 7]  # the best-ranked stocks not held
    assert set(buffered_top(score, ok, 10, 30, set(picks))) == set(picks)  # stable once traded
    assert buffered_top(score, ok, 10, 10, held) == [2] + [0, 1, 3, 4, 5, 6, 7, 8, 9]  # the old rule


def test_buffer_ignores_stocks_outside_the_pool_or_without_a_score():
    score = -np.arange(6.0)
    score[1] = np.nan
    ok = np.array([True, True, True, False, True, True])
    assert buffered_top(score, ok, 3, 4, {3}) == [0, 2, 4]  # 3 left the pool: sold, never kept


def test_momentum_target_keeps_slipped_holdings_and_reports_why(monkeypatch):
    days = pd.bdate_range("2026-10-26", "2026-11-13")
    names = [f"S{k:02d}" for k in range(40)]
    r6_1 = np.tile(np.linspace(0.9, -0.3, 40), (len(days), 1))  # S00 best, S39 worst, every day
    lab = SimpleNamespace(dates=days, m=SimpleNamespace(close=pd.DataFrame(columns=names, dtype=float)),
                          x={"r6_1": r6_1, "uni": np.ones((len(days), 40), bool),
                             "mcap_rank": np.ones((len(days), 40))})
    monkeypatch.setattr("backtest.strategies.build_lab", lambda: lab)
    monkeypatch.setattr(targets, "sessions", lambda paper: (days, days[-1]))
    fake = FakePaper(prices={s: 10.0 for s in names})
    fake.pos = {"S05": 100.0, "S20": 100.0, "S35": 100.0}  # ranks 6, 21 and 36
    t = targets.momentum_target(fake)
    assert t.decided == pd.Timestamp("2026-10-30") and t.tag == "momentum-20261030"
    assert sorted(t.weights) == ["S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S20"]
    assert all(w == pytest.approx(0.1) for w in t.weights.values())
    assert t.info[f"kept (still in the top {KEEP_RANK})"] == ["S05", "S20"]
    assert t.info[f"sold (below #{KEEP_RANK})"] == {"S35": "#36"}
    assert t.info["rank and 6-1 month return"]["S20"] == "#21 (+28.5%)"
