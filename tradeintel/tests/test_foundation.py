import dataclasses

import numpy as np
import pandas as pd
import pytest

from tradeintel.backtest import chronological_splits, run_backtest
from tradeintel.config import Settings
from tradeintel.indicators import add_indicators, swing_levels
from tradeintel.integrity import find_gaps, validate_ohlcv
from tradeintel.store import CandleStore
from tradeintel.sync import Syncer
from tradeintel.synthetic import FakeExchange, make_candles
from tradeintel.timeframes import tf_ms
from tradeintel.tradeparams import compute_trade_params, position_size

TF = "1h"
STEP = tf_ms(TF)


def make_sync(n=300, closed=250):
    candles = make_candles(n, TF)
    now = candles[closed].open_time + STEP // 2        # candle `closed` is still forming
    ex = FakeExchange({("AAAUSDT", TF): candles}, now)
    store = CandleStore()
    return ex, None, Syncer(ex, store), store, candles, now


def test_gap_detection():
    times = [0, STEP, 2 * STEP, 5 * STEP, 6 * STEP]
    gaps = find_gaps(times, STEP)
    assert len(gaps) == 1 and gaps[0].missing == 2 and gaps[0].start_ms == 3 * STEP


def test_validation_flags_bad_rows():
    df = pd.DataFrame({"open_time": [1, 2], "open": [10, 10], "high": [9, 12], "low": [8, 9],
                       "close": [10, 11], "volume": [1, -1]})
    problems = set(validate_ohlcv(df)["problem"])
    assert "high below open/close" in problems and "negative volume" in problems


def test_sync_stores_only_closed_candles_and_is_incremental():
    ex, _, syncer, store, candles, now = make_sync()
    since = candles[0].open_time
    added = syncer.sync("AAAUSDT", TF, since, now_ms=now)
    assert added == 250                                   # candles 0..249 closed, 250 still forming
    assert store.bounds("fake", "AAAUSDT", TF)[1] == candles[249].open_time
    assert store.count("fake", "AAAUSDT", TF) == len(set(store.open_times("fake", "AAAUSDT", TF)))

    ex.requests.clear()
    assert syncer.sync("AAAUSDT", TF, since, now_ms=now) == 0
    assert ex.requests == []                              # nothing new can exist: zero API calls

    later = now + 10 * STEP
    syncer.sync("AAAUSDT", TF, since, now_ms=later)
    assert all(start >= candles[250].open_time for _, _, start in ex.requests)   # no re-download


def test_sync_repairs_gaps_with_targeted_requests():
    ex, _, syncer, store, candles, now = make_sync()
    syncer.sync("AAAUSDT", TF, candles[0].open_time, now_ms=now)
    with store.conn:
        store.conn.execute("DELETE FROM candles WHERE open_time BETWEEN ? AND ?",
                           (candles[100].open_time, candles[104].open_time))
    ex.requests.clear()
    assert syncer.sync("AAAUSDT", TF, candles[0].open_time, now_ms=now) == 5
    assert [r[2] for r in ex.requests] == [candles[100].open_time]
    assert find_gaps(store.open_times("fake", "AAAUSDT", TF), STEP) == []


def test_indicators_are_causal():
    df = pd.DataFrame([c.__dict__ for c in make_candles(400, TF)])
    full = add_indicators(df)
    cut = add_indicators(df.iloc[:250])
    cols = ["ema20", "ema50", "rsi14", "atr14", "adx14", "rel_vol"]
    pd.testing.assert_frame_equal(full.iloc[:250][cols], cut[cols])


def test_swing_levels_ignore_unconfirmed_pivots():
    df = add_indicators(pd.DataFrame([c.__dict__ for c in make_candles(300, TF)]))
    i = 200
    highs_now, _ = swing_levels(df, i)
    highs_trunc, _ = swing_levels(df.iloc[:i + 1], i)
    assert highs_now == highs_trunc                      # future bars never change the answer


def test_trade_params_structure_and_sizing():
    df = add_indicators(pd.DataFrame([c.__dict__ for c in make_candles(400, TF)]))
    found = None
    for i in range(80, 399):
        for d in ("LONG", "SHORT"):
            p = compute_trade_params(df, i, d)
            if p:
                found = p
                break
        if found:
            break
    assert found is not None
    if found.direction == "LONG":
        assert found.sl < found.entry < found.tp1 < found.tp2 < found.tp3
    else:
        assert found.sl > found.entry > found.tp1 > found.tp2 > found.tp3
    qty = position_size(10_000, 0.01, found.entry, found.sl)
    assert qty * found.risk_per_unit <= 100.0001        # risk never exceeds 1% of capital


def test_targets_are_strictly_ordered_on_every_bar():
    """Regression: a far swing-level TP1 used to leave TP2 closer to entry than TP1."""
    for seed in (1, 2, 3):
        df = add_indicators(pd.DataFrame([c.__dict__ for c in make_candles(2500, TF, seed=seed)]))
        checked = 0
        for i in range(80, len(df) - 1):
            for d in ("LONG", "SHORT"):
                p = compute_trade_params(df, i, d)
                if p is None:
                    continue
                checked += 1
                if d == "LONG":
                    assert p.sl < p.entry < p.tp1 < p.tp2 < p.tp3, (seed, i, p)
                else:
                    assert p.sl > p.entry > p.tp1 > p.tp2 > p.tp3, (seed, i, p)
        assert checked > 200


def _frame(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 1.0
    df["atr14"] = 2.0
    return df


def test_backtest_executes_at_next_open_with_costs_and_stop_first(monkeypatch):
    from tradeintel import backtest as bt
    from tradeintel.tradeparams import TradeParams

    rows = [(100, 101, 99, 100)] * 3 + [
        (100, 111, 95, 100),       # entry bar touches BOTH stop (96) and TP1 (103): stop must win
        (100, 101, 99, 100), (100, 101, 99, 100),
    ]
    df = _frame(rows)
    monkeypatch.setattr(bt, "compute_trade_params", lambda d, i, dr: TradeParams(
        "LONG", 100.0, 96.0, 103.0, 108.0, 112.0, 4.0, 3.0) if i == 2 else None)
    calls = []

    def strat(d, i):
        calls.append(len(d.iloc[:i + 1]))
        return "LONG" if i == 2 else None

    s = Settings(fee_rate=0.001, slippage=0.0005, min_rr=1.8)
    res = run_backtest(df, strat, s)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.entry_bar == 3 and t.signal_bar == 2          # filled on the bar AFTER the signal
    assert t.entry == pytest.approx(100 * 1.0005)          # adverse slippage on the fill
    assert t.outcome == "SL"
    assert t.pnl < 0 and t.r_multiple < -1.0               # costs push the loss beyond 1R


def test_fees_reduce_profit():
    df = add_indicators(pd.DataFrame([c.__dict__ for c in make_candles(3000, TF, seed=3)]))
    from tradeintel.strategies import ema_pullback
    free = run_backtest(df, ema_pullback, Settings(fee_rate=0.0, slippage=0.0))
    paid = run_backtest(df, ema_pullback, Settings(fee_rate=0.001, slippage=0.0005))
    assert free.metrics["trades"] > 0
    assert paid.equity_curve[-1] < free.equity_curve[-1]


def test_walk_forward_windows_are_chronological_and_disjoint():
    splits = chronological_splits(2000, train=1000, test=250)
    assert splits
    for tr, te in splits:
        assert tr.stop == te.start and max(tr) < min(te)
    tests = [te for _, te in splits]
    assert all(a.stop <= b.start for a, b in zip(tests, tests[1:]))
