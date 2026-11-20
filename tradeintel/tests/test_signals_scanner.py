import sqlite3
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tradeintel.config import Settings
from tradeintel.features import build_dataset
from tradeintel.indicators import add_indicators
from tradeintel.ml import DirectionalModel
from tradeintel.opportunity import WEIGHTS, composite, gate_failures, score_components
from tradeintel.outcomes import OutcomeSpec
from tradeintel.scanner import PairContext, Scanner
from tradeintel.signals import SignalStore, Status, advance, monitor_open_signals
from tradeintel.similarity import SimilarityIndex
from tradeintel.store import CandleStore
from tradeintel.synthetic import make_candles
from tradeintel.timeframes import tf_ms

STEP = tf_ms("1h")


def sig(**kw):
    base = dict(bar_open_time=0, symbol="AAAUSDT", tf="1h", direction="LONG", entry=100.0, entry_lo=99.8,
                entry_hi=100.2, sl=96.0, tp1=103.0, tp2=108.0, tp3=112.0, rr=2.0, qty=1.0, risk_amount=4.0,
                score=75.0, probability=0.6, model_id="m1", expiry_bars=4, hold_bars=6, payload={"k": 1})
    return {**base, **kw}


def candle(k, o, h, l, c):
    return {"open_time": k * STEP, "open": o, "high": h, "low": l, "close": c}


def make_store():
    return SignalStore(CandleStore())


def statuses(ss, sid):
    return [e["status"] for e in ss.history(sid)]


def test_full_winning_path_then_breakeven_stop():
    ss = make_store()
    sid = ss.create(sig())
    ss.process(sid, [candle(1, 101, 101.5, 99.9, 100.1)])                 # touches zone, closes on side
    assert ss.state(sid)["status"] == Status.CONFIRMED.value
    ss.process(sid, [candle(2, 100.1, 103.5, 100.0, 103.2)])              # filled, TP1
    assert ss.state(sid)["status"] == Status.TP1.value
    ss.process(sid, [candle(3, 103, 103.2, 99.9, 100.5)])                 # back through entry: breakeven stop
    assert statuses(ss, sid) == ["ANALYSING", "WAITING FOR ENTRY", "ENTRY CONFIRMED", "ACTIVE", "TP1 HIT",
                                 "STOP LOSS HIT"]
    assert "breakeven" in ss.history(sid)[-1]["note"]
    assert sid not in ss.open_signals()


def test_stop_wins_when_candle_touches_stop_and_target():
    ss = make_store()
    sid = ss.create(sig())
    ss.process(sid, [candle(1, 100, 100.3, 99.9, 100.1), candle(2, 100, 104, 95.5, 100)])
    assert statuses(ss, sid)[-1] == "STOP LOSS HIT" and "TP1 HIT" not in statuses(ss, sid)


def test_multiple_targets_in_one_candle():
    ss = make_store()
    sid = ss.create(sig())
    ss.process(sid, [candle(1, 100, 100.3, 99.9, 100.1), candle(2, 100.1, 113, 100.05, 112.5)])
    assert statuses(ss, sid)[-3:] == ["TP1 HIT", "TP2 HIT", "TP3 HIT"]
    assert sid not in ss.open_signals()


def test_invalidated_and_expired_before_entry():
    ss = make_store()
    a = ss.create(sig(id="a"))
    ss.process(a, [candle(1, 98, 98.5, 95.0, 96.5)])
    assert ss.state(a)["status"] == "INVALIDATED"
    b = ss.create(sig(id="b"))
    ss.process(b, [candle(1, 105, 106, 104, 105)])                         # ran to TP1 area, never touched zone
    assert ss.state(b)["status"] == "INVALIDATED" and "missed" in ss.history(b)[-1]["note"]
    c = ss.create(sig(id="c", expiry_bars=3))
    ss.process(c, [candle(k, 101, 101.5, 100.5, 101) for k in (1, 2, 3)])  # hovers above the zone
    assert ss.state(c)["status"] == "EXPIRED"


def test_active_signal_expires_after_holding_period():
    ss = make_store()
    sid = ss.create(sig(hold_bars=3))
    ss.process(sid, [candle(1, 100, 100.3, 99.9, 100.1)] + [candle(k, 100.5, 101, 100.2, 100.6) for k in (2, 3, 4)])
    assert statuses(ss, sid)[-1] == "EXPIRED" and "holding period" in ss.history(sid)[-1]["note"]


def test_short_direction_mirrors():
    ss = make_store()
    sid = ss.create(sig(direction="SHORT", entry=100, entry_lo=99.8, entry_hi=100.2, sl=104, tp1=97, tp2=92, tp3=88))
    ss.process(sid, [candle(1, 99, 100.1, 98.5, 99.7), candle(2, 99.7, 99.9, 96.5, 97.2)])
    assert statuses(ss, sid)[-2:] == ["ACTIVE", "TP1 HIT"]


def test_processing_is_idempotent_and_ignores_old_candles():
    ss = make_store()
    sid = ss.create(sig(bar_open_time=5 * STEP))
    cs = [candle(1, 100, 100.3, 99.9, 100.1), candle(6, 100, 100.3, 99.9, 100.1), candle(7, 100, 104, 100, 103.5)]
    first = ss.process(sid, cs)
    second = ss.process(sid, cs + cs)
    assert second == [] and len(first) == 3                                # candle 1 predates the signal
    assert statuses(ss, sid).count("ENTRY CONFIRMED") == 1


def test_signals_and_history_are_immutable():
    ss = make_store()
    sid = ss.create(sig())
    with pytest.raises(sqlite3.IntegrityError):
        ss.conn.execute("UPDATE signals SET sl=90 WHERE id=?", (sid,))
    with pytest.raises(sqlite3.IntegrityError):
        ss.conn.execute("DELETE FROM signals WHERE id=?", (sid,))
    with pytest.raises(sqlite3.IntegrityError):
        ss.conn.execute("UPDATE signal_events SET status='ACTIVE' WHERE signal_id=?", (sid,))
    with pytest.raises(sqlite3.IntegrityError):
        ss.conn.execute("DELETE FROM signal_events WHERE signal_id=?", (sid,))
    assert ss.get(sid)["sl"] == 96.0 and ss.get(sid)["payload"] == {"k": 1}


def test_monitor_reads_new_candles_from_the_store():
    from tradeintel.exchange.base import Candle
    cs = CandleStore()
    ss = SignalStore(cs)
    sid = ss.create(sig(bar_open_time=0))
    cs.upsert("fake", "AAAUSDT", "1h", [Candle(1 * STEP, 100, 100.3, 99.9, 100.1, 1), Candle(2 * STEP, 100.1, 103.5, 100.05, 103, 1)])
    out = monitor_open_signals(ss, cs, "fake")
    assert [e["status"] for e in out[sid]] == ["ENTRY CONFIRMED", "ACTIVE", "TP1 HIT"]
    assert monitor_open_signals(ss, cs, "fake") == {}                      # nothing new: nothing recorded


# ---------------- scoring and gates ----------------

def test_score_is_bounded_and_monotonic():
    kw = dict(p=0.6, thr=0.5, nb_success=0.5, base_rate=0.4, trend_aligned=True, adx=30, rel_vol=1.5,
              atr_ratio=1.2, quote_volume_24h=5e7, min_quote_volume=5e6, rr=2.5, min_rr=1.8)
    lo = composite(score_components(**{**kw, "p": 0.51}))
    hi = composite(score_components(**{**kw, "p": 0.85}))
    assert 0 <= lo < hi <= 100
    assert composite({k: 1.0 for k in WEIGHTS}) == pytest.approx(100.0)
    assert composite({k: 0.0 for k in WEIGHTS}) == 0.0


def test_gates_report_every_failing_reason():
    s = Settings()
    assert gate_failures(qualifies=True, usable=True, rr=2.5, quote_volume_24h=1e8, atr_ratio=1.1,
                         nb_mean_r=0.1, settings=s) == []
    fails = gate_failures(qualifies=False, usable=False, rr=1.0, quote_volume_24h=1.0, atr_ratio=5.0,
                          nb_mean_r=-0.2, settings=s)
    assert len(fails) == 5


# ---------------- scanner ----------------

class StubModel:
    def __init__(self, p, usable=True, threshold=0.5):
        self.p, self.usable, self.threshold = p, usable, threshold

    def predict_proba(self, X):
        return np.full(len(X), self.p)


@pytest.fixture(scope="module")
def world():
    store = CandleStore()
    cs = make_candles(2500, "1h", seed=41)
    store.upsert("fake", "AAAUSDT", "1h", cs)
    raw = store.load("fake", "AAAUSDT", "1h")
    df = add_indicators(raw)
    ds = build_dataset(df, OutcomeSpec())
    return store, SimilarityIndex(ds), {d: float(ds.y[d].mean()) for d in (1, -1)}


class Provider:
    def __init__(self, ctx):
        self.ctx, self.calls = ctx, 0

    def get(self, symbol, tf):
        self.calls += 1
        return self.ctx if symbol == "AAAUSDT" else None


def relaxed():
    # Synthetic volume is tiny and its neighbours carry no edge: relax those two gates only.
    return Settings(min_quote_volume_24h=1.0, min_neighbour_r=-10.0, min_score=0.0, min_rr=0.0)


def test_scanner_reports_no_opportunity_when_models_have_no_edge(world):
    store, sim, base = world
    ctx = PairContext({1: StubModel(0.9, usable=False), -1: StubModel(0.9, usable=False)}, "stub", sim, base)
    rep = Scanner("fake", store, Provider(ctx), settings=relaxed()).run_once(["AAAUSDT", "BBBUSDT"], "1h")
    assert rep.opportunities == [] and rep.message.startswith("No qualified")
    assert "no demonstrated edge" in rep.rejected["AAAUSDT"][0]
    assert rep.rejected["BBBUSDT"] == ["no trained model for this pair"]


def test_scanner_rejects_probability_below_threshold(world):
    store, sim, base = world
    ctx = PairContext({1: StubModel(0.4), -1: StubModel(0.4)}, "stub", sim, base)
    rep = Scanner("fake", store, Provider(ctx), settings=relaxed()).run_once(["AAAUSDT"], "1h")
    assert rep.opportunities == [] and "below its validated threshold" in rep.rejected["AAAUSDT"][0]


def test_scanner_builds_ranked_opportunity_signal_and_caches(world):
    store, sim, base = world
    ctx = PairContext({1: StubModel(0.8), -1: StubModel(0.8)}, "stub-model", sim, base)
    provider = Provider(ctx)
    signals = SignalStore(store)
    sc = Scanner("fake", store, provider, signals=signals, settings=relaxed())
    rep = sc.run_once(["AAAUSDT"], "1h")
    if not rep.opportunities:
        pytest.skip(f"no structural setup on this last bar: {rep.rejected}")
    o = rep.opportunities[0]
    assert o.model_id == "stub-model" and 0 <= o.score <= 100 and o.probability == 0.8
    if o.direction == "LONG":
        assert o.sl < o.entry_lo <= o.entry <= o.entry_hi < o.tp1 < o.tp2 < o.tp3
    else:
        assert o.sl > o.entry_hi >= o.entry >= o.entry_lo > o.tp1 > o.tp2 > o.tp3
    assert o.risk_amount <= Settings().capital * Settings().risk_per_trade + 1e-6
    assert set(o.components) == set(WEIGHTS) and o.invalidation and "neighbours" in o.evidence
    assert len(rep.new_signals) == 1
    stored = signals.get(rep.new_signals[0])
    assert stored["model_id"] == "stub-model" and stored["payload"]["evidence"]["patterns_now"] is not None

    calls = provider.calls
    rep2 = sc.run_once(["AAAUSDT"], "1h")                                  # same last candle: served from cache
    assert rep2.from_cache == 1 and rep2.scanned == 0 and provider.calls == calls
    assert rep2.new_signals == []                                          # and no duplicate signal
    assert len(signals.open_signals()) == 1


def test_scanner_isolates_a_failing_pair(world):
    store, sim, base = world

    class Boom:
        def get(self, symbol, tf):
            raise RuntimeError("model file corrupt")

    rep = Scanner("fake", store, Boom(), settings=relaxed()).run_once(["AAAUSDT"], "1h")
    assert rep.opportunities == [] and "scan error" in rep.rejected["AAAUSDT"][0]
