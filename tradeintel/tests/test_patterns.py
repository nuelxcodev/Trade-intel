import math

import pandas as pd
import pytest

from tradeintel.indicators import add_indicators
from tradeintel.knowledge import KnowledgeBase, build_events, summarize, validate_chronologically, wilson
from tradeintel.outcomes import OutcomeSpec, baseline_rates, measure
from tradeintel.patterns import detect_all, market_context
from tradeintel.store import CandleStore
from tradeintel.synthetic import make_candles


def frame(n=1500, seed=11):
    return add_indicators(pd.DataFrame([c.__dict__ for c in make_candles(n, "1h", seed=seed)]))


def test_detectors_are_causal():
    df = frame(600)
    full, cut = detect_all(df), detect_all(df.iloc[:400])
    pd.testing.assert_frame_equal(full.iloc[:400], cut)
    pd.testing.assert_frame_equal(market_context(df).iloc[:400], market_context(df.iloc[:400]))


def test_detectors_fire_on_synthetic_data():
    sig = detect_all(frame(4000))
    silent = sig.columns[(sig == 0).all()].tolist()
    # Synthetic wicks are small and symmetric, so pin bars never form; they are tested directly below.
    assert silent == ["pin_bar"], silent


def test_pin_bar_hand_built():
    from tradeintel.patterns import pin_bar
    df = pd.DataFrame({
        "open": [10.0, 10.0, 10.0], "close": [10.1, 10.9, 9.9],
        "high": [10.15, 11.0, 11.0], "low": [9.0, 9.95, 9.9],
    })
    out = pin_bar(df)
    assert out.iloc[0] == 1      # long lower wick, small body, close near the high: bullish hammer
    assert out.iloc[1] == 0      # big body, no rejection wick
    assert out.iloc[2] == -1     # long upper wick, close near the low: bearish shooting star


def _flat(rows, atr=1.0):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["atr14"] = atr
    return df


def test_measure_win_loss_and_stop_first():
    spec = OutcomeSpec(horizon=3, target_atr=1.5, stop_atr=1.0)
    base = [(10, 10.2, 9.8, 10)]
    win = _flat(base + [(10, 11.6, 9.9, 11), (11, 11, 10.5, 11), (11, 11, 10.5, 11), (11, 11, 10.5, 11)])
    assert measure(win, 0, 1, spec)["outcome"] == "win"
    both = _flat(base + [(10, 11.6, 8.9, 10), (10, 10, 10, 10), (10, 10, 10, 10), (10, 10, 10, 10)])
    assert measure(both, 0, 1, spec)["outcome"] == "loss"          # stop first on an ambiguous bar
    short = _flat(base + [(10, 10.1, 8.4, 9), (9, 9, 9, 9), (9, 9, 9, 9), (9, 9, 9, 9)])
    assert measure(short, 0, -1, spec)["outcome"] == "win"
    drift = _flat(base + [(10, 10.3, 9.8, 10)] * 3)
    assert measure(drift, 0, 1, spec)["outcome"] == "timeout"


def test_unresolved_horizon_is_excluded():
    df = _flat([(10, 10.2, 9.8, 10)] * 6)
    assert measure(df, 4, 1, OutcomeSpec(horizon=3)) is None       # only 1 future bar exists


def test_wilson_interval():
    lo, hi = wilson(60, 100)
    assert 0.5 < lo < 0.6 < hi < 0.7
    assert wilson(0, 0)[0] != wilson(0, 0)[0]                       # NaN for empty samples
    assert wilson(5, 5)[1] <= 1.0 and wilson(5, 5)[0] < 1.0         # never claims certainty


def test_knowledge_base_roundtrip_and_stats():
    df = frame(3000)
    spec = OutcomeSpec()
    events = build_events(df, "SYN", "1h", spec)
    assert len(events) > 100
    assert events["bar_index"].max() + 1 + spec.horizon <= len(df)  # nothing unresolved
    kb = KnowledgeBase(CandleStore())
    assert kb.save_events(events, "SYN", "1h") == len(events)
    assert len(kb.events("SYN", "1h")) == len(events)
    assert kb.save_events(events, "SYN", "1h") == len(events)       # re-learning replaces, not duplicates
    assert len(kb.events("SYN", "1h")) == len(events)

    base = baseline_rates(df, spec)
    stats = summarize(events, base, total_bars=len(df))
    assert {"n", "success_rate", "ci_low", "ci_high", "lift", "per_1000_bars"} <= set(stats.columns)
    assert ((stats["ci_low"] <= stats["success_rate"]) & (stats["success_rate"] <= stats["ci_high"])).all()


def test_random_walk_patterns_do_not_hold_out_of_sample():
    """On pure noise (no drift, no regimes worth learning) patterns must not be crowned as edges."""
    df = frame(6000, seed=5)
    spec = OutcomeSpec()
    base = baseline_rates(df, spec)
    val = validate_chronologically(build_events(df, "SYN", "1h", spec), base)
    assert len(val) > 0
    # Multiple comparisons: a few chance 'holds' are possible; most patterns must not hold.
    assert val["holds"].mean() < 0.35


def test_validation_split_is_chronological():
    df = frame(3000)
    events = build_events(df, "SYN", "1h")
    val = validate_chronologically(events, baseline_rates(df, OutcomeSpec()), split=0.7)
    cut = events["bar_index"].quantile(0.7)
    assert val["n_train"].sum() == (events["bar_index"] <= cut).sum()
    assert val["n_test"].sum() == (events["bar_index"] > cut).sum()
