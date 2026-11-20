"""Pattern detectors.

Each detector returns a Series aligned with the frame: +1 bullish, -1 bearish, 0 none.
Every detector is causal: the value at bar t uses only bars <= t (rolling windows are shifted
by one bar wherever they define a level that bar t is compared against).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sign(cond_bull: pd.Series, cond_bear: pd.Series) -> pd.Series:
    return cond_bull.astype(int) - cond_bear.astype(int)


def engulfing(df: pd.DataFrame) -> pd.Series:
    po, pc = df["open"].shift(1), df["close"].shift(1)
    o, c = df["open"], df["close"]
    body, pbody = (c - o).abs(), (pc - po).abs()
    bull = (pc < po) & (c > o) & (c >= po) & (o <= pc) & (body > pbody)
    bear = (pc > po) & (c < o) & (c <= po) & (o >= pc) & (body > pbody)
    return _sign(bull, bear)


def pin_bar(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    bull = (lower >= 2 * body) & (lower / rng >= 0.6) & ((df["close"] - df["low"]) / rng >= 0.66)
    bear = (upper >= 2 * body) & (upper / rng >= 0.6) & ((df["high"] - df["close"]) / rng >= 0.66)
    return _sign(bull, bear)


def breakout(df: pd.DataFrame, n: int = 20, min_rel_vol: float = 1.5) -> pd.Series:
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    vol_ok = df["rel_vol"] >= min_rel_vol
    return _sign((df["close"] > hi) & vol_ok, (df["close"] < lo) & vol_ok)


def fakeout(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Price pokes beyond the n-bar range but closes back inside it (trapped breakout traders)."""
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    bear = (df["high"] > hi) & (df["close"] < hi)          # upthrust, bearish
    bull = (df["low"] < lo) & (df["close"] > lo)           # spring, bullish
    return _sign(bull, bear)


def liquidity_sweep(df: pd.DataFrame, n: int = 10, min_rel_vol: float = 1.2) -> pd.Series:
    """Short-range extreme is taken out with a rejection wick and a close back beyond the level."""
    lo = df["low"].rolling(n).min().shift(1)
    hi = df["high"].rolling(n).max().shift(1)
    vol_ok = df["rel_vol"] >= min_rel_vol
    mid = (df["high"] + df["low"]) / 2
    bull = (df["low"] < lo) & (df["close"] > lo) & (df["close"] > mid) & vol_ok
    bear = (df["high"] > hi) & (df["close"] < hi) & (df["close"] < mid) & vol_ok
    return _sign(bull, bear)


def volume_spike(df: pd.DataFrame, min_rel_vol: float = 2.5) -> pd.Series:
    spike = df["rel_vol"] >= min_rel_vol
    return _sign(spike & (df["close"] > df["open"]), spike & (df["close"] < df["open"]))


def momentum_shift(df: pd.DataFrame) -> pd.Series:
    prev = df["rsi14"].shift(1)
    bull = (prev < 50) & (df["rsi14"] >= 50) & (df["ema20"] > df["ema50"])
    bear = (prev > 50) & (df["rsi14"] <= 50) & (df["ema20"] < df["ema50"])
    return _sign(bull, bear)


def volatility_expansion(df: pd.DataFrame, ratio: float = 1.5) -> pd.Series:
    base = df["atr14"].rolling(50).mean().shift(1)
    wide = (df["high"] - df["low"]) >= ratio * base
    return _sign(wide & (df["close"] > df["open"]), wide & (df["close"] < df["open"]))


def trend_pullback_resume(df: pd.DataFrame) -> pd.Series:
    up, down = df["ema20"] > df["ema50"], df["ema20"] < df["ema50"]
    bull = up & (df["low"] <= df["ema20"]) & (df["close"] > df["ema20"]) & (df["close"] > df["open"])
    bear = down & (df["high"] >= df["ema20"]) & (df["close"] < df["ema20"]) & (df["close"] < df["open"])
    return _sign(bull, bear)


DETECTORS = {
    "engulfing": engulfing,
    "pin_bar": pin_bar,
    "breakout": breakout,
    "fakeout": fakeout,
    "liquidity_sweep": liquidity_sweep,
    "volume_spike": volume_spike,
    "momentum_shift": momentum_shift,
    "volatility_expansion": volatility_expansion,
    "trend_pullback_resume": trend_pullback_resume,
}


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """One column per pattern. `df` must carry indicators from add_indicators."""
    return pd.DataFrame({name: fn(df).fillna(0).astype(int) for name, fn in DETECTORS.items()},
                        index=df.index)


def market_context(df: pd.DataFrame) -> pd.DataFrame:
    """Conditions in force at each bar's close: trend, trend strength, volatility regime, prior direction."""
    atr_ratio = df["atr14"] / df["atr14"].rolling(50).mean().shift(1)
    return pd.DataFrame({
        "trend": np.sign(df["ema20"] - df["ema50"]).map({1.0: "up", -1.0: "down", 0.0: "flat"}),
        "strength": np.where(df["adx14"] >= 25, "trending", "ranging"),
        "vol_regime": pd.cut(atr_ratio, [-np.inf, 0.8, 1.25, np.inf], labels=["low", "normal", "high"]).astype(object),
        "prev_dir": np.sign(df["close"] - df["close"].shift(10)).map({1.0: "up", -1.0: "down", 0.0: "flat"}),
    }, index=df.index)
