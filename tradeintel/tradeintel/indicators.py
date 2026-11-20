"""Technical indicators. Every function is causal: the value at bar t uses only bars <= t."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"],
                      (df["high"] - prev_close).abs(),
                      (df["low"] - prev_close).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    a = atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["rsi14"] = rsi(out["close"], 14)
    out["atr14"] = atr(out, 14)
    out["adx14"] = adx(out, 14)
    out["vol_ma20"] = out["volume"].rolling(20, min_periods=20).mean()
    out["rel_vol"] = out["volume"] / out["vol_ma20"]
    return out


def swing_levels(df: pd.DataFrame, i: int, side: int = 3, lookback: int = 120) -> tuple[list[float], list[float]]:
    """Confirmed swing highs and lows visible at bar i.

    A pivot at bar p needs `side` bars on each side, so it is only confirmed at bar p+side.
    Only pivots with p + side <= i are returned, which keeps the result free of look-ahead.
    """
    lo = max(side, i - lookback)
    highs, lows = [], []
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    for p in range(lo, i - side + 1):
        if h[p] == h[p - side:p + side + 1].max() and h[p] > h[p - 1]:
            highs.append(float(h[p]))
        if l[p] == l[p - side:p + side + 1].min() and l[p] < l[p - 1]:
            lows.append(float(l[p]))
    return highs, lows
