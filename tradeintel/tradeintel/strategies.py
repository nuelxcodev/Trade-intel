"""Rule-based baseline strategies. A strategy is a function (df, i) -> "LONG" | "SHORT" | None
that may only read rows <= i of an indicator-augmented frame."""

from __future__ import annotations

import pandas as pd


def ema_pullback(df: pd.DataFrame, i: int) -> str | None:
    """Trade pullbacks inside an established trend, only when trend strength is adequate."""
    if i < 60:
        return None
    r, p = df.iloc[i], df.iloc[i - 1]
    if pd.isna(r["adx14"]) or pd.isna(r["ema50"]) or pd.isna(r["rsi14"]):
        return None
    if r["adx14"] < 20:
        return None                                   # ranging market: unsuitable
    if r["ema20"] > r["ema50"] and r["close"] > r["ema50"] and 40 <= p["rsi14"] <= 55 and r["rsi14"] > p["rsi14"]:
        return "LONG"
    if r["ema20"] < r["ema50"] and r["close"] < r["ema50"] and 45 <= p["rsi14"] <= 60 and r["rsi14"] < p["rsi14"]:
        return "SHORT"
    return None
