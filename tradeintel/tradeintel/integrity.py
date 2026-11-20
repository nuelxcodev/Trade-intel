"""Detect missing or inconsistent historical candles."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Gap:
    start_ms: int      # open_time of the first missing candle
    end_ms: int        # open_time of the last missing candle
    missing: int       # number of missing candles


def find_gaps(open_times: list[int], step_ms: int) -> list[Gap]:
    """Gaps between consecutive stored candles. Input must be sorted ascending."""
    gaps = []
    for prev, cur in zip(open_times, open_times[1:]):
        diff = cur - prev
        if diff > step_ms:
            missing = diff // step_ms - 1
            gaps.append(Gap(prev + step_ms, cur - step_ms, int(missing)))
    return gaps


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Rows violating basic OHLCV invariants, with a `problem` column."""
    problems = []
    checks = {
        "non-positive price": (df[["open", "high", "low", "close"]] <= 0).any(axis=1),
        "high below open/close": df["high"] < df[["open", "close"]].max(axis=1),
        "low above open/close": df["low"] > df[["open", "close"]].min(axis=1),
        "negative volume": df["volume"] < 0,
        "duplicate open_time": df["open_time"].duplicated(keep=False),
    }
    for name, mask in checks.items():
        bad = df[mask].copy()
        if not bad.empty:
            bad["problem"] = name
            problems.append(bad)
    return pd.concat(problems) if problems else df.iloc[0:0].assign(problem=pd.Series(dtype=str))
