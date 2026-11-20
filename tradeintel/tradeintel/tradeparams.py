"""Entry, stop loss and take profit calculation.

Stops come from market structure (recent swing) plus an ATR buffer. Targets come from
confirmed swing levels, with minimum reward multiples so a target never sits closer than the
risk it is paying for. Nothing here is a fixed percentage of price.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import swing_levels

MIN_R = (1.0, 2.0, 3.0)  # minimum reward multiples for TP1, TP2, TP3


@dataclass
class TradeParams:
    direction: str            # "LONG" or "SHORT"
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    risk_per_unit: float
    rr: float                 # reward/risk to TP2
    notes: list[str] = field(default_factory=list)


def compute_trade_params(df: pd.DataFrame, i: int, direction: str, *, stop_lookback: int = 20,
                         atr_buffer: float = 0.25, min_risk_atr: float = 0.8,
                         max_risk_atr: float = 3.0) -> TradeParams | None:
    """Parameters for a signal raised at the close of bar i. Returns None if the setup is unsuitable."""
    atr = df["atr14"].iloc[i]
    if pd.isna(atr) or atr <= 0 or i < stop_lookback:
        return None
    long = direction == "LONG"
    entry = float(df["close"].iloc[i])
    window = df.iloc[i - stop_lookback:i + 1]
    notes = []

    if long:
        sl = float(window["low"].min()) - atr_buffer * atr
        risk = entry - sl
    else:
        sl = float(window["high"].max()) + atr_buffer * atr
        risk = sl - entry
    notes.append(f"stop beyond the {stop_lookback}-bar swing plus {atr_buffer} ATR buffer")

    if risk < min_risk_atr * atr:                      # too tight: noise would stop it out
        risk = min_risk_atr * atr
        sl = entry - risk if long else entry + risk
        notes.append(f"stop widened to {min_risk_atr} ATR minimum")
    if risk > max_risk_atr * atr:                      # structure too far away: poor setup
        return None

    highs, lows = swing_levels(df, i)
    levels = sorted(h for h in highs if h > entry) if long else sorted((l for l in lows if l < entry), reverse=True)
    targets, prev = [], entry
    for k, r in enumerate(MIN_R, start=1):
        floor = entry + r * risk if long else entry - r * risk
        pick = next((lv for lv in levels
                     if (lv >= floor if long else lv <= floor)
                     and (lv > prev if long else lv < prev)
                     and abs(lv - floor) <= 1.5 * risk), None)
        tp = pick if pick is not None else floor
        # Targets must keep moving away from entry, even when an earlier target snapped to a far swing level.
        step = 0.5 * risk
        tp = max(tp, prev + step) if long else min(tp, prev - step)
        notes.append(f"TP{k} at {'swing level' if pick is not None else f'{r}R projection'}")
        targets.append(tp)
        prev = tp
    rr = abs(targets[1] - entry) / risk
    return TradeParams(direction, entry, sl, targets[0], targets[1], targets[2], risk, rr, notes)


def position_size(capital: float, risk_fraction: float, entry: float, sl: float) -> float:
    """Units to trade so that hitting the stop loses `risk_fraction` of capital (no leverage)."""
    per_unit = abs(entry - sl)
    if per_unit <= 0:
        return 0.0
    return min(capital * risk_fraction / per_unit, capital / entry)
