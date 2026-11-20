"""Measure what price did after each pattern event.

An event at bar i is known at the close of bar i, so the measured move starts from the open of
bar i+1. Moves are expressed in ATR(i) units so events from different pairs and regimes compare.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutcomeSpec:
    horizon: int = 24          # bars to resolve within (the prediction horizon)
    target_atr: float = 1.5    # favourable move that counts as success
    stop_atr: float = 1.0      # adverse move that counts as failure


def measure(df: pd.DataFrame, i: int, direction: int, spec: OutcomeSpec) -> dict | None:
    """Outcome of acting on `direction` (+1/-1) at the close of bar i.

    Returns None when the horizon is not fully observed yet; those events stay out of the
    knowledge base until their outcome is known, so an unfinished move is never mislabelled.
    """
    n = len(df)
    if i + 1 + spec.horizon > n:
        return None
    atr = df["atr14"].iat[i]
    if not np.isfinite(atr) or atr <= 0:
        return None
    entry = df["open"].iat[i + 1]
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    mfe = mae = 0.0
    outcome, bars = "timeout", spec.horizon
    for k in range(spec.horizon):
        j = i + 1 + k
        best, worst = (hi[j], lo[j]) if direction == 1 else (lo[j], hi[j])
        fav = (best - entry) * direction / atr
        adv = (entry - worst) * direction / atr
        mfe, mae = max(mfe, fav), max(mae, adv)
        if adv >= spec.stop_atr:                 # stop first when one bar touches both
            outcome, bars = "loss", k + 1
            break
        if fav >= spec.target_atr:
            outcome, bars = "win", k + 1
            break
    return {"mfe_atr": mfe, "mae_atr": -mae, "bars": bars, "outcome": outcome,
            "success": int(outcome == "win")}


def baseline_rates(df: pd.DataFrame, spec: OutcomeSpec) -> dict[int, float]:
    """Success rate of acting on EVERY bar in each direction. A pattern only has an edge if it beats this."""
    out = {}
    for d in (1, -1):
        res = [measure(df, i, d, spec) for i in range(60, len(df))]
        res = [r for r in res if r]
        out[d] = float(np.mean([r["success"] for r in res])) if res else float("nan")
    return out
