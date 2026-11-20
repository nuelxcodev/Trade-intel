"""Opportunity gating and scoring.

Hard gates decide whether something may be shown at all. The score only orders what passed.
The weights are hand-set priors, not fitted values: treat the score as a ranking aid, never
as a probability. The calibrated probability is reported separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT, Settings

WEIGHTS = {"model": 0.30, "similarity": 0.20, "structure": 0.15, "volume": 0.10,
           "volatility": 0.10, "liquidity": 0.05, "risk_reward": 0.10}


@dataclass
class Opportunity:
    symbol: str
    tf: str
    direction: str
    bar_open_time: int
    score: float
    probability: float
    threshold: float
    entry: float
    entry_lo: float
    entry_hi: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    rr: float
    qty: float
    risk_amount: float
    notional: float
    hold_bars: int
    expiry_bars: int
    model_id: str
    components: dict = field(default_factory=dict)
    invalidation: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)       # structured inputs for the AI explainer


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, x)))


def score_components(*, p: float, thr: float, nb_success: float, base_rate: float, trend_aligned: bool,
                     adx: float, rel_vol: float, atr_ratio: float, quote_volume_24h: float,
                     min_quote_volume: float, rr: float, min_rr: float) -> dict[str, float]:
    return {
        "model": _clip((p - thr) / max(0.9 - thr, 1e-9)),
        "similarity": _clip((nb_success - base_rate) / 0.15),
        "structure": _clip(0.5 * trend_aligned + 0.5 * min(adx / 40.0, 1.0)),
        "volume": _clip(rel_vol / 2.0),
        "volatility": _clip(1 - abs(atr_ratio - 1.2) / 1.2),
        "liquidity": _clip(math.log10(max(quote_volume_24h, 1) / max(min_quote_volume, 1)) / 2.0),
        "risk_reward": _clip((rr - min_rr) / 2.0),
    }


def composite(components: dict[str, float], weights: dict[str, float] = WEIGHTS) -> float:
    total = sum(weights.values())
    return 100.0 * sum(weights[k] * components[k] for k in weights) / total


def gate_failures(*, qualifies: bool, usable: bool, rr: float, quote_volume_24h: float, atr_ratio: float,
                  nb_mean_r: float, settings: Settings = DEFAULT) -> list[str]:
    """Reasons an otherwise-detected setup must not be shown. Empty list means it passed."""
    out = []
    if not usable:
        out.append("model has no demonstrated edge on its test window")
    elif not qualifies:
        out.append("model probability is below its validated threshold")
    if rr < settings.min_rr:
        out.append(f"risk/reward {rr:.2f} is below the minimum {settings.min_rr:.2f}")
    if quote_volume_24h < settings.min_quote_volume_24h:
        out.append("24h traded volume is below the liquidity floor")
    if not (0.5 <= atr_ratio <= 3.0):
        out.append("volatility regime is unsuitable (too quiet or too violent)")
    if nb_mean_r <= settings.min_neighbour_r:
        out.append("similar historical conditions did not produce positive net results")
    return out


def rank(opps: list[Opportunity]) -> list[Opportunity]:
    return sorted(opps, key=lambda o: o.score, reverse=True)
