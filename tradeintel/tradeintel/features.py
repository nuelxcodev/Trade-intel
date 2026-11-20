"""Model features and labelled datasets.

Features at bar i use only bars <= i. Labels come from outcomes.measure and look forward from
the open of bar i+1, so any two rows closer than the horizon share future bars. Splits therefore
leave a purge gap of horizon+1 bars between train, validation and test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT, Settings
from .outcomes import OutcomeSpec, measure
from .patterns import detect_all

WARMUP = 60


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal feature matrix aligned with df (indicator-augmented)."""
    atr = df["atr14"]
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    rng = (h - l).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    for k in (1, 3, 10, 30):
        f[f"ret_{k}_atr"] = (c - c.shift(k)) / atr
    f["dist_ema20_atr"] = (c - df["ema20"]) / atr
    f["dist_ema50_atr"] = (c - df["ema50"]) / atr
    f["ema_spread_atr"] = (df["ema20"] - df["ema50"]) / atr
    f["rsi"] = df["rsi14"]
    f["adx"] = df["adx14"]
    f["atr_ratio"] = atr / atr.rolling(50).mean().shift(1)
    f["atr_pct"] = atr / c
    f["rel_vol"] = df["rel_vol"]
    f["range_atr"] = (h - l) / atr
    f["body_atr"] = (c - o) / atr
    f["upper_wick"] = (h - df[["open", "close"]].max(axis=1)) / rng
    f["lower_wick"] = (df[["open", "close"]].min(axis=1) - l) / rng
    f["dist_hi20_atr"] = (h.rolling(20).max() - c) / atr
    f["dist_lo20_atr"] = (c - l.rolling(20).min()) / atr
    for name, col in detect_all(df).items():
        f[f"pat_{name}"] = col
    return f.replace([np.inf, -np.inf], np.nan)


@dataclass
class Dataset:
    X: pd.DataFrame
    y: dict[int, pd.Series]        # direction -> 1 if the trade hit target before stop
    r: dict[int, pd.Series]        # direction -> realised R net of costs
    spec: OutcomeSpec

    @property
    def bar_index(self) -> pd.Index:
        return self.X.index


def build_dataset(df: pd.DataFrame, spec: OutcomeSpec = OutcomeSpec(),
                  settings: Settings = DEFAULT) -> Dataset:
    feats = make_features(df)
    rows, y = [], {1: {}, -1: {}}
    r = {1: {}, -1: {}}
    target_r = spec.target_atr / spec.stop_atr
    atr, close = df["atr14"].to_numpy(), df["close"].to_numpy()
    for i in range(WARMUP, len(df)):
        if feats.iloc[i].isna().any():
            continue
        res = {d: measure(df, i, d, spec) for d in (1, -1)}
        if any(v is None for v in res.values()):
            continue                                   # horizon not fully observed yet
        cost_r = 2 * (settings.fee_rate + settings.slippage) * close[i] / (spec.stop_atr * atr[i])
        rows.append(i)
        for d, v in res.items():
            y[d][i] = v["success"]
            gross = target_r if v["outcome"] == "win" else (-1.0 if v["outcome"] == "loss" else 0.0)
            r[d][i] = gross - cost_r
    X = feats.loc[rows]
    return Dataset(X, {d: pd.Series(y[d]) for d in y}, {d: pd.Series(r[d]) for d in r}, spec)


def purged_split(bar_index: np.ndarray, spec: OutcomeSpec, train_frac: float = 0.6, val_frac: float = 0.2):
    """Boolean masks (train, val, test) over rows ordered by time, with purge gaps between them."""
    bi = np.asarray(bar_index)
    gap = spec.horizon + 1
    val_start = bi[int(len(bi) * train_frac)]
    test_start = bi[int(len(bi) * (train_frac + val_frac))]
    return bi < val_start - gap, (bi >= val_start) & (bi < test_start - gap), bi >= test_start
