"""Find past bars whose market state resembles the current one, and report what followed.

The index is built only from bars before a cutoff whose outcomes are fully resolved, and the
scaler is fitted on those bars alone, so a query never sees its own future.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .features import Dataset

# Continuous state features only; pattern flags are sparse and dominate distances if included.
STATE_FEATURES = ["ret_3_atr", "ret_10_atr", "ret_30_atr", "dist_ema20_atr", "dist_ema50_atr",
                  "ema_spread_atr", "rsi", "adx", "atr_ratio", "rel_vol", "dist_hi20_atr", "dist_lo20_atr"]


@dataclass
class Neighbours:
    k: int
    mean_distance: float
    bar_index: list[int]
    long_success: float
    short_success: float
    long_mean_r: float
    short_mean_r: float


class SimilarityIndex:
    def __init__(self, ds: Dataset, cutoff_bar: int | None = None, min_gap: int | None = None):
        """Index bars with bar_index < cutoff_bar - gap (gap defaults to horizon+1)."""
        gap = ds.spec.horizon + 1 if min_gap is None else min_gap
        X = ds.X if cutoff_bar is None else ds.X[ds.X.index < cutoff_bar - gap]
        if len(X) < 50:
            raise ValueError("not enough resolved history to build a similarity index")
        self.rows = X.index.to_numpy()
        self.scaler = StandardScaler().fit(X[STATE_FEATURES])
        self.nn = NearestNeighbors(metric="euclidean").fit(self.scaler.transform(X[STATE_FEATURES]))
        self.y = {d: ds.y[d].loc[X.index].to_numpy() for d in (1, -1)}
        self.r = {d: ds.r[d].loc[X.index].to_numpy() for d in (1, -1)}

    def query(self, features_row: pd.DataFrame, k: int = 50) -> Neighbours:
        z = self.scaler.transform(features_row[STATE_FEATURES])
        dist, idx = self.nn.kneighbors(z, n_neighbors=min(k, len(self.rows)))
        idx = idx[0]
        return Neighbours(
            k=len(idx), mean_distance=float(dist[0].mean()), bar_index=[int(b) for b in self.rows[idx]],
            long_success=float(self.y[1][idx].mean()), short_success=float(self.y[-1][idx].mean()),
            long_mean_r=float(self.r[1][idx].mean()), short_mean_r=float(self.r[-1][idx].mean()))
