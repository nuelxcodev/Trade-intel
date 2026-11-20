"""Historical knowledge base: every pattern occurrence with its measured outcome, plus honest statistics.

Statistics are reported with Wilson confidence intervals and compared with the unconditional
base rate. A success rate is a description of the past sample, not a probability of the next trade.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .outcomes import OutcomeSpec, baseline_rates, measure
from .patterns import detect_all, market_context
from .store import CandleStore

EVENT_COLUMNS = ["symbol", "tf", "pattern", "direction", "bar_index", "bar_time", "trend", "strength",
                 "vol_regime", "prev_dir", "mfe_atr", "mae_atr", "bars", "outcome", "success"]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def build_events(df: pd.DataFrame, symbol: str, tf: str, spec: OutcomeSpec = OutcomeSpec()) -> pd.DataFrame:
    """All pattern events in `df` (indicator-augmented) with resolved outcomes."""
    signals, ctx = detect_all(df), market_context(df)
    rows = []
    for name in signals.columns:
        for i in np.flatnonzero(signals[name].to_numpy() != 0):
            if i < 60:
                continue
            d = int(signals[name].iat[i])
            res = measure(df, int(i), d, spec)
            if res is None:
                continue
            rows.append({"symbol": symbol, "tf": tf, "pattern": name, "direction": d, "bar_index": int(i),
                         "bar_time": df["open_time"].iat[i], **ctx.iloc[i].to_dict(), **res})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def summarize(events: pd.DataFrame, baseline: dict[int, float] | None = None,
              by: tuple[str, ...] = ("pattern", "direction"), total_bars: int | None = None) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    out = []
    for key, g in events.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        n, wins = len(g), int(g["success"].sum())
        lo, hi = wilson(wins, n)
        row = dict(zip(by, key))
        row.update(n=n, success_rate=wins / n, ci_low=lo, ci_high=hi,
                   avg_mfe_atr=g["mfe_atr"].mean(), avg_mae_atr=g["mae_atr"].mean(),
                   median_bars=float(g["bars"].median()))
        if total_bars:
            row["per_1000_bars"] = 1000 * n / total_bars
        if baseline and "direction" in by:
            row["baseline"] = baseline[row["direction"]]
            row["lift"] = row["success_rate"] - row["baseline"]
        out.append(row)
    return pd.DataFrame(out).sort_values("n", ascending=False).reset_index(drop=True)


def validate_chronologically(events: pd.DataFrame, baseline: dict[int, float], split: float = 0.7,
                             min_test_n: int = 30) -> pd.DataFrame:
    """Learn on the earlier part of history, check on the later part.

    A pattern 'holds' only if its test-period success rate beats the base rate AND the test
    sample is large enough; discoveries that vanish out of sample are flagged, not hidden.
    """
    if events.empty:
        return pd.DataFrame()
    cut = events["bar_index"].quantile(split)
    train, test = events[events["bar_index"] <= cut], events[events["bar_index"] > cut]
    rows = []
    for (pat, d), g in events.groupby(["pattern", "direction"]):
        tr = train[(train["pattern"] == pat) & (train["direction"] == d)]
        te = test[(test["pattern"] == pat) & (test["direction"] == d)]
        tr_rate = tr["success"].mean() if len(tr) else float("nan")
        te_rate = te["success"].mean() if len(te) else float("nan")
        lo, _ = wilson(int(te["success"].sum()), len(te))
        base = baseline[d]
        rows.append({"pattern": pat, "direction": d, "n_train": len(tr), "n_test": len(te),
                     "train_rate": tr_rate, "test_rate": te_rate, "baseline": base,
                     "train_lift": tr_rate - base, "test_lift": te_rate - base,
                     "holds": bool(len(te) >= min_test_n and tr_rate > base and lo > base)})
    return pd.DataFrame(rows).sort_values("n_test", ascending=False).reset_index(drop=True)


class KnowledgeBase:
    """Persists events and summaries next to the candles."""

    def __init__(self, store: CandleStore):
        self.conn = store.conn

    def save_events(self, events: pd.DataFrame, symbol: str, tf: str) -> int:
        self.conn.execute("CREATE TABLE IF NOT EXISTS pattern_events (" + ",".join(
            f"{c} TEXT" if c in ("symbol", "tf", "pattern", "trend", "strength", "vol_regime", "prev_dir", "outcome")
            else f"{c} REAL" for c in EVENT_COLUMNS) + ")")
        with self.conn:
            self.conn.execute("DELETE FROM pattern_events WHERE symbol=? AND tf=?", (symbol, tf))
            if not events.empty:
                events.to_sql("pattern_events", self.conn, if_exists="append", index=False)
        return len(events)

    def events(self, symbol: str | None = None, tf: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM pattern_events WHERE 1=1", []
        if symbol:
            q += " AND symbol=?"; args.append(symbol)
        if tf:
            q += " AND tf=?"; args.append(tf)
        return pd.read_sql_query(q, self.conn, params=args)
