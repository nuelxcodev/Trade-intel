"""Synthetic OHLCV for tests and offline demos. Not market data: never use it to judge a strategy."""

from __future__ import annotations

import numpy as np

from .exchange.base import Candle, Exchange
from .timeframes import tf_ms


def make_candles(n: int, tf: str = "1h", start_ms: int = 1_700_000_000_000, seed: int = 7,
                 start_price: float = 100.0, drift: float = 0.0, vol: float = 0.006) -> list[Candle]:
    rng = np.random.default_rng(seed)
    step = tf_ms(tf)
    start_ms -= start_ms % step
    # regime-switching drift so the series contains trends and ranges
    regime = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=n // 80 + 1), 80)[:n]
    rets = drift + regime * 0.0012 + rng.normal(0, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, vol * 0.6, n))
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    volume = rng.lognormal(5, 0.4, n)
    return [Candle(start_ms + k * step, float(open_[k]), float(high[k]), float(low[k]),
                   float(close[k]), float(volume[k]), float(volume[k] * close[k]), 100)
            for k in range(n)]


class FakeExchange(Exchange):
    """In-memory exchange that counts requests, for sync tests and offline runs."""
    name = "fake"

    def __init__(self, candles: dict[tuple[str, str], list[Candle]], now_ms: int):
        self.candles, self.now_ms, self.requests = candles, now_ms, []

    def list_symbols(self, quote: str) -> list[str]:
        return sorted({s for s, _ in self.candles})

    def fetch_klines(self, symbol: str, tf: str, start_ms: int, limit: int) -> list[Candle]:
        self.requests.append((symbol, tf, start_ms))
        rows = [c for c in self.candles.get((symbol, tf), []) if c.open_time >= start_ms]
        return rows[:limit]                  # includes forming candles, like the real API

    def server_time_ms(self) -> int:
        return self.now_ms
