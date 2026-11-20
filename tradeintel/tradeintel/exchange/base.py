"""Exchange abstraction. Adding another exchange or market means adding one adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Candle:
    open_time: int  # ms since epoch, start of the candle
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0


class Exchange(ABC):
    name: str

    @abstractmethod
    def list_symbols(self, quote: str) -> list[str]:
        """Tradable spot symbols quoted in `quote`."""

    @abstractmethod
    def fetch_klines(self, symbol: str, tf: str, start_ms: int, limit: int) -> list[Candle]:
        """Candles with open_time >= start_ms, oldest first, at most `limit`.

        May include the still-forming candle; the sync layer discards it.
        """

    @abstractmethod
    def server_time_ms(self) -> int:
        """Exchange clock, used to decide which candles are closed."""
