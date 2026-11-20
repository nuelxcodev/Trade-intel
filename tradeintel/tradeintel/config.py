"""Central settings. Every threshold the spec calls 'configurable' lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    db_path: str = field(default_factory=lambda: os.environ.get("TI_DB_PATH", "tradeintel.db"))
    exchange: str = "binance"
    quote_asset: str = "USDT"

    # Risk and execution assumptions (used by the trade-parameter engine and backtester).
    capital: float = 10_000.0
    risk_per_trade: float = 0.01      # fraction of equity risked per trade
    fee_rate: float = 0.001           # per side; Binance spot taker default
    slippage: float = 0.0005          # fraction of price, applied to market fills and stops
    min_rr: float = 1.8               # minimum reward/risk to TP2 for a signal to qualify
    max_hold_bars: int = 48           # signal expiry for backtests

    # Opportunity quality gates and scanning.
    min_score: float = 60.0                   # 0-100 composite score needed to be shown
    min_quote_volume_24h: float = 5_000_000.0 # liquidity floor, in quote currency
    expiry_bars: int = 8                      # bars a signal may wait for its entry zone
    hold_bars: int = 24                       # expected holding period, in bars
    scan_neighbours: int = 50                 # similar historical bars compared per pair
    min_neighbour_r: float = 0.0              # neighbours' mean net R must exceed this

    # Data sync.
    kline_batch: int = 1000


DEFAULT = Settings()
