"""Incremental candle synchronisation.

Rules that keep API usage low and data clean:
  * Resume from the last stored candle; never re-download history already held.
  * Skip the request entirely if the next candle cannot have closed yet.
  * Never store the still-forming candle (its OHLC would change after storage).
  * Repair detected gaps by requesting only the missing ranges.
"""

from __future__ import annotations

import logging
import time

from .config import DEFAULT, Settings
from .exchange.base import Candle, Exchange
from .integrity import find_gaps
from .store import CandleStore
from .timeframes import tf_ms

log = logging.getLogger(__name__)


class Syncer:
    def __init__(self, exchange: Exchange, store: CandleStore, settings: Settings = DEFAULT):
        self.exchange = exchange
        self.store = store
        self.settings = settings

    def _pull(self, symbol: str, tf: str, start_ms: int, stop_ms: int | None, now_ms: int) -> int:
        """Download closed candles from start_ms (up to stop_ms exclusive) and store them."""
        step = tf_ms(tf)
        added = 0
        cursor = start_ms
        while True:
            if cursor + step > now_ms:      # candle at cursor is not closed yet
                break
            if stop_ms is not None and cursor >= stop_ms:
                break
            batch = self.exchange.fetch_klines(symbol, tf, cursor, self.settings.kline_batch)
            batch = [c for c in batch if c.open_time + step <= now_ms]       # closed only
            if stop_ms is not None:
                batch = [c for c in batch if c.open_time < stop_ms]
            if not batch:
                break
            added += self.store.upsert(self.exchange.name, symbol, tf, batch)
            nxt = batch[-1].open_time + step
            if nxt <= cursor:
                break
            cursor = nxt
        return added

    def sync(self, symbol: str, tf: str, since_ms: int, now_ms: int | None = None) -> int:
        """Bring (symbol, tf) up to date. Returns the number of new rows stored."""
        step = tf_ms(tf)
        now = now_ms if now_ms is not None else self.exchange.server_time_ms()
        bounds = self.store.bounds(self.exchange.name, symbol, tf)
        start = since_ms if bounds is None else bounds[1] + step
        start -= start % step               # align to the candle grid
        added = self._pull(symbol, tf, start, None, now)
        added += self.repair_gaps(symbol, tf, now)
        self.store.log_sync(self.exchange.name, symbol, tf, now, added)
        log.info("synced %s %s: +%d rows", symbol, tf, added)
        return added

    def repair_gaps(self, symbol: str, tf: str, now_ms: int) -> int:
        step = tf_ms(tf)
        times = self.store.open_times(self.exchange.name, symbol, tf)
        added = 0
        for gap in find_gaps(times, step):
            log.warning("gap in %s %s: %d candles from %d", symbol, tf, gap.missing, gap.start_ms)
            added += self._pull(symbol, tf, gap.start_ms, gap.end_ms + step, now_ms)
        return added

    def sync_all(self, tfs: list[str], since_ms: int, symbols: list[str] | None = None) -> dict:
        symbols = symbols or self.exchange.list_symbols(self.settings.quote_asset)
        report: dict[tuple[str, str], int | str] = {}
        for sym in symbols:
            for tf in tfs:
                try:
                    report[(sym, tf)] = self.sync(sym, tf, since_ms)
                except Exception as exc:                # one bad pair must not stop the run
                    log.exception("sync failed for %s %s", sym, tf)
                    report[(sym, tf)] = f"error: {exc}"
        return report


def now_ms() -> int:
    return int(time.time() * 1000)
