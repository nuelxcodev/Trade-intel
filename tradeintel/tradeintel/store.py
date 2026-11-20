"""SQLite candle store.

SQLite keeps the foundation dependency-free. The schema (one row per exchange, symbol,
timeframe, open_time) maps directly onto a TimescaleDB hypertable when the platform scales.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

import pandas as pd

from .exchange.base import Candle

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    exchange     TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    tf           TEXT    NOT NULL,
    open_time    INTEGER NOT NULL,
    open         REAL    NOT NULL,
    high         REAL    NOT NULL,
    low          REAL    NOT NULL,
    close        REAL    NOT NULL,
    volume       REAL    NOT NULL,
    quote_volume REAL    NOT NULL DEFAULT 0,
    trades       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (exchange, symbol, tf, open_time)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS sync_log (
    exchange   TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    tf         TEXT NOT NULL,
    synced_at  INTEGER NOT NULL,
    rows_added INTEGER NOT NULL
);
"""


class CandleStore:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def upsert(self, exchange: str, symbol: str, tf: str, candles: Iterable[Candle]) -> int:
        rows = [(exchange, symbol, tf, c.open_time, c.open, c.high, c.low, c.close,
                 c.volume, c.quote_volume, c.trades) for c in candles]
        if not rows:
            return 0
        before = self.count(exchange, symbol, tf)
        with self.conn:
            self.conn.executemany(
                """INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(exchange, symbol, tf, open_time) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume,
                     quote_volume=excluded.quote_volume, trades=excluded.trades""",
                rows,
            )
        return self.count(exchange, symbol, tf) - before

    def count(self, exchange: str, symbol: str, tf: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM candles WHERE exchange=? AND symbol=? AND tf=?",
            (exchange, symbol, tf)).fetchone()[0]

    def bounds(self, exchange: str, symbol: str, tf: str) -> tuple[int, int] | None:
        row = self.conn.execute(
            "SELECT MIN(open_time), MAX(open_time) FROM candles WHERE exchange=? AND symbol=? AND tf=?",
            (exchange, symbol, tf)).fetchone()
        return None if row[0] is None else (row[0], row[1])

    def open_times(self, exchange: str, symbol: str, tf: str) -> list[int]:
        cur = self.conn.execute(
            "SELECT open_time FROM candles WHERE exchange=? AND symbol=? AND tf=? ORDER BY open_time",
            (exchange, symbol, tf))
        return [r[0] for r in cur]

    def load(self, exchange: str, symbol: str, tf: str,
             start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
        q = ("SELECT open_time, open, high, low, close, volume, quote_volume, trades FROM candles "
             "WHERE exchange=? AND symbol=? AND tf=?")
        args: list = [exchange, symbol, tf]
        if start_ms is not None:
            q += " AND open_time>=?"; args.append(start_ms)
        if end_ms is not None:
            q += " AND open_time<?"; args.append(end_ms)
        q += " ORDER BY open_time"
        df = pd.read_sql_query(q, self.conn, params=args)
        df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df.reset_index(drop=True)

    def log_sync(self, exchange: str, symbol: str, tf: str, synced_at: int, rows_added: int) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO sync_log VALUES (?,?,?,?,?)",
                              (exchange, symbol, tf, synced_at, rows_added))
