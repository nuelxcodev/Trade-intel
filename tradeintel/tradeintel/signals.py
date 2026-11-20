"""Signal lifecycle.

A signal's original recommendation is immutable (database triggers forbid UPDATE and DELETE).
Everything that happens afterwards is appended to signal_events, so the full history can always
be replayed. Status changes are driven only by CLOSED candles and are idempotent: processing the
same candle twice changes nothing.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import Enum

from .store import CandleStore
from .timeframes import tf_ms


class Status(str, Enum):
    ANALYSING = "ANALYSING"
    WAITING = "WAITING FOR ENTRY"
    CONFIRMED = "ENTRY CONFIRMED"
    ACTIVE = "ACTIVE"
    TP1 = "TP1 HIT"
    TP2 = "TP2 HIT"
    TP3 = "TP3 HIT"
    STOP = "STOP LOSS HIT"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


TERMINAL = {Status.TP3, Status.STOP, Status.INVALIDATED, Status.EXPIRED}
_TP = {1: Status.TP1, 2: Status.TP2, 3: Status.TP3}

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY, created_at INTEGER NOT NULL, bar_open_time INTEGER NOT NULL,
    symbol TEXT NOT NULL, tf TEXT NOT NULL, direction TEXT NOT NULL,
    entry REAL NOT NULL, entry_lo REAL NOT NULL, entry_hi REAL NOT NULL, sl REAL NOT NULL,
    tp1 REAL NOT NULL, tp2 REAL NOT NULL, tp3 REAL NOT NULL, rr REAL NOT NULL,
    qty REAL NOT NULL, risk_amount REAL NOT NULL, score REAL NOT NULL, probability REAL,
    model_id TEXT NOT NULL, expiry_bars INTEGER NOT NULL, hold_bars INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signal_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id TEXT NOT NULL, bar_open_time INTEGER NOT NULL,
    status TEXT NOT NULL, price REAL, note TEXT, recorded_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS signal_state (
    signal_id TEXT PRIMARY KEY, status TEXT NOT NULL, tp_hit INTEGER NOT NULL,
    bars_waiting INTEGER NOT NULL, bars_active INTEGER NOT NULL, last_bar INTEGER NOT NULL
);
CREATE TRIGGER IF NOT EXISTS signals_no_update BEFORE UPDATE ON signals
    BEGIN SELECT RAISE(ABORT, 'signals are immutable'); END;
CREATE TRIGGER IF NOT EXISTS signals_no_delete BEFORE DELETE ON signals
    BEGIN SELECT RAISE(ABORT, 'signals are immutable'); END;
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON signal_events
    BEGIN SELECT RAISE(ABORT, 'signal history is append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON signal_events
    BEGIN SELECT RAISE(ABORT, 'signal history is append-only'); END;
"""


def advance(sig: dict, st: dict, c: dict) -> tuple[list[dict], dict]:
    """Apply one closed candle to a signal. Returns (new events, new state). Pure function.

    Conservative rules: if one candle touches both the stop and a target, the stop is assumed
    first; after TP1 the stop is treated as moved to entry (as in the backtester).
    """
    status = Status(st["status"])
    if status in TERMINAL:
        return [], st
    st, events = dict(st), []
    s = 1 if sig["direction"] == "LONG" else -1
    best, worst = (c["high"], c["low"]) if s == 1 else (c["low"], c["high"])

    def emit(new: Status, price: float | None, note: str) -> None:
        events.append({"bar_open_time": c["open_time"], "status": new.value, "price": price, "note": note})
        st["status"] = new.value

    if status in (Status.ANALYSING, Status.WAITING):
        st["bars_waiting"] += 1
        touched = c["low"] <= sig["entry_hi"] and c["high"] >= sig["entry_lo"]
        if s * (worst - sig["sl"]) <= 0:
            emit(Status.INVALIDATED, sig["sl"], "stop level breached before entry")
        elif touched:
            on_side = c["close"] >= sig["entry_lo"] if s == 1 else c["close"] <= sig["entry_hi"]
            if on_side:
                emit(Status.CONFIRMED, c["close"], "entry zone touched and the candle closed on the trade side")
        elif s * (best - sig["tp1"]) >= 0:
            emit(Status.INVALIDATED, sig["tp1"], "first target reached without an entry; the move was missed")
        if Status(st["status"]) in (Status.ANALYSING, Status.WAITING) and st["bars_waiting"] >= sig["expiry_bars"]:
            emit(Status.EXPIRED, c["close"], "signal expired without an entry")
        return events, st

    if status == Status.CONFIRMED:
        emit(Status.ACTIVE, sig["entry"], "position assumed filled at the recommended entry")
        st["bars_active"] = 0

    st["bars_active"] += 1
    stop = sig["sl"] if st["tp_hit"] == 0 else sig["entry"]
    if s * (worst - stop) <= 0:
        emit(Status.STOP, stop, "stop loss" if st["tp_hit"] == 0 else "stop at entry after TP1 (breakeven)")
        return events, st
    for k in (1, 2, 3):
        if k > st["tp_hit"] and s * (best - sig[f"tp{k}"]) >= 0:
            st["tp_hit"] = k
            emit(_TP[k], sig[f"tp{k}"], f"take profit {k} reached")
    if Status(st["status"]) not in TERMINAL and st["bars_active"] >= sig["hold_bars"]:
        emit(Status.EXPIRED, c["close"], "expected holding period elapsed")
    return events, st


class SignalStore:
    def __init__(self, store: CandleStore):
        self.conn = store.conn
        self.conn.executescript(SCHEMA)
        self.conn.row_factory = None

    def _now(self) -> int:
        return int(time.time())

    def create(self, sig: dict) -> str:
        """Persist a new signal. `sig` holds the fields of the signals table plus a free-form `payload`."""
        sid = sig.get("id") or uuid.uuid4().hex[:12]
        cols = ["id", "created_at", "bar_open_time", "symbol", "tf", "direction", "entry", "entry_lo", "entry_hi",
                "sl", "tp1", "tp2", "tp3", "rr", "qty", "risk_amount", "score", "probability", "model_id",
                "expiry_bars", "hold_bars", "payload"]
        row = {**sig, "id": sid, "created_at": self._now(),
               "payload": json.dumps(sig.get("payload", {}), default=float)}
        now = self._now()
        with self.conn:
            self.conn.execute(f"INSERT INTO signals ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                              [row.get(c) for c in cols])
            for status, note in ((Status.ANALYSING, "signal generated by the scanner"),
                                 (Status.WAITING, "waiting for price to reach the entry zone")):
                self.conn.execute("INSERT INTO signal_events (signal_id, bar_open_time, status, price, note, recorded_at)"
                                  " VALUES (?,?,?,?,?,?)", (sid, sig["bar_open_time"], status.value, None, note, now))
            self.conn.execute("INSERT INTO signal_state VALUES (?,?,?,?,?,?)",
                              (sid, Status.WAITING.value, 0, 0, 0, sig["bar_open_time"]))
        return sid

    def get(self, sid: str) -> dict:
        cur = self.conn.execute("SELECT * FROM signals WHERE id=?", (sid,))
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        if row is None:
            raise KeyError(sid)
        out = dict(zip(cols, row))
        out["payload"] = json.loads(out["payload"])
        return out

    def state(self, sid: str) -> dict:
        r = self.conn.execute("SELECT status, tp_hit, bars_waiting, bars_active, last_bar FROM signal_state "
                              "WHERE signal_id=?", (sid,)).fetchone()
        return dict(zip(("status", "tp_hit", "bars_waiting", "bars_active", "last_bar"), r))

    def history(self, sid: str) -> list[dict]:
        cur = self.conn.execute("SELECT bar_open_time, status, price, note FROM signal_events "
                                "WHERE signal_id=? ORDER BY id", (sid,))
        return [dict(zip(("bar_open_time", "status", "price", "note"), r)) for r in cur]

    def open_signals(self) -> list[str]:
        terminal = tuple(s.value for s in TERMINAL)
        q = ",".join("?" * len(terminal))
        return [r[0] for r in self.conn.execute(
            f"SELECT signal_id FROM signal_state WHERE status NOT IN ({q})", terminal)]

    def has_open(self, symbol: str, tf: str, direction: str) -> bool:
        ids = set(self.open_signals())
        return any(r[0] in ids for r in self.conn.execute(
            "SELECT id FROM signals WHERE symbol=? AND tf=? AND direction=?", (symbol, tf, direction)))

    def process(self, sid: str, candles: list[dict]) -> list[dict]:
        """Feed closed candles (any order, duplicates allowed). Returns the events recorded."""
        sig, st, recorded = self.get(sid), self.state(sid), []
        now = self._now()
        for c in sorted(candles, key=lambda c: c["open_time"]):
            if c["open_time"] <= st["last_bar"] or Status(st["status"]) in TERMINAL:
                continue
            events, st = advance(sig, st, c)
            st["last_bar"] = c["open_time"]
            with self.conn:
                for e in events:
                    self.conn.execute("INSERT INTO signal_events (signal_id, bar_open_time, status, price, note,"
                                      " recorded_at) VALUES (?,?,?,?,?,?)",
                                      (sid, e["bar_open_time"], e["status"], e["price"], e["note"], now))
                self.conn.execute("UPDATE signal_state SET status=?, tp_hit=?, bars_waiting=?, bars_active=?,"
                                  " last_bar=? WHERE signal_id=?",
                                  (st["status"], st["tp_hit"], st["bars_waiting"], st["bars_active"],
                                   st["last_bar"], sid))
            recorded += events
        return recorded


def monitor_open_signals(signals: SignalStore, candles: CandleStore, exchange: str) -> dict[str, list[dict]]:
    """Advance every open signal with the closed candles stored since it last moved."""
    out = {}
    for sid in signals.open_signals():
        sig, st = signals.get(sid), signals.state(sid)
        df = candles.load(exchange, sig["symbol"], sig["tf"], start_ms=st["last_bar"] + tf_ms(sig["tf"]))
        rows = df[["open_time", "open", "high", "low", "close"]].to_dict("records")
        if rows:
            events = signals.process(sid, rows)
            if events:
                out[sid] = events
    return out
