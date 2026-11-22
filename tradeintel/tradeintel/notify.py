"""Notifications: in-app, Telegram and email.

Messages are rendered from the persisted signal row, never from live recomputation, so what a
user is told is exactly what is stored and traceable. Every channel send is recorded once per
(signal, user, channel): a restart or a repeated scan can never double-send. A failing channel
is retried with backoff and never blocks the others. Secrets come from the environment.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import time
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Callable

import requests

from .signals import SignalStore
from .store import CandleStore

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_prefs (user_id TEXT PRIMARY KEY, prefs TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, signal_id TEXT NOT NULL,
    title TEXT NOT NULL, body TEXT NOT NULL, created_at INTEGER NOT NULL, read INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS delivery_log (
    signal_id TEXT NOT NULL, user_id TEXT NOT NULL, channel TEXT NOT NULL,
    status TEXT NOT NULL, attempts INTEGER NOT NULL, detail TEXT, at INTEGER NOT NULL,
    PRIMARY KEY (signal_id, user_id, channel)
);
"""


@dataclass
class Prefs:
    min_score: float = 60.0
    timeframes: list[str] = field(default_factory=list)      # empty = all
    channels: list[str] = field(default_factory=lambda: ["in_app"])
    min_probability: float = 0.0
    telegram_chat_id: str | None = None
    email: str | None = None

    def wants(self, sig: dict) -> bool:
        if sig["score"] < self.min_score:
            return False
        if self.timeframes and sig["tf"] not in self.timeframes:
            return False
        return (sig.get("probability") or 0.0) >= self.min_probability


def _fmt(x: float) -> str:
    return f"{x:,.6g}" if abs(x) < 1 else f"{x:,.2f}" if abs(x) < 1000 else f"{x:,.0f}"


def format_signal(sig: dict) -> tuple[str, str]:
    """Title and body from a persisted signal. Every number comes from the stored row."""
    pl, ev = sig["payload"], sig["payload"].get("evidence", {})
    arrow = "LONG" if sig["direction"] == "LONG" else "SHORT"
    title = f"{arrow} {sig['symbol']} ({sig['tf']}) score {sig['score']:.0f}"
    comps = pl.get("components", {})
    top = ", ".join(k.replace("_", "/") for k, _ in sorted(comps.items(), key=lambda kv: -kv[1])[:3])
    nb = ev.get("neighbours", {})
    d = "long" if sig["direction"] == "LONG" else "short"
    lines = [
        f"{arrow} {sig['symbol']} on the {sig['tf']} chart",
        f"Entry {_fmt(sig['entry'])} (zone {_fmt(sig['entry_lo'])} to {_fmt(sig['entry_hi'])})",
        f"Stop loss {_fmt(sig['sl'])}",
        f"Take profit {_fmt(sig['tp1'])} / {_fmt(sig['tp2'])} / {_fmt(sig['tp3'])}",
        f"Risk/reward to TP2: 1 : {sig['rr']:.2f}",
        f"Size {sig['qty']:.6g} units, risking {_fmt(sig['risk_amount'])}",
        f"Score {sig['score']:.0f}/100. Model probability {sig['probability']:.0%} (calibrated estimate, not a guarantee).",
    ]
    if top:
        lines.append(f"Strongest factors: {top}.")
    if ev.get("patterns_now"):
        lines.append("Patterns on the signal candle: " + ", ".join(sorted(ev["patterns_now"])) + ".")
    if nb:
        lines.append(f"{nb.get('k', 0)} similar past setups: {nb.get(d + '_success', 0):.0%} reached the target "
                     f"(all-bar base rate {ev.get('base_rate', 0):.0%}).")
    lines.append(f"Wait for price to enter the zone within {sig['expiry_bars']} candles; do not chase.")
    if pl.get("invalidation"):
        lines.append("Abandon if: " + "; ".join(pl["invalidation"]) + ".")
    lines.append(f"Model {sig['model_id']}. Signal {sig['id']}. Decision support, not financial advice.")
    return title, "\n".join(lines)


# ---------------- channels ----------------

class Channel:
    name = ""

    def send(self, user_id: str, prefs: Prefs, title: str, body: str, signal_id: str) -> None:
        raise NotImplementedError


class InAppChannel(Channel):
    name = "in_app"

    def __init__(self, store: CandleStore):
        self.conn = store.conn

    def send(self, user_id, prefs, title, body, signal_id):
        with self.conn:
            self.conn.execute("INSERT INTO notifications (user_id, signal_id, title, body, created_at) "
                              "VALUES (?,?,?,?,?)", (user_id, signal_id, title, body, int(time.time())))


class TelegramChannel(Channel):
    name = "telegram"

    def __init__(self, token: str | None = None, session: requests.Session | None = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.session = session or requests.Session()

    def send(self, user_id, prefs, title, body, signal_id):
        if not self.token or not prefs.telegram_chat_id:
            raise RuntimeError("telegram is not configured (bot token or chat id missing)")
        resp = self.session.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                                 json={"chat_id": prefs.telegram_chat_id, "text": f"{title}\n\n{body}"}, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"telegram HTTP {resp.status_code}")


class EmailChannel(Channel):
    name = "email"

    def __init__(self, host: str | None = None, port: int | None = None, user: str | None = None,
                 password: str | None = None, sender: str | None = None,
                 smtp_factory: Callable[..., smtplib.SMTP] = smtplib.SMTP):
        self.host = host or os.environ.get("SMTP_HOST")
        self.port = port or int(os.environ.get("SMTP_PORT", "587"))
        self.user, self.password = user or os.environ.get("SMTP_USER"), password or os.environ.get("SMTP_PASSWORD")
        self.sender = sender or os.environ.get("SMTP_FROM") or self.user
        self.smtp_factory = smtp_factory

    def send(self, user_id, prefs, title, body, signal_id):
        if not (self.host and self.sender and prefs.email):
            raise RuntimeError("email is not configured (SMTP host, sender or recipient missing)")
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = title, self.sender, prefs.email
        msg.set_content(body)
        with self.smtp_factory(self.host, self.port, timeout=20) as smtp:
            smtp.starttls()
            if self.user:
                smtp.login(self.user, self.password or "")
            smtp.send_message(msg)


# ---------------- dispatcher ----------------

class Notifier:
    def __init__(self, store: CandleStore, signals: SignalStore, channels: list[Channel] | None = None,
                 retries: int = 3, backoff: float = 1.0, sleep: Callable[[float], None] = time.sleep):
        self.conn = store.conn
        self.conn.executescript(SCHEMA)
        self.signals = signals
        self.channels = {c.name: c for c in (channels or [InAppChannel(store)])}
        self.retries, self.backoff, self.sleep = retries, backoff, sleep

    def set_prefs(self, user_id: str, prefs: Prefs) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO user_prefs VALUES (?,?) ON CONFLICT(user_id) DO UPDATE SET prefs=excluded.prefs",
                              (user_id, json.dumps(prefs.__dict__)))

    def get_prefs(self, user_id: str) -> Prefs:
        r = self.conn.execute("SELECT prefs FROM user_prefs WHERE user_id=?", (user_id,)).fetchone()
        return Prefs(**json.loads(r[0])) if r else Prefs()

    def _users(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT user_id FROM user_prefs")]

    def _already(self, sid: str, user: str, channel: str) -> bool:
        r = self.conn.execute("SELECT status FROM delivery_log WHERE signal_id=? AND user_id=? AND channel=?",
                              (sid, user, channel)).fetchone()
        return bool(r and r[0] == "sent")

    def _record(self, sid, user, channel, status, attempts, detail):
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO delivery_log VALUES (?,?,?,?,?,?,?)",
                              (sid, user, channel, status, attempts, detail, int(time.time())))

    def notify_signal(self, signal_id: str) -> dict[str, dict[str, str]]:
        sig = self.signals.get(signal_id)
        sig["id"] = signal_id
        title, body = format_signal(sig)
        result: dict[str, dict[str, str]] = {}
        for user in self._users():
            prefs = self.get_prefs(user)
            if not prefs.wants(sig):
                continue
            for ch_name in prefs.channels:
                ch = self.channels.get(ch_name)
                if ch is None or self._already(signal_id, user, ch_name):
                    continue
                err = None
                for attempt in range(1, self.retries + 1):
                    try:
                        ch.send(user, prefs, title, body, signal_id)
                        err = None
                        break
                    except Exception as exc:                  # one bad channel must not block the rest
                        err = str(exc)
                        log.warning("%s to %s failed (attempt %d): %s", ch_name, user, attempt, err)
                        if attempt < self.retries:
                            self.sleep(self.backoff * 2 ** (attempt - 1))
                status = "sent" if err is None else "failed"
                self._record(signal_id, user, ch_name, status, attempt, err)
                result.setdefault(user, {})[ch_name] = status
        return result

    def notify_new(self, signal_ids: list[str]) -> dict[str, dict]:
        return {sid: self.notify_signal(sid) for sid in signal_ids}

    def inbox(self, user_id: str, unread_only: bool = False) -> list[dict]:
        q = "SELECT id, signal_id, title, body, created_at, read FROM notifications WHERE user_id=?"
        if unread_only:
            q += " AND read=0"
        cur = self.conn.execute(q + " ORDER BY id DESC", (user_id,))
        return [dict(zip(("id", "signal_id", "title", "body", "created_at", "read"), r)) for r in cur]
