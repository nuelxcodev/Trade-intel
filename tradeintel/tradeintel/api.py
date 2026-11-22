"""Small JSON API for the dashboard, built on the standard library (no web framework needed).

    GET  /health                        liveness and worker status, never needs a key
    GET  /api/overview                  pairs, timeframes, signal counts, model status
    GET  /api/opportunities?tf=1h       latest scan: ranked opportunities or the explicit "none" message
    GET  /api/signals?status=open|all   signals with their current status
    GET  /api/signals/<id>              one signal with its full history
    GET  /api/performance               win rate, profit factor, drawdown and breakdowns
    GET  /api/models                    trained models and whether each showed an edge on its test window
    GET  /api/notifications?user=...    in-app inbox
    POST /api/prefs/<user>              notification thresholds (always requires the API key)

If TI_API_KEY is set, every /api route needs it as `X-API-Key` or `Authorization: Bearer`.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .notify import Notifier, Prefs
from .performance import build_report
from .service import STATE, model_dir_for
from .signals import SignalStore
from .store import CandleStore


def clean(o):
    """JSON-safe copy: numpy to Python, NaN and infinity to null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


class Config:
    def __init__(self, db_path: str, model_dir: str | None = None, api_key: str | None = None,
                 cors_origin: str = "*", exchange: str = "binance", symbols: list[str] | None = None,
                 tfs: list[str] | None = None):
        self.db_path, self.model_dir = db_path, model_dir or model_dir_for(db_path)
        self.api_key, self.cors_origin, self.exchange = api_key, cors_origin, exchange
        self.symbols, self.tfs = symbols or [], tfs or []


def make_handler(cfg: Config):
    class Handler(BaseHTTPRequestHandler):
        server_version = "tradeintel"

        def log_message(self, fmt, *args):  # quiet: the host collects stdout
            pass

        def _send(self, code: int, payload) -> None:
            body = json.dumps(clean(payload), allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", cfg.cors_origin)
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self) -> bool:
            if not cfg.api_key:
                return True
            given = self.headers.get("X-API-Key") or self.headers.get("Authorization", "").removeprefix("Bearer ")
            return hmac.compare_digest(given.encode(), cfg.api_key.encode())

        def do_OPTIONS(self):
            self._send(204, {})

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/health":
                return self._send(200, health())
            if not url.path.startswith("/api/"):
                return self._send(404, {"error": "not found"})
            if not self._authorised():
                return self._send(401, {"error": "missing or invalid API key"})
            store = CandleStore(cfg.db_path)
            try:
                code, payload = route_get(cfg, store, url.path, q)
            except Exception as exc:
                code, payload = 500, {"error": str(exc)}
            finally:
                store.close()
            self._send(code, payload)

        def do_POST(self):
            url = urlparse(self.path)
            if not cfg.api_key:
                return self._send(403, {"error": "set TI_API_KEY to enable write endpoints"})
            if not self._authorised():
                return self._send(401, {"error": "missing or invalid API key"})
            parts = url.path.strip("/").split("/")
            if len(parts) != 3 or parts[:2] != ["api", "prefs"]:
                return self._send(404, {"error": "not found"})
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                allowed = set(Prefs.__dataclass_fields__)
                unknown = set(data) - allowed
                if unknown:
                    return self._send(400, {"error": f"unknown fields {sorted(unknown)}"})
                prefs = Prefs(**data)
                if prefs.channels and not set(prefs.channels) <= {"in_app", "telegram", "email"}:
                    return self._send(400, {"error": "channels must be in_app, telegram or email"})
            except (ValueError, TypeError) as exc:
                return self._send(400, {"error": str(exc)})
            store = CandleStore(cfg.db_path)
            try:
                Notifier(store, SignalStore(store), []).set_prefs(parts[2], prefs)
            finally:
                store.close()
            self._send(200, {"user": parts[2], "prefs": prefs.__dict__})

    return Handler


def health() -> dict:
    last = STATE["last_cycle_at"]
    return {"status": "ok", "uptime_seconds": int(time.time() - STATE["started_at"]), "cycles": STATE["cycles"],
            "last_cycle_seconds_ago": None if last is None else int(time.time() - last),
            "last_error": STATE["last_error"]}


def _signal_row(ss: SignalStore, sid: str) -> dict:
    sig = ss.get(sid)
    sig["id"] = sid
    sig["state"] = ss.state(sid)
    return sig


def route_get(cfg: Config, store: CandleStore, path: str, q: dict) -> tuple[int, dict]:
    ss = SignalStore(store)
    if path == "/api/overview":
        models = list_models(cfg)
        return 200, {"exchange": cfg.exchange, "symbols": cfg.symbols, "timeframes": cfg.tfs,
                     "monitored_pairs": len(cfg.symbols), "open_signals": len(ss.open_signals()),
                     "models": len(models), "models_with_edge": sum(any(m["usable"].values()) for m in models),
                     "worker": health()}
    if path == "/api/opportunities":
        tf = q.get("tf") or (cfg.tfs[0] if cfg.tfs else "1h")
        row = store.conn.execute("SELECT report FROM scan_results WHERE tf=?", (tf,)).fetchone() \
            if store.conn.execute("SELECT name FROM sqlite_master WHERE name='scan_results'").fetchone() else None
        if row is None:
            return 200, {"tf": tf, "message": "No scan has completed yet", "opportunities": [], "rejected": {}}
        return 200, json.loads(row[0])
    if path == "/api/signals":
        ids = [r[0] for r in store.conn.execute("SELECT id FROM signals ORDER BY created_at DESC")]
        if q.get("status", "all") == "open":
            open_ids = set(ss.open_signals())
            ids = [i for i in ids if i in open_ids]
        return 200, {"signals": [_signal_row(ss, i) for i in ids[: int(q.get("limit", 100))]]}
    if path.startswith("/api/signals/"):
        sid = path.rsplit("/", 1)[1]
        try:
            return 200, {**_signal_row(ss, sid), "history": ss.history(sid)}
        except KeyError:
            return 404, {"error": "signal not found"}
    if path == "/api/performance":
        return 200, build_report(ss, candles=store, exchange=cfg.exchange)
    if path == "/api/models":
        return 200, {"models": list_models(cfg)}
    if path == "/api/notifications":
        user = q.get("user", "default")
        return 200, {"user": user, "notifications": Notifier(store, ss, []).inbox(user, q.get("unread") == "1")}
    return 404, {"error": "not found"}


def list_models(cfg: Config) -> list[dict]:
    out = []
    for p in sorted(Path(cfg.model_dir).glob("*/meta.json")) if Path(cfg.model_dir).exists() else []:
        m = json.loads(p.read_text())
        out.append({"model_id": m["model_id"], "symbol": m["symbol"], "tf": m["tf"], "created_at": m["created_at"],
                    "trained_through_open_time": m["trained_through_open_time"],
                    "usable": {k: v["usable"] for k, v in m["models"].items()},
                    "test": {k: {x: v["metrics"].get(x) for x in ("auc_test", "signals_test", "mean_r_signals_test",
                                                                  "win_rate_signals_test", "ece_test")}
                             for k, v in m["models"].items()}})
    return out


def make_server(cfg: Config, host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(cfg))


def config_from_env() -> Config:
    from .service import DEFAULT_SYMBOLS, env_list, env_settings
    s = env_settings()
    return Config(s.db_path, api_key=os.environ.get("TI_API_KEY") or None,
                  cors_origin=os.environ.get("TI_CORS_ORIGIN", "*"),
                  symbols=env_list("TI_SYMBOLS", DEFAULT_SYMBOLS), tfs=env_list("TI_TIMEFRAMES", "1h"))
