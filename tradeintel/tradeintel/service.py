"""The running system: one cycle of sync, model upkeep, signal monitoring, scanning and notification.

All configuration comes from environment variables so the same code runs locally and on a host:

    TI_DB_PATH            SQLite file (put it on a persistent volume)         default tradeintel.db
    TI_MODEL_DIR          where trained models are saved                      default <db dir>/models
    TI_SYMBOLS            comma separated pairs                               default 8 large USDT pairs
    TI_TIMEFRAMES         comma separated timeframes                          default 1h
    TI_HISTORY_DAYS       history to keep per pair                            default 365
    TI_CYCLE_SECONDS      pause between cycles                                default 300
    TI_BINANCE_URL        market-data base URL                                default data-api.binance.vision
    TI_CAPITAL, TI_RISK_PER_TRADE, TI_MIN_SCORE, TI_MIN_RR, TI_MIN_QUOTE_VOLUME
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import Settings
from .exchange import BinanceSpot
from .exchange.base import Exchange
from .exchange.binance import BASE_URL
from .features import build_dataset
from .indicators import add_indicators
from .ml import needs_retrain, save_models, train_models
from .notify import EmailChannel, InAppChannel, Notifier, Prefs, TelegramChannel
from .outcomes import OutcomeSpec
from .scanner import RegistryProvider, Scanner, ScanReport
from .signals import SignalStore, monitor_open_signals
from .store import CandleStore
from .sync import Syncer
from .timeframes import TF_MS

log = logging.getLogger(__name__)

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT"
MIN_TRAIN_BARS = 1500

# Shared with the API thread: plain values only, replaced atomically.
STATE: dict = {"started_at": time.time(), "cycles": 0, "last_cycle_at": None, "last_error": None,
               "last_summary": None}

SCAN_TABLE = "CREATE TABLE IF NOT EXISTS scan_results (tf TEXT PRIMARY KEY, at INTEGER NOT NULL, report TEXT NOT NULL)"


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not serialisable: {type(o)}")


def env_settings() -> Settings:
    s, env = Settings(), os.environ
    fields = {"TI_CAPITAL": "capital", "TI_RISK_PER_TRADE": "risk_per_trade", "TI_MIN_SCORE": "min_score",
              "TI_MIN_RR": "min_rr", "TI_MIN_QUOTE_VOLUME": "min_quote_volume_24h"}
    return replace(s, **{f: float(env[k]) for k, f in fields.items() if env.get(k)})


def env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


def model_dir_for(db_path: str) -> str:
    return os.environ.get("TI_MODEL_DIR") or str(Path(os.path.abspath(db_path)).parent / "models")


def default_channels(store: CandleStore):
    channels = [InAppChannel(store)]
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        channels.append(TelegramChannel())
    if os.environ.get("SMTP_HOST"):
        channels.append(EmailChannel())
    return channels


def validate_timeframes(tfs: list[str]) -> list[str]:
    bad = [t for t in tfs if t not in TF_MS]
    if bad:
        raise ValueError(f"unsupported timeframes {bad}; choose from {sorted(TF_MS)}")
    return tfs


class Pipeline:
    """Owns its own database connection, so construct and run it from a single thread."""

    def __init__(self, exchange: Exchange, symbols: list[str], tfs: list[str], db_path: str, model_dir: str,
                 settings: Settings, history_days: int = 365, channels=None):
        self.exchange, self.symbols, self.tfs = exchange, symbols, validate_timeframes(tfs)
        self.model_dir, self.settings, self.history_days = model_dir, settings, history_days
        self.store = CandleStore(db_path)
        self.store.conn.execute(SCAN_TABLE)
        self.signals = SignalStore(self.store)
        self.notifier = Notifier(self.store, self.signals, channels or default_channels(self.store))
        if not self.notifier._users():                       # out of the box the in-app inbox works
            self.notifier.set_prefs("default", Prefs(min_score=settings.min_score))
        self.spec = OutcomeSpec()
        self.provider = RegistryProvider(self.store, exchange.name, model_dir)
        self.scanner = Scanner(exchange.name, self.store, self.provider, self.signals, settings)

    # ---- steps ----
    def _sync(self, tf: str) -> dict[str, str]:
        errors = {}
        syncer = Syncer(self.exchange, self.store, self.settings)
        since = self.exchange.server_time_ms() - self.history_days * 86_400_000
        for sym in self.symbols:
            try:
                syncer.sync(sym, tf, since)
            except Exception as exc:                          # one failing pair must not stop the cycle
                log.exception("sync failed for %s %s", sym, tf)
                errors[sym] = str(exc)
        return errors

    def _latest_meta(self, sym: str, tf: str) -> dict | None:
        metas = [json.loads(p.read_text()) for p in Path(self.model_dir).glob(f"{sym}-{tf}-*/meta.json")]
        return max(metas, key=lambda m: m["created_at"]) if metas else None

    def _ensure_model(self, sym: str, tf: str) -> str | None:
        raw = self.store.load(self.exchange.name, sym, tf)
        if len(raw) < MIN_TRAIN_BARS:
            return f"only {len(raw)} candles, need {MIN_TRAIN_BARS} to train"
        meta = self._latest_meta(sym, tf)
        if meta is not None and not needs_retrain(meta, raw):
            return None
        df = add_indicators(raw)
        models = train_models(build_dataset(df, self.spec))
        mid = save_models(models, df, sym, tf, self.spec, self.model_dir)
        log.info("trained %s (long usable=%s, short usable=%s)", mid, models[1].usable, models[-1].usable)
        return None

    def _persist(self, tf: str, rep: ScanReport) -> None:
        report = {"tf": tf, "at": int(time.time()), "scanned": rep.scanned, "from_cache": rep.from_cache,
                  "message": rep.message, "opportunities": [asdict(o) for o in rep.opportunities],
                  "rejected": rep.rejected, "new_signals": rep.new_signals}
        with self.store.conn:
            self.store.conn.execute("INSERT OR REPLACE INTO scan_results VALUES (?,?,?)",
                                    (tf, report["at"], json.dumps(report, default=_json_default)))

    def cycle(self) -> dict:
        summary: dict = {}
        for tf in self.tfs:
            sync_errors = self._sync(tf)
            model_notes = {}
            for sym in self.symbols:
                try:
                    note = self._ensure_model(sym, tf)
                except Exception as exc:
                    log.exception("model upkeep failed for %s %s", sym, tf)
                    note = f"error: {exc}"
                if note:
                    model_notes[sym] = note
            moved = monitor_open_signals(self.signals, self.store, self.exchange.name)
            rep = self.scanner.run_once(self.symbols, tf)
            sent = self.notifier.notify_new(rep.new_signals)
            self._persist(tf, rep)
            summary[tf] = {"sync_errors": sync_errors, "model_notes": model_notes, "signals_moved": len(moved),
                           "opportunities": len(rep.opportunities), "new_signals": len(rep.new_signals),
                           "notifications": sent}
        return summary


def build_pipeline() -> Pipeline:
    settings = env_settings()
    exchange = BinanceSpot(base_url=os.environ.get("TI_BINANCE_URL") or BASE_URL)
    return Pipeline(exchange, env_list("TI_SYMBOLS", DEFAULT_SYMBOLS), env_list("TI_TIMEFRAMES", "1h"),
                    settings.db_path, model_dir_for(settings.db_path), settings,
                    history_days=int(os.environ.get("TI_HISTORY_DAYS", "365")))


def worker_loop(pause: float | None = None, stop=None) -> None:
    """Run cycles forever. Errors are logged and recorded, never fatal."""
    pause = float(os.environ.get("TI_CYCLE_SECONDS", "300")) if pause is None else pause
    pipeline = build_pipeline()
    while not (stop and stop.is_set()):
        try:
            STATE["last_summary"] = pipeline.cycle()
            STATE["last_error"] = None
        except Exception as exc:
            log.exception("cycle failed")
            STATE["last_error"] = str(exc)
        STATE["cycles"] += 1
        STATE["last_cycle_at"] = time.time()
        if stop:
            stop.wait(pause)
        else:
            time.sleep(pause)
