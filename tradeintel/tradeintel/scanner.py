"""Live market scanner.

Independent of any frontend. For every pair with a trained model it reads the latest closed
candles, computes features, asks the model and the similarity index, builds trade parameters,
applies the quality gates and ranks what survives. Results are cached per (pair, timeframe,
last closed candle), so an unchanged market costs nothing to re-scan.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT, Settings
from .features import build_dataset, make_features
from .indicators import add_indicators
from .ml import DirectionalModel, load_models
from .opportunity import Opportunity, composite, gate_failures, rank, score_components
from .outcomes import OutcomeSpec
from .patterns import detect_all
from .signals import SignalStore
from .similarity import SimilarityIndex
from .store import CandleStore
from .timeframes import tf_ms
from .tradeparams import compute_trade_params, position_size

log = logging.getLogger(__name__)
LOOKBACK_BARS = 400


@dataclass
class PairContext:
    models: dict[int, DirectionalModel]
    model_id: str
    sim_index: SimilarityIndex
    base_rate: dict[int, float]


class RegistryProvider:
    """Loads the newest saved model for a pair and builds its similarity index once."""

    def __init__(self, store: CandleStore, exchange: str, directory: str = "models"):
        self.store, self.exchange, self.directory = store, exchange, directory
        self._cache: dict[str, PairContext] = {}

    def get(self, symbol: str, tf: str) -> PairContext | None:
        root = Path(self.directory)
        metas = []
        for p in root.glob(f"{symbol}-{tf}-*/meta.json") if root.exists() else []:
            m = json.loads(p.read_text())
            metas.append(m)
        if not metas:
            return None
        meta = max(metas, key=lambda m: m["created_at"])
        mid = meta["model_id"]
        if mid not in self._cache:
            models, meta = load_models(mid, self.directory)
            raw = self.store.load(self.exchange, symbol, tf, end_ms=meta["trained_through_open_time"] + 1)
            spec = OutcomeSpec(**meta["spec"])
            ds = build_dataset(add_indicators(raw), spec)
            self._cache[mid] = PairContext(models, mid, SimilarityIndex(ds),
                                           {d: float(ds.y[d].mean()) for d in (1, -1)})
        return self._cache[mid]


@dataclass
class ScanReport:
    scanned: int = 0
    from_cache: int = 0
    opportunities: list[Opportunity] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)     # symbol -> reasons
    new_signals: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        if self.opportunities:
            return f"{len(self.opportunities)} qualified opportunity(ies)"
        return "No qualified trading opportunities right now"


class Scanner:
    def __init__(self, exchange: str, store: CandleStore, provider, signals: SignalStore | None = None,
                 settings: Settings = DEFAULT):
        self.exchange, self.store, self.provider = exchange, store, provider
        self.signals, self.settings = signals, settings
        self._seen: dict[tuple[str, str], int] = {}
        self._last: dict[tuple[str, str], tuple[Opportunity | None, list[str]]] = {}

    def _scan_pair(self, symbol: str, tf: str) -> tuple[Opportunity | None, list[str]]:
        ctx = self.provider.get(symbol, tf)
        if ctx is None:
            return None, ["no trained model for this pair"]
        bounds = self.store.bounds(self.exchange, symbol, tf)
        if bounds is None:
            return None, ["no candle data"]
        step = tf_ms(tf)
        raw = self.store.load(self.exchange, symbol, tf, start_ms=bounds[1] - LOOKBACK_BARS * step)
        if len(raw) < 120:
            return None, ["not enough recent candles"]
        df = add_indicators(raw)
        i = len(df) - 1
        feats = make_features(df)
        row = feats.iloc[[i]]
        if row.isna().any(axis=None):
            return None, ["features not yet available"]
        nb = ctx.sim_index.query(row, k=self.settings.scan_neighbours)
        quote_vol = float(df["quote_volume"].iloc[-max(1, int(86_400_000 // step)):].sum())
        atr_ratio = float(row["atr_ratio"].iloc[0])
        patterns_now = {n: int(v) for n, v in detect_all(df).iloc[i].items() if v != 0}

        best: tuple[Opportunity | None, list[str]] = (None, [])
        reasons_all: list[str] = []
        for d, name in ((1, "LONG"), (-1, "SHORT")):
            m = ctx.models[d]
            p = float(m.predict_proba(row)[0])
            qualifies = bool(m.usable and m.threshold is not None and p >= m.threshold)
            if not qualifies:
                reasons_all.append(f"{name}: " + gate_failures(
                    qualifies=False, usable=m.usable, rr=99, quote_volume_24h=1e18, atr_ratio=1.0,
                    nb_mean_r=1.0, settings=self.settings)[0])
                continue
            params = compute_trade_params(df, i, name)
            if params is None:
                reasons_all.append(f"{name}: no structural stop within a sensible distance")
                continue
            nb_success = nb.long_success if d == 1 else nb.short_success
            nb_r = nb.long_mean_r if d == 1 else nb.short_mean_r
            fails = gate_failures(qualifies=True, usable=m.usable, rr=params.rr, quote_volume_24h=quote_vol,
                                  atr_ratio=atr_ratio, nb_mean_r=nb_r, settings=self.settings)
            if fails:
                reasons_all += [f"{name}: {f}" for f in fails]
                continue
            r = df.iloc[i]
            comps = score_components(
                p=p, thr=m.threshold, nb_success=nb_success, base_rate=ctx.base_rate[d],
                trend_aligned=bool((r["ema20"] > r["ema50"]) == (d == 1)), adx=float(r["adx14"]),
                rel_vol=float(r["rel_vol"]), atr_ratio=atr_ratio, quote_volume_24h=quote_vol,
                min_quote_volume=self.settings.min_quote_volume_24h, rr=params.rr, min_rr=self.settings.min_rr)
            score = composite(comps)
            if score < self.settings.min_score:
                reasons_all.append(f"{name}: composite score {score:.0f} is below {self.settings.min_score:.0f}")
                continue
            qty = position_size(self.settings.capital, self.settings.risk_per_trade, params.entry, params.sl)
            half_zone = 0.2 * float(r["atr14"])
            opp = Opportunity(
                symbol=symbol, tf=tf, direction=name, bar_open_time=int(df["open_time"].iat[i]),
                score=round(score, 1), probability=p, threshold=m.threshold, entry=params.entry,
                entry_lo=params.entry - half_zone, entry_hi=params.entry + half_zone, sl=params.sl,
                tp1=params.tp1, tp2=params.tp2, tp3=params.tp3, rr=params.rr, qty=qty,
                risk_amount=qty * params.risk_per_unit, notional=qty * params.entry,
                hold_bars=self.settings.hold_bars, expiry_bars=self.settings.expiry_bars, model_id=ctx.model_id,
                components={k: round(v, 3) for k, v in comps.items()},
                invalidation=[f"a candle trades through the stop loss at {params.sl:.6g} before entry",
                              f"first target {params.tp1:.6g} is reached without an entry",
                              f"no entry within {self.settings.expiry_bars} candles"],
                evidence={"patterns_now": patterns_now, "neighbours": nb.__dict__,
                          "base_rate": ctx.base_rate[d], "trade_notes": params.notes,
                          "market": {"adx": float(r["adx14"]), "rsi": float(r["rsi14"]), "atr_ratio": atr_ratio,
                                     "rel_vol": float(r["rel_vol"]), "quote_volume_24h": quote_vol}})
            if best[0] is None or opp.score > best[0].score:
                best = (opp, [])
        return (best[0], reasons_all if best[0] is None else [])

    def run_once(self, symbols: list[str], tf: str) -> ScanReport:
        rep = ScanReport()
        for sym in symbols:
            key = (sym, tf)
            b = self.store.bounds(self.exchange, sym, tf)
            last_bar = b[1] if b else None
            if last_bar is not None and self._seen.get(key) == last_bar:
                opp, reasons = self._last[key]
                rep.from_cache += 1
                new = False
            else:
                try:
                    opp, reasons = self._scan_pair(sym, tf)
                except Exception as exc:                  # one broken pair must not stop the scan
                    log.exception("scan failed for %s", sym)
                    opp, reasons = None, [f"scan error: {exc}"]
                self._seen[key], self._last[key] = last_bar, (opp, reasons)
                rep.scanned += 1
                new = True
            if opp is None:
                rep.rejected[sym] = reasons
                continue
            rep.opportunities.append(opp)
            if new and self.signals and not self.signals.has_open(sym, tf, opp.direction):
                rep.new_signals.append(self.signals.create({
                    "bar_open_time": opp.bar_open_time, "symbol": sym, "tf": tf, "direction": opp.direction,
                    "entry": opp.entry, "entry_lo": opp.entry_lo, "entry_hi": opp.entry_hi, "sl": opp.sl,
                    "tp1": opp.tp1, "tp2": opp.tp2, "tp3": opp.tp3, "rr": opp.rr, "qty": opp.qty,
                    "risk_amount": opp.risk_amount, "score": opp.score, "probability": opp.probability,
                    "model_id": opp.model_id, "expiry_bars": opp.expiry_bars, "hold_bars": opp.hold_bars,
                    "payload": {"components": opp.components, "invalidation": opp.invalidation,
                                "evidence": opp.evidence, "threshold": opp.threshold}}))
        rep.opportunities = rank(rep.opportunities)
        return rep
