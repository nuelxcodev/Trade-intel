"""Performance of generated signals, measured from their immutable history.

Exit model for a signal (matches the lifecycle rules): one third of the position exits at each
of TP1, TP2 and TP3. A stop before TP1 loses 1R; after TP1 the stop sits at entry, so the
unfilled thirds exit at 0. An expired active signal exits its remaining thirds at the expiry
close. Round-trip fees and slippage are subtracted in R. Signals that never reached their entry
are counted separately and are not wins or losses. Open signals are excluded until they finish.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import DEFAULT, Settings
from .signals import SignalStore, Status
from .store import CandleStore
from .timeframes import tf_ms

_TP_STATUS = {Status.TP1.value: 1, Status.TP2.value: 2, Status.TP3.value: 3}
_FINAL = {Status.STOP.value, Status.TP3.value, Status.INVALIDATED.value, Status.EXPIRED.value}


def signal_outcome(sig: dict, events: list[dict], settings: Settings = DEFAULT) -> dict:
    statuses = [e["status"] for e in events]
    last = events[-1]
    entered = Status.ACTIVE.value in statuses
    base = {"id": sig["id"], "symbol": sig["symbol"], "tf": sig["tf"], "direction": sig["direction"],
            "model_id": sig["model_id"], "planned_rr": sig["rr"], "final_status": last["status"],
            "patterns": sorted(sig["payload"].get("evidence", {}).get("patterns_now", {})) or ["none"],
            "entered": entered, "closed": last["status"] in _FINAL}
    if not base["closed"]:
        return {**base, "state": "open"}
    if not entered:
        return {**base, "state": "no_entry"}
    s = 1 if sig["direction"] == "LONG" else -1
    risk = abs(sig["entry"] - sig["sl"])
    tp_hit = max([_TP_STATUS[x] for x in statuses if x in _TP_STATUS], default=0)
    r = sum(abs(sig[f"tp{k}"] - sig["entry"]) / risk for k in range(1, tp_hit + 1)) / 3.0
    remaining = (3 - tp_hit) / 3.0
    if last["status"] == Status.STOP.value and tp_hit == 0:
        r -= remaining
    elif last["status"] == Status.EXPIRED.value:
        r += remaining * (last["price"] - sig["entry"]) * s / risk
    cost = 2 * (settings.fee_rate + settings.slippage) * sig["entry"] / risk
    exit_bar = last["bar_open_time"]
    active_bar = next(e["bar_open_time"] for e in events if e["status"] == Status.ACTIVE.value)
    return {**base, "state": "closed", "r_gross": r, "r_net": r - cost, "tp_hit": tp_hit,
            "active_bar": active_bar, "exit_bar": exit_bar}


def add_excursions(row: dict, sig: dict, candles: CandleStore, exchange: str) -> dict:
    """Maximum favourable and adverse excursion (in R) while the trade was live."""
    step = tf_ms(sig["tf"])
    df = candles.load(exchange, sig["symbol"], sig["tf"], start_ms=row["active_bar"], end_ms=row["exit_bar"] + step)
    if df.empty:
        return row
    s = 1 if sig["direction"] == "LONG" else -1
    risk = abs(sig["entry"] - sig["sl"])
    best = df["high"].max() if s == 1 else df["low"].min()
    worst = df["low"].min() if s == 1 else df["high"].max()
    return {**row, "mfe_r": max(0.0, (best - sig["entry"]) * s / risk),
            "mae_r": -max(0.0, (sig["entry"] - worst) * s / risk)}


def metrics(rows: list[dict]) -> dict:
    closed = [r for r in rows if r["state"] == "closed"]
    out = {"signals": len(rows), "open": sum(r["state"] == "open" for r in rows),
           "no_entry": sum(r["state"] == "no_entry" for r in rows), "trades": len(closed)}
    if not closed:
        return out
    r = np.array([x["r_net"] for x in closed])
    ordered = np.array([x["r_net"] for x in sorted(closed, key=lambda x: x["exit_bar"])])
    curve = np.concatenate([[0.0], np.cumsum(ordered)])
    wins, losses = r[r > 0], r[r < 0]
    out.update({
        "wins": int((r > 0).sum()), "losses": int((r < 0).sum()), "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()), "total_r": float(r.sum()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "avg_planned_rr": float(np.mean([x["planned_rr"] for x in closed])),
        "max_drawdown_r": float((np.maximum.accumulate(curve) - curve).max()),
    })
    for key in ("mfe_r", "mae_r"):
        vals = [x[key] for x in closed if key in x]
        if vals:
            out[f"avg_{key}"] = float(np.mean(vals))
    return out


def breakdown(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for k in (r[key] if isinstance(r[key], list) else [r[key]]):
            groups[k].append(r)
    return {k: metrics(v) for k, v in sorted(groups.items())}


def build_report(signals: SignalStore, settings: Settings = DEFAULT, candles: CandleStore | None = None,
                 exchange: str | None = None) -> dict:
    rows = []
    for (sid,) in signals.conn.execute("SELECT id FROM signals ORDER BY created_at, id").fetchall():
        sig = signals.get(sid)
        sig["id"] = sid
        row = signal_outcome(sig, signals.history(sid), settings)
        if candles is not None and row["state"] == "closed":
            row = add_excursions(row, sig, candles, exchange)
        rows.append(row)
    return {"overall": metrics(rows),
            "by_symbol": breakdown(rows, "symbol"), "by_timeframe": breakdown(rows, "tf"),
            "by_direction": breakdown(rows, "direction"), "by_pattern": breakdown(rows, "patterns"),
            "by_model": breakdown(rows, "model_id"), "signals": rows}
