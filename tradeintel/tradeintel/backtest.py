"""Backtester with fees, slippage and conservative execution.

Execution model (all chosen to avoid flattering results):
  * A signal is raised at the close of bar i using data up to bar i only.
  * The order fills at the OPEN of bar i+1, with adverse slippage.
  * Stops fill with adverse slippage. Take-profit limits fill at their price.
  * If a bar touches both the stop and a target, the stop is assumed to hit first.
  * Half the position exits at TP1 and the stop moves to entry; the rest exits at TP2.
  * One position at a time. Unfilled or expired setups are not traded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .config import DEFAULT, Settings
from .tradeparams import compute_trade_params, position_size

Strategy = Callable[[pd.DataFrame, int], "str | None"]


@dataclass
class Trade:
    signal_bar: int
    entry_bar: int
    exit_bar: int
    direction: str
    entry: float
    exit: float
    qty: float
    pnl: float
    r_multiple: float
    mfe_r: float
    mae_r: float
    outcome: str      # TP2, TP1+BE, SL, TIMEOUT


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[float]
    metrics: dict = field(default_factory=dict)


def _metrics(trades: list[Trade], equity: list[float], start_capital: float) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    pnl = np.array([t.pnl for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    return {
        "trades": n,
        "win_rate": float((pnl > 0).mean()),
        "avg_r": float(np.mean([t.r_multiple for t in trades])),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "total_return": float(eq[-1] / start_capital - 1),
        "max_drawdown": float(((peak - eq) / peak).max()),
        "avg_mfe_r": float(np.mean([t.mfe_r for t in trades])),
        "avg_mae_r": float(np.mean([t.mae_r for t in trades])),
    }


def run_backtest(df: pd.DataFrame, strategy: Strategy, settings: Settings = DEFAULT,
                 start: int = 0, end: int | None = None) -> BacktestResult:
    """Backtest bars [start, end). `df` must already hold causal indicators from add_indicators."""
    end = len(df) if end is None else end
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    fee, slip = settings.fee_rate, settings.slippage
    equity = settings.capital
    curve, trades = [equity], []
    i = max(start, 1)

    while i < end - 1:
        direction = strategy(df, i)
        params = compute_trade_params(df, i, direction) if direction else None
        if params is None or params.rr < settings.min_rr:
            i += 1
            continue

        s = 1 if direction == "LONG" else -1
        eb = i + 1
        fill = o[eb] * (1 + s * slip)                      # adverse fill at next open
        risk_unit = (fill - params.sl) * s
        if risk_unit <= 0:                                  # gapped through the stop: skip
            i += 1
            continue
        qty = position_size(equity, settings.risk_per_trade, fill, params.sl)
        if qty <= 0:
            i += 1
            continue
        init_risk = qty * risk_unit

        remaining, stop, tp1_done = qty, params.sl, False
        cash = -fee * fill * qty                            # entry fee
        mfe = mae = 0.0
        exit_bar, exit_price, outcome = None, None, "TIMEOUT"
        last = min(end - 1, eb + settings.max_hold_bars)

        for j in range(eb, last + 1):
            best = h[j] if s == 1 else l[j]
            worst = l[j] if s == 1 else h[j]
            mfe = max(mfe, (best - fill) * s / risk_unit)
            mae = min(mae, (worst - fill) * s / risk_unit)

            if (worst - stop) * s <= 0:                     # stop first (conservative)
                px = stop * (1 - s * slip)
                cash += remaining * (px - fill) * s - fee * px * remaining
                exit_bar, exit_price = j, px
                outcome = "SL" if not tp1_done else "TP1+BE"
                remaining = 0.0
                break
            if not tp1_done and (best - params.tp1) * s >= 0:
                half = qty / 2
                cash += half * (params.tp1 - fill) * s - fee * params.tp1 * half
                remaining -= half
                tp1_done, stop = True, fill
            if tp1_done and (best - params.tp2) * s >= 0:
                cash += remaining * (params.tp2 - fill) * s - fee * params.tp2 * remaining
                exit_bar, exit_price, outcome, remaining = j, params.tp2, "TP2", 0.0
                break

        if remaining > 0:                                   # signal expired: exit at the close
            exit_bar, exit_price = last, c[last]
            cash += remaining * (exit_price - fill) * s - fee * exit_price * remaining
            outcome = "TIMEOUT"

        equity += cash
        trades.append(Trade(i, eb, exit_bar, direction, fill, exit_price, qty, cash,
                            cash / init_risk, mfe, mae, outcome))
        curve.append(equity)
        i = exit_bar + 1

    return BacktestResult(trades, curve, _metrics(trades, curve, settings.capital))


def chronological_splits(n: int, train: int, test: int, step: int | None = None) -> list[tuple[range, range]]:
    """Walk-forward windows. Each test window starts exactly where its train window ends."""
    step = step or test
    out, start = [], 0
    while start + train + test <= n:
        out.append((range(start, start + train), range(start + train, start + train + test)))
        start += step
    return out


def walk_forward(df: pd.DataFrame, strategy: Strategy, settings: Settings = DEFAULT,
                 train: int = 1000, test: int = 250) -> list[BacktestResult]:
    """Out-of-sample results per fold. Rule-based strategies have nothing to fit, so only the
    test windows are traded; the train window supplies indicator history. ML strategies fit on
    the train range and are scored on the test range with the same interface."""
    return [run_backtest(df, strategy, settings, start=te.start, end=te.stop)
            for _, te in chronological_splits(len(df), train, test)]
