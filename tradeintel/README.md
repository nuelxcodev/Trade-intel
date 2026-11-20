# tradeintel

Foundation layer for the trading intelligence platform: data, indicators, trade parameters, backtesting.

    pip install -r requirements.txt
    python -m pytest -q
    python -m tradeintel.cli sync BTCUSDT 1h --days 90     # needs network access to api.binance.com
    python -m tradeintel.cli check BTCUSDT 1h               # gaps and invalid rows
    python -m tradeintel.cli backtest BTCUSDT 1h            # walk-forward, fees + slippage
    python -m tradeintel.cli backtest --synthetic           # offline wiring check, not market data

Modules: `exchange/` (adapter interface + Binance spot), `store` (SQLite, Timescale-shaped schema),
`sync` (incremental, closed candles only, gap repair), `integrity`, `indicators` (causal),
`tradeparams` (structure + ATR entry/SL/TP, risk-based sizing), `backtest` (next-open fills,
stop-first on ambiguous bars, walk-forward splits), `strategies` (baseline rule-based).

## Pattern knowledge base

    python -m tradeintel.cli learn BTCUSDT 1h               # detect, measure, store, validate
    python -m tradeintel.cli learn --synthetic              # offline wiring check

`patterns` (9 causal detectors + market context), `outcomes` (MFE/MAE/time-to-move in ATR units,
stop-first, unresolved horizons excluded, base rate over every bar), `knowledge` (SQLite event
store, Wilson intervals, lift over base rate, chronological out-of-sample validation).

## ML layer

    python -m tradeintel.cli train BTCUSDT 1h               # needs synced candles
    python -m tradeintel.cli train --synthetic              # wiring check only; saves a model, delete it

`features` (causal features, labels, purged splits), `ml` (per-direction gradient boosting, isotonic
calibration, threshold chosen on validation, net-of-cost R on test, model registry with data
fingerprint, retrain trigger), `similarity` (nearest past bars, built only from resolved history).
A model that shows no net edge on the test window sets `usable=False` and never qualifies a signal.

## Signals and scanner

`signals` (immutable signal rows, append-only history, closed-candle state machine, idempotent monitor),
`opportunity` (hard quality gates, explainable 0-100 score with hand-set weights), `scanner`
(per-pair scan from the latest closed candle, cached per candle, explicit "no qualified
opportunities" result, one open signal per pair and direction).
