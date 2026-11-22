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

## Performance and notifications

`performance` (net R per closed signal using the thirds exit model, win rate, profit factor,
max drawdown in R, MFE/MAE from stored candles; breakdowns by pair, timeframe, direction, pattern,
model; no-entry and open signals reported separately). `notify` (messages rendered from the stored
signal, in-app / Telegram / email channels, per-user score/timeframe/probability thresholds,
retry with backoff, one delivery per signal, user and channel). Secrets come from
TELEGRAM_BOT_TOKEN and SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASSWORD/SMTP_FROM.

## Running as a service (Railway or any host)

    python -m tradeintel.main        # starts the background worker and the API on $PORT

The Procfile, `railway.json` (start command and `/health` check) and a root `main.py` all point at it.
One process runs both the worker and the API because they share one SQLite file.

**Persistence.** The container disk is wiped on every deploy. Attach a volume (for example at `/data`)
and set `TI_DB_PATH=/data/tradeintel.db`; models are saved next to it unless `TI_MODEL_DIR` is set.
Without a volume, candles, signals and models are lost on each redeploy.

| Variable | Purpose | Default |
|---|---|---|
| `PORT` | HTTP port (set by the host) | 8000 |
| `TI_API_KEY` | Required as `X-API-Key` on every `/api` route; enables `POST /api/prefs/<user>` | unset: reads open, writes off |
| `TI_DB_PATH`, `TI_MODEL_DIR` | storage locations | `tradeintel.db`, `<db dir>/models` |
| `TI_SYMBOLS` | pairs to monitor, comma separated | 8 large USDT pairs |
| `TI_TIMEFRAMES` | timeframes, from 1m, 5m, 15m, 30m, 1h, 4h, 1d | `1h` |
| `TI_HISTORY_DAYS`, `TI_CYCLE_SECONDS` | history kept, pause between cycles | 365, 300 |
| `TI_BINANCE_URL` | market-data host; change it if sync logs HTTP 451 or 403 (region block) | `https://data-api.binance.vision` |
| `TI_CAPITAL`, `TI_RISK_PER_TRADE`, `TI_MIN_SCORE`, `TI_MIN_RR`, `TI_MIN_QUOTE_VOLUME` | trading thresholds | 10000, 0.01, 60, 1.8, 5000000 |
| `TELEGRAM_BOT_TOKEN`, `SMTP_HOST` `SMTP_PORT` `SMTP_USER` `SMTP_PASSWORD` `SMTP_FROM` | enable Telegram and email delivery | unset |
| `TI_CORS_ORIGIN` | allowed dashboard origin | `*` |

First boot downloads history, trains a model per pair (needs 1,500 candles), then scans. Expect
"No qualified trading opportunities" until a model shows an edge on its untouched test window.
Endpoints: `/health`, `/api/overview`, `/api/opportunities?tf=1h`, `/api/signals`, `/api/signals/<id>`,
`/api/performance`, `/api/models`, `/api/notifications?user=default`, `POST /api/prefs/<user>`.

## Dashboard

The service serves the dashboard at `/` (file: `tradeintel/web/index.html`, no build step). It reads the
live API, refreshes every 30 seconds, and asks for the access key if `TI_API_KEY` is set. It shows market
breadth and regime, ranked opportunities (or the explicit "none" message with the reason each pair was
rejected), an interactive-free candlestick chart with EMAs and entry, stop loss and take profit levels,
the trade plan, signal history with net R results, pattern performance, model status, and
evidence-based explanations. Banners surface sync failures (for example a region block) and models
still waiting for history. Explanations are generated from the engines' stored evidence, not by a
language model.
