"""Command line entry points.

    python -m tradeintel.cli sync BTCUSDT 1h --days 90
    python -m tradeintel.cli check BTCUSDT 1h
    python -m tradeintel.cli backtest BTCUSDT 1h
    python -m tradeintel.cli backtest --synthetic          # offline wiring check only
"""

from __future__ import annotations

import argparse
import json
import logging

from .backtest import walk_forward
from .config import DEFAULT
from .exchange import BinanceSpot
from .indicators import add_indicators
from .integrity import find_gaps, validate_ohlcv
from .store import CandleStore
from .strategies import ema_pullback
from .sync import Syncer, now_ms
from .synthetic import make_candles
from .timeframes import tf_ms


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tradeintel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sync", "check", "backtest", "learn", "train"):
        p = sub.add_parser(name)
        p.add_argument("symbol", nargs="?", default="BTCUSDT")
        p.add_argument("tf", nargs="?", default="1h")
        p.add_argument("--days", type=int, default=90)
        p.add_argument("--db", default=DEFAULT.db_path)
        if name in ("backtest", "learn", "train"):
            p.add_argument("--synthetic", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = CandleStore(args.db)

    if args.cmd == "sync":
        since = now_ms() - args.days * 86_400_000
        added = Syncer(BinanceSpot(), store).sync(args.symbol, args.tf, since)
        print(f"{args.symbol} {args.tf}: +{added} candles")
    elif args.cmd == "check":
        times = store.open_times("binance", args.symbol, args.tf)
        bad = validate_ohlcv(store.load("binance", args.symbol, args.tf))
        print(json.dumps({"candles": len(times), "gaps": len(find_gaps(times, tf_ms(args.tf))),
                          "invalid_rows": len(bad)}))
    elif args.cmd == "train":
        import pandas as pd
        from .features import build_dataset
        from .ml import predict_latest, save_models, train_models
        from .outcomes import OutcomeSpec
        if args.synthetic:
            raw = pd.DataFrame([c.__dict__ for c in make_candles(8000, args.tf)])
        else:
            raw = store.load("binance", args.symbol, args.tf)
        df, spec = add_indicators(raw), OutcomeSpec()
        models = train_models(build_dataset(df, spec))
        for d, m in models.items():
            s = m.metrics
            print(f"\n{'LONG' if d == 1 else 'SHORT'} usable={m.usable} threshold={m.threshold}")
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                              if k not in ("reliability_test", "bar_ranges")}))
        mid = save_models(models, df, args.symbol, args.tf, spec)
        print("\nsaved model", mid)
        print(json.dumps(predict_latest(models, df, mid), indent=2))
    elif args.cmd == "learn":
        import pandas as pd
        from .knowledge import KnowledgeBase, build_events, summarize, validate_chronologically
        from .outcomes import OutcomeSpec, baseline_rates
        if args.synthetic:
            raw = pd.DataFrame([c.__dict__ for c in make_candles(6000, args.tf)])
        else:
            raw = store.load("binance", args.symbol, args.tf)
        df, spec = add_indicators(raw), OutcomeSpec()
        base = baseline_rates(df, spec)
        events = build_events(df, args.symbol, args.tf, spec)
        KnowledgeBase(store).save_events(events, args.symbol, args.tf)
        pd.set_option("display.width", 200)
        print("baseline success rate (every bar):", {k: round(v, 3) for k, v in base.items()})
        print(summarize(events, base, total_bars=len(df)).round(3).to_string(index=False))
        print("\nout-of-sample check (learn on first 70%, test on last 30%):")
        print(validate_chronologically(events, base).round(3).to_string(index=False))
    elif args.cmd == "backtest":
        if args.synthetic:
            import pandas as pd
            raw = pd.DataFrame([c.__dict__ for c in make_candles(4000, args.tf)])
        else:
            raw = store.load("binance", args.symbol, args.tf)
        df = add_indicators(raw)
        for k, res in enumerate(walk_forward(df, ema_pullback, DEFAULT), start=1):
            print(f"fold {k}:", json.dumps({m: round(v, 4) if isinstance(v, float) else v
                                            for m, v in res.metrics.items()}))


if __name__ == "__main__":
    main()
