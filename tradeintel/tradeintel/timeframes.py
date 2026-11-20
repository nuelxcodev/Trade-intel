"""Supported candle timeframes and their durations in milliseconds."""

TF_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def tf_ms(tf: str) -> int:
    try:
        return TF_MS[tf]
    except KeyError:
        raise ValueError(f"Unsupported timeframe {tf!r}; choose from {sorted(TF_MS)}") from None
