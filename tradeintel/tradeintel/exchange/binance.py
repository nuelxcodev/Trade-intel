"""Binance spot adapter using the public REST API (no API key needed for market data)."""

from __future__ import annotations

import logging
import time

import requests

from .base import Candle, Exchange

log = logging.getLogger(__name__)

# Binance's market-data-only host. It serves klines, exchangeInfo and time, which is all this
# platform needs, and is reachable from more cloud regions than api.binance.com.
BASE_URL = "https://data-api.binance.vision"


class BinanceError(RuntimeError):
    pass


class BinanceSpot(Exchange):
    name = "binance"

    def __init__(self, base_url: str = BASE_URL, session: requests.Session | None = None,
                 max_retries: int = 5, timeout: float = 15.0):
        self.base_url = base_url
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.timeout = timeout

    def _get(self, path: str, params: dict | None = None):
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("Binance request failed (%s), attempt %d/%d", exc, attempt, self.max_retries)
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (418, 429):
                    # Rate limited or banned: honour Retry-After, never hammer the API.
                    delay = max(delay, float(resp.headers.get("Retry-After", delay)))
                    log.warning("Binance rate limit (%s), sleeping %.1fs", resp.status_code, delay)
                elif resp.status_code >= 500:
                    log.warning("Binance server error %s, attempt %d", resp.status_code, attempt)
                else:
                    hint = (" (this server's region is blocked by the exchange; set TI_BINANCE_URL to another "
                            "market-data endpoint or deploy in another region)" if resp.status_code in (403, 451) else "")
                    raise BinanceError(f"{resp.status_code}: {resp.text[:200]}{hint}")
            if attempt < self.max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise BinanceError(f"GET {path} failed after {self.max_retries} attempts")

    def list_symbols(self, quote: str) -> list[str]:
        info = self._get("/api/v3/exchangeInfo")
        return sorted(
            s["symbol"] for s in info["symbols"]
            if s["quoteAsset"] == quote and s["status"] == "TRADING" and s.get("isSpotTradingAllowed")
        )

    def fetch_klines(self, symbol: str, tf: str, start_ms: int, limit: int = 1000) -> list[Candle]:
        rows = self._get("/api/v3/klines", {
            "symbol": symbol, "interval": tf, "startTime": start_ms, "limit": min(limit, 1000),
        })
        return [
            Candle(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                   float(r[5]), float(r[7]), int(r[8]))
            for r in rows
        ]

    def server_time_ms(self) -> int:
        return int(self._get("/api/v3/time")["serverTime"])
