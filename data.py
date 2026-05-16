"""Data loading: top-100 universe (CoinGecko) and OHLCV (Binance via CCXT)."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import ccxt
import pandas as pd
import requests
import streamlit as st

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"


def _exchange(market: str):
    """Return a CCXT exchange instance for the given market."""
    if market == "futures":
        return ccxt.binanceusdm({"enableRateLimit": True})
    return ccxt.binance({"enableRateLimit": True})


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_top_symbols(quote: str = "USDT", limit: int = 100,
                    market: str = "spot") -> pd.DataFrame:
    """Return a DataFrame of the top-`limit` coins by market cap that trade on
    Binance. `market` is "spot" or "futures" (USDⓂ perpetuals).
    """
    rows: list[dict] = []
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "sparkline": "false",
    }
    r = requests.get(COINGECKO_URL, params=params, timeout=30)
    r.raise_for_status()
    coins = r.json()

    ex = _exchange(market)
    markets = ex.load_markets()
    if market == "futures":
        # USDⓂ perpetual swaps quoted in USDT
        available = {
            m: info for m, info in markets.items()
            if info.get("swap") and info.get("linear") and info.get("quote") == "USDT"
        }
    else:
        available = {m: info for m, info in markets.items() if info.get("spot")}

    for c in coins:
        sym = (c.get("symbol") or "").upper()
        if not sym:
            continue
        spot_pair = f"{sym}/{quote}"
        # For futures, CCXT id is "BASE/USDT:USDT". The base/quote match still works.
        match = None
        for m, info in available.items():
            if info.get("base") == sym and info.get("quote") == quote:
                match = m
                break
        if market == "spot" and spot_pair in available:
            match = spot_pair
        if match:
            rows.append({
                "rank": c["market_cap_rank"],
                "symbol": match,
                "name": c["name"],
                "market_cap": c["market_cap"],
                "price": c["current_price"],
            })
        if len(rows) >= limit:
            break

    return pd.DataFrame(rows)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_binance_fees(symbol: str, market: str = "spot") -> dict:
    """Return Binance's public default maker/taker fees for `symbol` (in bps).

    - Spot defaults: 10 bps maker / 10 bps taker.
    - USDⓂ futures defaults: 2 bps maker / 5 bps taker.
    """
    ex = _exchange(market)
    markets = ex.load_markets()
    m = markets.get(symbol, {})
    taker = m.get("taker")
    maker = m.get("maker")
    if taker is None:
        taker = ex.fees.get("trading", {}).get("taker", 0.0005 if market == "futures" else 0.001)
    if maker is None:
        maker = ex.fees.get("trading", {}).get("maker", 0.0002 if market == "futures" else 0.001)
    return {
        "maker_bps": float(maker) * 1e4,
        "taker_bps": float(taker) * 1e4,
    }


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_ohlcv(symbol: str, timeframe: str = "4h", since_ms: int | None = None,
                limit_per_call: int = 1000, market: str = "spot") -> pd.DataFrame:
    """Fetch OHLCV from Binance (spot or USDⓂ futures), paginating back to `since_ms`."""
    ex = _exchange(market)
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    if since_ms is None:
        # Default: ~3 years
        since_ms = int((datetime.now(timezone.utc).timestamp() - 3 * 365 * 86400) * 1000)

    all_rows: list[list] = []
    cursor = since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit_per_call)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        next_cursor = last_ts + tf_ms
        if next_cursor <= cursor or len(batch) < limit_per_call:
            break
        cursor = next_cursor
        time.sleep(ex.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt").drop(columns=["ts"])
    return df.astype(float)
