"""Auto-trading CLI driven by one of the indicators from `strategies.py`.

SECURITY:
- API key & secret are read from the BINANCE_API_KEY / BINANCE_API_SECRET
  environment variables ONLY. They are never accepted on the command line
  (argv is visible to other users via `ps`) and never written to disk by
  this script.
- The Streamlit launcher injects them into the subprocess environment and
  then wipes them from its own session state.

Behavior:
- Polls Binance OHLCV every `--poll-seconds` (default = one bar).
- Computes the indicator's target position {-1, 0, +1} on the latest
  CLOSED bar.
- Compares to the live exchange position (best-effort) and submits a
  market order to align.
- `--dry-run` (default ON) logs the order it would have submitted instead
  of sending anything.

Usage (env-only auth):
    BINANCE_API_KEY=... BINANCE_API_SECRET=... \
      python trade.py --market spot --symbol BTC/USDT --tf 1h \
                      --strategy "EMA Cross" --params '{"fast":20,"slow":50}' \
                      --notional-pct 25 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta

import ccxt

from strategies import STRATEGIES


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _build_exchange(market: str) -> ccxt.Exchange:
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        log("FATAL: BINANCE_API_KEY / BINANCE_API_SECRET env vars not set.")
        sys.exit(2)
    cls = ccxt.binanceusdm if market == "futures" else ccxt.binance
    ex = cls({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future" if market == "futures" else "spot"},
    })
    ex.verbose = False
    return ex


def _fetch_bars(ex: ccxt.Exchange, symbol: str, tf: str, n: int):
    """Return CLOSED bars only — the most recent (in-progress) bar from CCXT
    is dropped so signals match the backtester (decide on close of bar t,
    execute at open of bar t+1, no intra-bar flicker).
    """
    import pandas as pd
    bars = ex.fetch_ohlcv(symbol, timeframe=tf, limit=max(n, 400))
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt").drop(columns=["ts"]).astype(float)
    return df.iloc[:-1] if len(df) > 1 else df


def _get_position_size(ex: ccxt.Exchange, market: str, symbol: str) -> float:
    """Return current signed position in base units (longs positive)."""
    try:
        if market == "futures":
            positions = ex.fetch_positions([symbol])
            for p in positions:
                if p.get("symbol") == symbol:
                    return float(p.get("contracts") or 0.0) * (1 if (p.get("side") == "long") else -1)
            return 0.0
        else:
            base = symbol.split("/")[0]
            bal = ex.fetch_balance()
            return float(bal.get(base, {}).get("free", 0.0))
    except Exception as e:
        log(f"WARN: could not fetch position ({e}); assuming 0.")
        return 0.0


def _free_usdt(ex: ccxt.Exchange, market: str) -> float:
    try:
        bal = ex.fetch_balance()
        if market == "futures":
            return float(bal.get("USDT", {}).get("free")
                         or bal.get("total", {}).get("USDT", 0.0))
        return float(bal.get("USDT", {}).get("free", 0.0))
    except Exception as e:
        log(f"WARN: could not fetch USDT balance ({e}); assuming 0.")
        return 0.0


def _qty_for_pct(ex: ccxt.Exchange, market: str, price: float, pct: float,
                 leverage: float) -> float:
    if price <= 0 or pct <= 0:
        return 0.0
    usdt = _free_usdt(ex, market)
    margin = usdt * pct / 100.0
    # On futures, the user puts up `margin` and controls `margin * leverage`
    # of notional. On spot, leverage is always 1.
    eff_lev = float(leverage) if market == "futures" else 1.0
    notional = margin * eff_lev
    qty = notional / price
    log(f"  sizing: free={usdt:.4f} USDT × {pct}% = margin {margin:.4f} "
        f"× {eff_lev}x = notional {notional:.4f} → qty {qty:.8g}")
    return qty


def _meets_minimums(ex: ccxt.Exchange, symbol: str, qty: float, price: float) -> bool:
    m = ex.market(symbol)
    limits = m.get("limits", {}) or {}
    min_amt = (limits.get("amount") or {}).get("min")
    min_cost = (limits.get("cost") or {}).get("min")
    notional = qty * price
    if min_amt is not None and qty < float(min_amt):
        log(f"  SKIP entry: qty {qty:.8g} < market min amount {min_amt} "
            f"(notional ≈ {notional:.2f}). Increase notional %% or leverage.")
        return False
    if min_cost is not None and notional < float(min_cost):
        log(f"  SKIP entry: notional {notional:.4f} < market min cost {min_cost}. "
            f"Increase notional %% or leverage.")
        return False
    return True


def _place_order(ex: ccxt.Exchange, symbol: str, side: str, qty: float,
                 price: float, dry_run: bool) -> None:
    if not _meets_minimums(ex, symbol, qty, price):
        return
    qty = ex.amount_to_precision(symbol, qty)
    if dry_run:
        log(f"DRY-RUN would submit MARKET {side.upper()} {qty} {symbol}")
        return
    log(f"SUBMIT MARKET {side.upper()} {qty} {symbol}")
    o = ex.create_order(symbol, "market", side, float(qty))
    log(f"  ✓ order id={o.get('id')} status={o.get('status')}")


def _set_leverage(ex: ccxt.Exchange, symbol: str, lev: float) -> None:
    try:
        ex.set_leverage(int(lev), symbol)
        log(f"Leverage set: {symbol} = {int(lev)}x")
    except Exception as e:
        log(f"WARN: set_leverage failed ({e}); continuing with exchange default.")


def main() -> None:
    p = argparse.ArgumentParser(description="Indicator-driven Binance auto-trader.")
    p.add_argument("--market", choices=["spot", "futures"], required=True)
    p.add_argument("--symbol", required=True, help="e.g. BTC/USDT")
    p.add_argument("--tf", required=True, help="Binance timeframe (1m, 5m, 1h, 4h, 1d, ...)")
    p.add_argument("--strategy", required=True, help=f"One of: {', '.join(STRATEGIES)}")
    p.add_argument("--params", default="{}", help="JSON of strategy params.")
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--notional-pct", type=float, required=True,
                   help="Notional per position as %% of available USDT balance "
                        "(e.g. 25 = use 25%% of free USDT). Re-evaluated each entry.")
    p.add_argument("--leverage", type=float, default=1.0, help="Futures only.")
    p.add_argument("--poll-seconds", type=int, default=0,
                   help="Polling interval. 0 = use bar duration.")
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="Log orders instead of submitting (default ON).")
    p.add_argument("--live", action="store_true",
                   help="Disable dry-run and submit real orders.")
    args = p.parse_args()

    if args.strategy not in STRATEGIES:
        log(f"FATAL: unknown strategy '{args.strategy}'.")
        sys.exit(2)

    dry_run = not args.live
    spec = STRATEGIES[args.strategy]
    try:
        params = json.loads(args.params)
    except Exception as e:
        log(f"FATAL: --params is not valid JSON ({e}).")
        sys.exit(2)
    params["allow_short"] = bool(args.allow_short)

    ex = _build_exchange(args.market)
    ex.load_markets()
    if args.market == "futures":
        _set_leverage(ex, args.symbol, args.leverage)

    tf_ms = ex.parse_timeframe(args.tf) * 1000
    poll = args.poll_seconds if args.poll_seconds > 0 else max(30, tf_ms // 1000 // 4)

    log("─" * 60)
    log(f"Auto-trader starting")
    log(f"  market    = {args.market}")
    log(f"  symbol    = {args.symbol}")
    log(f"  timeframe = {args.tf}")
    log(f"  strategy  = {args.strategy}  params={ {k:v for k,v in params.items() if k != 'allow_short'} }")
    log(f"  allow_short = {args.allow_short}")
    log(f"  notional  = {args.notional_pct}% of free USDT  leverage = {args.leverage}x")
    log(f"  poll      = every {poll}s")
    log(f"  mode      = {'DRY-RUN' if dry_run else '⚠ LIVE TRADING ⚠'}")
    log("─" * 60)

    last_bar_handled = None
    last_target = 0
    stop_requested = {"v": False}

    def _sig(_sig, _frame):
        stop_requested["v"] = True
        log("Signal received, will exit after current iteration.")
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    while not stop_requested["v"]:
        try:
            df = _fetch_bars(ex, args.symbol, args.tf, 400)
            if df.empty:
                log("WARN: empty OHLCV response; sleeping.")
                time.sleep(poll); continue
            last_bar = df.index[-1]
            price = float(df["close"].iloc[-1])

            positions = spec.func(df, **params)
            target = int(positions.iloc[-1]) if not positions.empty else 0
            current = _get_position_size(ex, args.market, args.symbol)
            cur_side = 0 if abs(current) < 1e-12 else (1 if current > 0 else -1)

            tag = "" if last_bar != last_bar_handled else "  (no new bar)"
            log(f"bar={last_bar:%Y-%m-%d %H:%M} px={price:.6g} target={target} held={cur_side}{tag}")

            if last_bar != last_bar_handled or target != last_target:
                # Plan transitions: close-if-needed, then open-if-needed.
                if cur_side != target:
                    if cur_side != 0:
                        side = "sell" if cur_side == 1 else "buy"
                        _place_order(ex, args.symbol, side, abs(current), price, dry_run)
                    if target != 0:
                        qty = _qty_for_pct(ex, args.market, price,
                                           args.notional_pct, args.leverage)
                        if qty <= 0:
                            log("  WARN: zero sizing (no free USDT?); skipping entry.")
                        else:
                            side = "buy" if target == 1 else "sell"
                            _place_order(ex, args.symbol, side, qty, price, dry_run)
                last_bar_handled = last_bar
                last_target = target

        except ccxt.NetworkError as e:
            log(f"NETWORK: {e}; will retry.")
        except ccxt.ExchangeError as e:
            log(f"EXCHANGE: {e}; will retry.")
        except Exception as e:
            log(f"ERROR: {type(e).__name__}: {e}")

        # Sleep in small chunks so SIGTERM is responsive.
        slept = 0
        while slept < poll and not stop_requested["v"]:
            time.sleep(1); slept += 1

    log("Auto-trader stopped.")


if __name__ == "__main__":
    main()
