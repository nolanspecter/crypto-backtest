"""Streamlit crypto backtesting app.

Run: streamlit run app.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data import fetch_ohlcv, get_binance_fees, get_top_symbols
from strategies import STRATEGIES
from backtest import run_backtest


st.set_page_config(page_title="Crypto 4H Backtester", page_icon="📈", layout="wide")
st.title("📈 Crypto 4H Backtester")
st.caption(
    "Top-100 crypto universe (CoinGecko), Binance OHLCV via CCXT, "
    "11 strategies across trend / mean-reversion / momentum / volatility."
)


def fmt_pct(x): return "—" if pd.isna(x) else f"{x*100:.2f}%"
def fmt_num(x): return "—" if pd.isna(x) else f"{x:.2f}"


# ===== Sidebar (shared controls) =====
with st.sidebar:
    st.header("Universe & Data")
    market_label = st.radio("Market", ["Spot", "Futures (USDⓂ Perp)"], index=0, horizontal=True)
    market = "futures" if market_label.startswith("Futures") else "spot"
    quote = st.selectbox("Quote currency", ["USDT", "USDC", "BUSD"], 0,
                         disabled=(market == "futures"),
                         help="USDⓂ perpetuals are quoted in USDT.")
    if market == "futures":
        quote = "USDT"
    with st.spinner(f"Loading top-100 {market} universe…"):
        try:
            universe = get_top_symbols(quote=quote, limit=100, market=market)
        except Exception as e:
            st.error(f"Failed to load universe: {e}")
            st.stop()
    if universe.empty:
        st.error("No symbols available.")
        st.stop()

    label_map = {
        f"#{int(r['rank']):>3}  {r['symbol']}  ·  {r['name']}": r["symbol"]
        for _, r in universe.iterrows()
    }

    st.header("Mode")
    mode = st.radio(
        "Backtest mode",
        ["Single strategy", "Find best strategy (all strategies × symbols)"],
        index=0,
    )

    BINANCE_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
                          "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
    tf_choice = st.selectbox("Timeframe", BINANCE_TIMEFRAMES + ["Custom…"],
                             index=BINANCE_TIMEFRAMES.index("4h"))
    if tf_choice == "Custom…":
        timeframe = st.text_input(
            "Custom timeframe", value="4h",
            help="Any Binance-supported timeframe: 1m, 3m, 5m, 15m, 30m, 1h, "
                 "2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.",
        ).strip()
    else:
        timeframe = tf_choice
    st.markdown("**Backtest period**")
    today = datetime.now(timezone.utc).date()
    default_start = today - timedelta(days=365 * 2)
    min_start = today - timedelta(days=365 * 5)
    date_col1, date_col2 = st.columns(2)
    start_date = date_col1.date_input(
        "Start", value=default_start, min_value=min_start, max_value=today - timedelta(days=1),
    )
    end_date = date_col2.date_input(
        "End", value=today, min_value=start_date + timedelta(days=1), max_value=today,
    )
    if end_date <= start_date:
        st.error("End date must be after start date.")
        st.stop()
    lookback_days = (end_date - start_date).days
    st.caption(f"➡ Lookback: **{lookback_days} days** ({start_date} → {end_date}).")

    # Costs depend on selected symbol (single) or the market default (tournament)
    if mode.startswith("Single"):
        labels = list(label_map.keys())
        label = st.selectbox(
            f"Symbol ({len(universe)} available)", labels, index=0,
            help="Click to open, then type to search (e.g. BTC, sol, doge).",
        )
        symbol = label_map[label]
        symbols_selected: list[str] = [symbol]
    else:
        defaults = list(label_map.keys())[:5]
        labels = st.multiselect(
            f"Symbols ({len(universe)} available)",
            list(label_map.keys()),
            default=defaults,
            help="Click to open, then type to search. Pick 1+ symbols.",
        )
        symbols_selected = [label_map[l] for l in labels]
        symbol = symbols_selected[0] if symbols_selected else next(iter(label_map.values()))

    st.header("Costs (live from Binance)")
    try:
        fees = get_binance_fees(symbol, market=market)
        binance_taker = fees["taker_bps"]
        binance_maker = fees["maker_bps"]
        st.caption(
            f"Binance {market} **{symbol}** — maker `{binance_maker:.2f} bps` · "
            f"taker `{binance_taker:.2f} bps`."
        )
    except Exception as e:
        binance_taker, binance_maker = (5.0, 2.0) if market == "futures" else (10.0, 10.0)
        st.warning(f"Could not fetch Binance fees ({e}); using defaults.")

    fee_mode = st.radio(
        "Fee model",
        ["Taker (market orders)", "Maker (limit orders)", "Custom"],
        index=0, horizontal=True,
    )
    if fee_mode.startswith("Taker"):
        fee_bps = float(binance_taker)
    elif fee_mode.startswith("Maker"):
        fee_bps = float(binance_maker)
    else:
        fee_bps = st.number_input("Fee (bps per side)", 0.0, 100.0, float(binance_taker), 0.5)

    slip_bps = st.number_input("Slippage (bps per side)", 0.0, 100.0, 5.0, 1.0,
                               help="Added on top of fees per fill.")
    st.caption(f"➡ Round-trip cost per turn: **{(fee_bps + slip_bps):.2f} bps** (charged on **notional**).")

    st.header("Position Sizing")
    if market == "futures":
        risk_pct = st.slider("Capital allocated per trade (%)", 1.0, 100.0, 5.0, 1.0,
                             help="Fraction of equity put up as margin. Default 5%.")
        leverage = st.slider("Leverage (×)", 1.0, 25.0, 1.0, 1.0,
                             help="Notional multiplier. 1× = no leverage.")
        position_size = risk_pct / 100.0
        notional_pct = position_size * leverage * 100
        st.caption(
            f"➡ Notional per trade: **{notional_pct:.1f}%** of equity "
            f"({risk_pct:.0f}% capital × {leverage:.0f}× leverage). "
            f"Fees are charged on this notional, not just capital."
        )
    else:
        risk_pct = st.slider("Capital allocated per trade (%)", 1.0, 100.0, 100.0, 1.0,
                             help="Spot default: 100% (fully invested when in a position).")
        leverage = 1.0
        position_size = risk_pct / 100.0
        st.caption(f"➡ Notional per trade: **{position_size * 100:.1f}%** of equity.")

    allow_short = st.checkbox("Allow shorting", value=False)

    # Defaults so the live-trade form works in every mode
    strat_name: str = next(iter(STRATEGIES))
    params: dict = {"allow_short": allow_short}

    # Strategy picker only in single mode
    if mode.startswith("Single"):
        st.header("Strategy")
        strat_name = st.selectbox("Strategy", list(STRATEGIES.keys()))
        spec = STRATEGIES[strat_name]
        st.caption(f"**{spec.family}** — {spec.description}")
        with st.expander("Entry / Exit / TP·SL rules", expanded=True):
            st.markdown(
                f"**Entry:** {spec.entry_rule}\n\n"
                f"**Exit (backtested):** {spec.exit_rule}\n\n"
                f"**Suggested TP / SL (not in backtest):** {spec.suggested_tp_sl}"
            )

        params: dict = {"allow_short": allow_short}
        for pname, p in spec.params.items():
            default, lo, hi, step = p
            if isinstance(default, float) or isinstance(step, float):
                params[pname] = st.slider(pname, float(lo), float(hi), float(default), float(step))
            else:
                params[pname] = st.slider(pname, int(lo), int(hi), int(default), int(step))
        run = st.button("▶ Run backtest", type="primary", use_container_width=True)
    else:
        st.header("Strategies")
        st.caption("All strategies run with **default parameters**.")
        rank_metric = st.selectbox(
            "Rank by",
            ["Total Return", "CAGR", "Sharpe", "Sortino", "Calmar", "Profit Factor", "R:R"],
            index=0,
        )
        run = st.button("▶ Run tournament", type="primary", use_container_width=True)

    st.divider()
    st.header("Live trading")
    if st.button("🚀 Start trading", use_container_width=True,
                 help="Opens a form to launch a real (or dry-run) auto-trader using your API keys."):
        st.session_state["show_trade_form"] = True
    if st.session_state.get("trader") and st.button(
        "■ Stop running trader", use_container_width=True
    ):
        try:
            os.kill(int(st.session_state["trader"]["pid"]), signal.SIGTERM)
            st.success("Sent SIGTERM to trader.")
        except ProcessLookupError:
            st.info("Trader already exited.")
        except Exception as e:
            st.error(f"Failed to stop: {e}")


# ===== Live trader form & status panel =====
_LOG_DIR = Path("/tmp")
_KEY_STORE = Path.home() / ".crypto-backtest" / "binance.json"


def _load_saved_keys() -> dict:
    """Best-effort load of {api_key, api_secret} from a local file."""
    try:
        if _KEY_STORE.exists():
            return json.loads(_KEY_STORE.read_text())
    except Exception:
        pass
    return {}


def _save_keys(api_key: str, api_secret: str) -> Path:
    """Persist keys to ~/.crypto-backtest/binance.json with mode 0600.

    Uses os.open with O_CREAT|O_WRONLY|O_TRUNC and mode 0o600 so the file is
    created with restrictive permissions from the start (no chmod race).
    """
    _KEY_STORE.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(_KEY_STORE.parent, 0o700)
    except OSError:
        pass
    payload = json.dumps({"api_key": api_key, "api_secret": api_secret})
    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
    fd = os.open(str(_KEY_STORE), flags, 0o600)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    return _KEY_STORE


def _delete_saved_keys() -> bool:
    try:
        if _KEY_STORE.exists():
            _KEY_STORE.unlink()
            return True
    except Exception:
        pass
    return False


def _trader_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _render_trade_form():
    st.subheader("🚀 Launch auto-trader")
    saved = _load_saved_keys()
    if saved:
        st.info(
            f"🗝 Loaded saved API keys from `{_KEY_STORE}`. "
            "Untick 'Remember keys' below before launching if you want them removed.",
            icon="ℹ️",
        )
    st.warning(
        "Your API keys are passed to the trader subprocess **via environment "
        "variables only** and then cleared from this app's session.\n\n"
        "If you tick **'Remember keys on this machine'**, the keys are written "
        f"in plaintext to `{_KEY_STORE}` (mode 0600). Anyone with read access "
        "to your home directory can recover them. Leave unticked for the safer "
        "behavior (no disk persistence).\n\n"
        "**For your protection:** create a Binance API key with **trading "
        "enabled but withdrawals disabled**, and IP-whitelist it.",
        icon="🔐",
    )
    # Prefill from saved keys exactly once (before the widget is instantiated).
    if saved:
        st.session_state.setdefault("_trade_api_key", saved.get("api_key", ""))
        st.session_state.setdefault("_trade_api_secret", saved.get("api_secret", ""))

    # NB: Not using st.form so changing Strategy reruns the param widgets
    # immediately. We use explicit keys for the secret inputs so we can
    # delete them from session_state after launch.
    c1, c2 = st.columns(2)
    api_key = c1.text_input(
        "Binance API key", type="password", autocomplete="off",
        key="_trade_api_key",
    )
    api_secret = c2.text_input(
        "Binance API secret", type="password", autocomplete="off",
        key="_trade_api_secret",
    )
    c1, c2 = st.columns([3, 1])
    remember = c1.checkbox(
        "💾 Remember keys on this machine",
        value=bool(saved),
        help=f"Saves to {_KEY_STORE} (mode 0600). Plaintext on disk — see warning above.",
        key="_trade_remember_keys",
    )
    if saved and c2.button("🗑 Forget saved keys"):
        if _delete_saved_keys():
            for k in ("_trade_api_key", "_trade_api_secret"):
                st.session_state.pop(k, None)
            st.success(f"Removed {_KEY_STORE}.")
            st.rerun()

    c1, c2, c3 = st.columns(3)
    f_market = c1.selectbox("Market", ["spot", "futures"],
                            index=0 if market == "spot" else 1)
    f_symbol = c2.text_input("Symbol", value=symbols_selected[0] if symbols_selected else "BTC/USDT")
    f_tf = c3.selectbox("Timeframe", BINANCE_TIMEFRAMES,
                        index=BINANCE_TIMEFRAMES.index(timeframe) if timeframe in BINANCE_TIMEFRAMES else BINANCE_TIMEFRAMES.index("1h"))

    c1, c2, c3 = st.columns(3)
    f_strat = c1.selectbox(
        "Strategy", list(STRATEGIES.keys()),
        index=(list(STRATEGIES.keys()).index(strat_name)
               if mode.startswith("Single") else 0),
        key="_trade_strat",
    )
    f_notional_pct = c2.number_input(
        "Notional per trade (% of free USDT)",
        min_value=1.0, max_value=100.0, value=25.0, step=1.0,
        help="Re-evaluated on each entry from the live USDT balance.",
    )
    f_lev = c3.number_input("Leverage (futures only)", min_value=1.0, max_value=25.0,
                            value=float(leverage), step=1.0)

    spec_for_trade = STRATEGIES[f_strat]
    st.caption(f"**Entry:** {spec_for_trade.entry_rule}")
    param_vals: dict = {}
    cols = st.columns(max(1, len(spec_for_trade.params)))
    for (pname, (default, lo, hi, step)), col in zip(spec_for_trade.params.items(), cols):
        # Key by strategy so switching strategy rebuilds widgets cleanly
        # instead of reusing stale numeric values from another strategy.
        wkey = f"_trade_param_{f_strat}_{pname}"
        if isinstance(default, float) or isinstance(step, float):
            src = params.get(pname) if mode.startswith("Single") and strat_name == f_strat else default
            param_vals[pname] = col.number_input(pname, float(lo), float(hi),
                                                 float(src), float(step), key=wkey)
        else:
            src = params.get(pname) if mode.startswith("Single") and strat_name == f_strat else default
            param_vals[pname] = col.number_input(pname, int(lo), int(hi),
                                                 int(src), int(step), key=wkey)

    c1, c2 = st.columns(2)
    f_allow_short = c1.checkbox("Allow shorting", value=bool(allow_short))
    f_live = c2.checkbox(
        "⚠️ LIVE — submit real orders",
        value=False,
        help="Unchecked = dry-run (default). Logs the orders it would place but submits nothing.",
    )

    c1, c2 = st.columns(2)
    submit = c1.button("▶ Launch trader", type="primary",
                       use_container_width=True, key="_trade_submit")
    cancel = c2.button("Cancel", use_container_width=True, key="_trade_cancel")

    if cancel:
        for k in ("_trade_api_key", "_trade_api_secret"):
            st.session_state.pop(k, None)
        st.session_state["show_trade_form"] = False
        st.rerun()

    if not submit:
        return

    if not api_key or not api_secret:
        st.error("Both API key and secret are required.")
        return

    cmd = [
        sys.executable, "-u", "trade.py",
        "--market", f_market, "--symbol", f_symbol, "--tf", f_tf,
        "--strategy", f_strat, "--params", json.dumps(param_vals),
        "--notional-pct", str(f_notional_pct), "--leverage", str(f_lev),
    ]
    if f_allow_short:
        cmd.append("--allow-short")
    if f_live:
        cmd.append("--live")

    # Persist (or delete) the key file BEFORE wiping locals.
    if remember:
        path = _save_keys(api_key, api_secret)
        st.toast(f"🔐 Keys saved to {path} (mode 0600).", icon="🔐")
    else:
        if _delete_saved_keys():
            st.toast(f"🗑 Removed {_KEY_STORE}.", icon="🗑")

    env = os.environ.copy()
    env["BINANCE_API_KEY"] = api_key
    env["BINANCE_API_SECRET"] = api_secret

    log_path = _LOG_DIR / f"auto_trader_{int(datetime.now().timestamp())}.log"
    # Open the log file, hand its FD to the child, then close it in the parent
    # so Streamlit holds no reference. start_new_session detaches the child
    # from Streamlit's process group / controlling terminal so SIGHUP / Ctrl-C
    # on the Streamlit process won't propagate. stdin → /dev/null so the child
    # never blocks on tty input.
    log_f = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            cmd, env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_f.close()  # parent FD released; child has its own dup.

    # ⚠ Wipe keys from local memory and Streamlit session_state immediately.
    # (Python doesn't guarantee the underlying string buffer is zeroed, but at
    # least no reference is held by the app process beyond this point.)
    api_key = api_secret = None
    for k in ("_trade_api_key", "_trade_api_secret"):
        st.session_state.pop(k, None)
    del env  # the subprocess has its own copy now.

    st.session_state["trader"] = {
        "pid": proc.pid,
        "log_path": str(log_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "LIVE" if f_live else "DRY-RUN",
        "symbol": f_symbol, "strategy": f_strat, "market": f_market, "tf": f_tf,
    }
    st.session_state["show_trade_form"] = False
    st.success(
        f"Trader launched (PID {proc.pid}). Log: `{log_path}`\n\n"
        "ℹ️ The trader runs in its own session — **closing or restarting "
        "this Streamlit app will NOT stop it**. Use the **■ Stop running "
        "trader** button (sidebar) or `kill " + str(proc.pid) + "` to stop."
    )
    st.rerun()


_AUTO_REFRESH_CHOICES = {"Off": None, "1s": 1, "3s": 3, "5s": 5, "10s": 10}


def _render_trader_status():
    t = st.session_state.get("trader")
    if not t:
        return
    st.subheader("📡 Auto-trader status")
    interval_label = st.selectbox(
        "Auto-refresh", list(_AUTO_REFRESH_CHOICES.keys()),
        index=2, key="_trader_refresh_interval",
        help="How often to re-poll the log file and re-render this panel.",
    )
    interval = _AUTO_REFRESH_CHOICES[interval_label]

    def _body():
        pid = int(t["pid"])
        alive = _trader_alive(pid)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PID", str(pid))
        c2.metric("Status", "🟢 running" if alive else "⚪ stopped")
        c3.metric("Mode", t["mode"])
        c4.metric("Symbol", t["symbol"])
        c5.metric("Strategy", t["strategy"])
        st.caption(
            f"Started: {t['started_at']}  ·  Market: {t['market']}  ·  "
            f"TF: {t['tf']}  ·  Log: `{t['log_path']}`  ·  "
            f"Tail tip: `tail -f {t['log_path']}` in a local terminal"
        )
        try:
            tail = Path(t["log_path"]).read_text(errors="replace").splitlines()[-300:]
            st.code("\n".join(tail) or "(no output yet)", language="text")
        except FileNotFoundError:
            st.info("Log file not yet created.")
        cA, cB = st.columns(2)
        if cA.button("🔄 Refresh now", key="_trader_refresh_now"):
            st.rerun()
        if alive and cB.button("■ Send SIGTERM", key="_trader_sigterm"):
            try:
                os.kill(pid, signal.SIGTERM)
                st.success("Sent SIGTERM.")
            except Exception as e:
                st.error(f"Failed: {e}")

    # st.fragment(run_every=N) re-runs only this panel every N seconds without
    # re-running the whole app. Falls back to a manual-refresh button on older
    # Streamlit versions.
    fragment = getattr(st, "fragment", None)
    if fragment is None:
        _body()
        return
    if interval is None:
        fragment(_body)()
    else:
        fragment(_body, run_every=interval)()


if st.session_state.get("show_trade_form"):
    _render_trade_form()
_render_trader_status()


# ===== Landing screen =====
if not run:
    st.info("Configure your test in the sidebar and click **Run**.")
    st.subheader("Available strategies")
    st.dataframe(
        pd.DataFrame([
            {
                "Strategy": s.name,
                "Family": s.family,
                "Entry": s.entry_rule,
                "Exit (backtested)": s.exit_rule,
                "Suggested TP / SL": s.suggested_tp_sl,
            }
            for s in STRATEGIES.values()
        ]),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Backtester uses **signal-based exits** (e.g. opposite cross, midline revert). "
        "*Suggested TP / SL* is reference for live trading — not applied in the backtest."
    )
    st.stop()


# Common: convert date range to since_ms / until_ts (UTC midnights)
start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
end_dt = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
since_ms = int(start_dt.timestamp() * 1000)


def _trim_to_end(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.index <= end_dt] if not df.empty else df


# ===== Mode 1: Single strategy =====
if mode.startswith("Single"):
    with st.spinner(f"Fetching {market} {symbol} {timeframe} candles…"):
        try:
            df = _trim_to_end(fetch_ohlcv(symbol, timeframe=timeframe, since_ms=since_ms, market=market))
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()

    if len(df) < 100:
        st.error(f"Not enough data ({len(df)} bars). Try a longer lookback.")
        st.stop()

    st.success(f"Loaded {len(df):,} {timeframe} bars · {df.index[0].date()} → {df.index[-1].date()}")

    positions = spec.func(df, **params)
    result = run_backtest(df, positions, fee_bps=fee_bps, slippage_bps=slip_bps,
                          position_size=position_size, leverage=leverage)


    m = result.metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return", fmt_pct(m["Total Return"]), f"B&H: {fmt_pct(m['Buy & Hold Return'])}")
    c2.metric("CAGR", fmt_pct(m["CAGR"]))
    c3.metric("Sharpe", fmt_num(m["Sharpe"]))
    c4.metric("Max DD", fmt_pct(m["Max Drawdown"]))
    c5.metric("# Trades", str(m["# Trades"]))
    c6, c7, c8, c9, c10, c11 = st.columns(6)
    c6.metric("Sortino", fmt_num(m["Sortino"]))
    c7.metric("Calmar", fmt_num(m["Calmar"]))
    c8.metric("Win Rate", fmt_pct(m["Win Rate"]))
    c9.metric("Profit Factor", fmt_num(m["Profit Factor"]))
    c10.metric("R:R", fmt_num(m["R:R"]),
               f"avg win {fmt_pct(m['Avg Win'])} / loss {fmt_pct(m['Avg Loss'])}")
    c11.metric("Exposure", fmt_pct(m["Exposure"]))

    bh_equity = (df["close"] / df["close"].iloc[0])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.04,
                        subplot_titles=("Equity (strategy vs buy & hold)", "Price with position shading"))
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values,
                             name="Strategy", line=dict(color="#2E86DE", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity.values,
                             name="Buy & Hold", line=dict(color="#888", width=1, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="Price",
                                 showlegend=False), row=2, col=1)

    held = result.positions
    def _shade(mask, color):
        in_block = False
        start = None
        for ts, v in mask.items():
            if v and not in_block:
                start = ts; in_block = True
            elif not v and in_block:
                fig.add_vrect(x0=start, x1=ts, fillcolor=color, opacity=0.12,
                              line_width=0, row=2, col=1)
                in_block = False
        if in_block:
            fig.add_vrect(x0=start, x1=mask.index[-1], fillcolor=color, opacity=0.12,
                          line_width=0, row=2, col=1)
    _shade((held == 1), "#26a69a")
    _shade((held == -1), "#ef5350")

    fig.update_layout(height=720, xaxis_rangeslider_visible=False,
                      xaxis2_rangeslider_visible=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"Trade ledger ({len(result.trades)} trades)", expanded=False):
        if result.trades.empty:
            st.write("No completed trades.")
        else:
            t = result.trades.copy()
            t["return"] = (t["return"] * 100).round(3)
            st.dataframe(t, use_container_width=True, hide_index=True)
            st.download_button("Download trades CSV", result.trades.to_csv(index=False).encode(),
                               file_name=f"{symbol.replace('/', '')}_{strat_name}_trades.csv",
                               mime="text/csv")
    with st.expander("Equity curve CSV"):
        eq = pd.DataFrame({"equity": result.equity, "returns": result.returns,
                           "position": result.positions})
        st.download_button("Download equity CSV", eq.to_csv().encode(),
                           file_name=f"{symbol.replace('/', '')}_{strat_name}_equity.csv",
                           mime="text/csv")

# ===== Mode 2: Find best strategy =====
else:
    if not symbols_selected:
        st.error("Pick at least one symbol.")
        st.stop()

    st.subheader(f"🏆 Strategy tournament — {len(symbols_selected)} symbol(s) × {len(STRATEGIES)} strategies")
    st.caption(f"**Lookback:** {lookback_days} days  ·  **Timeframe:** {timeframe}")

    # Fetch all symbols' data once
    data: dict[str, pd.DataFrame] = {}
    prog = st.progress(0.0, text="Fetching data…")
    for i, sym in enumerate(symbols_selected, 1):
        try:
            d = _trim_to_end(fetch_ohlcv(sym, timeframe=timeframe, since_ms=since_ms, market=market))
            if len(d) >= 100:
                data[sym] = d
            else:
                st.warning(f"Skipping {sym}: only {len(d)} bars.")
        except Exception as e:
            st.warning(f"Skipping {sym}: {e}")
        prog.progress(i / len(symbols_selected), text=f"Fetching {sym}…")
    prog.empty()

    if not data:
        st.error("No usable data for the selected symbols.")
        st.stop()

    # Run all strategies on each symbol
    rows: list[dict] = []
    equity_curves: dict[tuple[str, str], pd.Series] = {}  # (symbol, strategy) -> equity
    total = len(data) * len(STRATEGIES)
    prog = st.progress(0.0, text="Running backtests…")
    step = 0
    for sym, df in data.items():
        for sname, sp in STRATEGIES.items():
            step += 1
            prog.progress(step / total, text=f"{sym} · {sname}")
            try:
                default_params = {k: v[0] for k, v in sp.params.items()}
                pos = sp.func(df, allow_short=allow_short, **default_params)
                res = run_backtest(df, pos, fee_bps=fee_bps, slippage_bps=slip_bps,
                                   position_size=position_size, leverage=leverage)
                m = res.metrics
                rows.append({
                    "Symbol": sym, "Strategy": sname, "Family": sp.family,
                    "Total Return": m["Total Return"], "CAGR": m["CAGR"],
                    "Buy & Hold Return": m["Buy & Hold Return"],
                    "Sharpe": m["Sharpe"], "Sortino": m["Sortino"],
                    "Calmar": m["Calmar"], "Max DD": m["Max Drawdown"],
                    "Profit Factor": m["Profit Factor"], "R:R": m["R:R"],
                    "Win Rate": m["Win Rate"],
                    "# Trades": m["# Trades"], "Exposure": m["Exposure"],
                })
                equity_curves[(sym, sname)] = res.equity
            except Exception as e:
                rows.append({"Symbol": sym, "Strategy": sname, "Family": sp.family,
                             "Total Return": float("nan"), "error": str(e)})
    prog.empty()

    results = pd.DataFrame(rows)

    # ===== Best (symbol × strategy) combination by profit =====
    profit_results = results.dropna(subset=["Total Return"])
    if not profit_results.empty:
        best = profit_results.sort_values("Total Return", ascending=False).iloc[0]
        st.subheader("🏆 Best (symbol × indicator) combination by total profit")
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        b1.metric("Symbol", best["Symbol"])
        b2.metric("Indicator", best["Strategy"])
        b3.metric("Total Return", fmt_pct(best["Total Return"]),
                  f"B&H: {fmt_pct(best['Buy & Hold Return'])}")
        b4.metric("CAGR", fmt_pct(best["CAGR"]))
        b5.metric("Sharpe", fmt_num(best["Sharpe"]))
        b6.metric("Max DD", fmt_pct(best["Max DD"]))
        eq = equity_curves.get((best["Symbol"], best["Strategy"]))
        if eq is not None:
            best_fig = go.Figure()
            best_fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy",
                                          line=dict(color="#2E86DE", width=2)))
            bh_df = data[best["Symbol"]]
            bh_eq = bh_df["close"] / bh_df["close"].iloc[0]
            best_fig.add_trace(go.Scatter(x=bh_eq.index, y=bh_eq.values,
                                          name="Buy & Hold",
                                          line=dict(color="#888", width=1, dash="dot")))
            best_fig.update_layout(height=340, margin=dict(t=20, b=20),
                                   yaxis_title="Equity (start = 1.0)",
                                   title=f"{best['Strategy']} on {best['Symbol']}")
            st.plotly_chart(best_fig, use_container_width=True)

        # Top-10 leaderboard of (symbol × indicator) pairs
        with st.expander("Top 10 (symbol × indicator) combinations by profit", expanded=False):
            top10 = profit_results.sort_values("Total Return", ascending=False).head(10)
            top10_disp = top10[["Symbol", "Strategy", "Family", "Total Return", "CAGR",
                                "Buy & Hold Return", "Sharpe", "Max DD", "Profit Factor",
                                "Win Rate", "# Trades"]].copy()
            for c in ["Total Return", "CAGR", "Buy & Hold Return", "Max DD", "Win Rate"]:
                top10_disp[c] = top10_disp[c].map(fmt_pct)
            for c in ["Sharpe", "Profit Factor"]:
                top10_disp[c] = top10_disp[c].map(fmt_num)
            st.dataframe(top10_disp, use_container_width=True, hide_index=True)

    # Per-symbol best
    st.subheader("🥇 Best strategy per symbol")
    best_per_sym = (results.sort_values(rank_metric, ascending=False)
                          .groupby("Symbol", as_index=False).first())
    show_cols = ["Symbol", "Strategy", "Family", "Total Return", "CAGR",
                 "Buy & Hold Return", "Sharpe", "Max DD", "Profit Factor", "R:R",
                 "Win Rate", "# Trades"]
    disp = best_per_sym[show_cols].copy()
    for c in ["Total Return", "CAGR", "Buy & Hold Return", "Max DD", "Win Rate"]:
        disp[c] = disp[c].map(fmt_pct)
    for c in ["Sharpe", "Profit Factor", "R:R"]:
        disp[c] = disp[c].map(fmt_num)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # Aggregate across symbols
    st.subheader(f"📊 Strategy leaderboard (averaged across {len(data)} symbol(s), ranked by {rank_metric})")
    agg = (results.groupby(["Strategy", "Family"], as_index=False)
                  .agg({"Total Return": "mean", "CAGR": "mean",
                        "Sharpe": "mean", "Sortino": "mean", "Calmar": "mean",
                        "Max DD": "mean", "Profit Factor": "mean", "R:R": "mean",
                        "Win Rate": "mean", "# Trades": "mean"})
                  .sort_values(rank_metric, ascending=False))
    agg_disp = agg.copy()
    for c in ["Total Return", "CAGR", "Max DD", "Win Rate"]:
        agg_disp[c] = agg_disp[c].map(fmt_pct)
    for c in ["Sharpe", "Sortino", "Calmar", "Profit Factor", "R:R"]:
        agg_disp[c] = agg_disp[c].map(fmt_num)
    agg_disp["# Trades"] = agg_disp["# Trades"].round(0).astype(int)
    st.dataframe(agg_disp, use_container_width=True, hide_index=True)

    winner = agg.iloc[0]["Strategy"]
    st.success(f"🏆 Overall winner by **{rank_metric}**: **{winner}** "
               f"(mean {rank_metric}: {fmt_pct(agg.iloc[0][rank_metric]) if rank_metric in ['Total Return','CAGR','Max DD','Win Rate'] else fmt_num(agg.iloc[0][rank_metric])})")

    # Equity curves of the winner across all symbols
    st.subheader(f"Equity curves — {winner}")
    fig = go.Figure()
    for sym in data.keys():
        eq = equity_curves.get((sym, winner))
        if eq is not None:
            fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=sym, mode="lines"))
    fig.update_layout(height=450, margin=dict(t=30, b=20),
                      yaxis_title="Equity (start = 1.0)")
    st.plotly_chart(fig, use_container_width=True)

    # Full results table + download
    with st.expander(f"All results ({len(results)} rows)", expanded=False):
        full = results.copy()
        for c in ["Total Return", "CAGR", "Buy & Hold Return", "Max DD",
                  "Win Rate", "Exposure"]:
            if c in full:
                full[c] = full[c].map(fmt_pct)
        for c in ["Sharpe", "Sortino", "Calmar", "Profit Factor", "R:R"]:
            if c in full:
                full[c] = full[c].map(fmt_num)
        st.dataframe(full, use_container_width=True, hide_index=True)
        st.download_button("Download full results CSV", results.to_csv(index=False).encode(),
                           file_name="tournament_results.csv", mime="text/csv")
