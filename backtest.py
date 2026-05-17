"""Vectorized backtester. Apply positions at next bar's open with fees & slippage."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series              # equity curve, starting at 1.0
    returns: pd.Series             # per-bar strategy returns (net of costs)
    positions: pd.Series           # position held during each bar (-1/0/1)
    trades: pd.DataFrame           # one row per round-trip
    bars_per_year: float
    metrics: dict

    def as_dict(self) -> dict:
        return asdict(self)


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    deltas = np.diff(index.view("i8")) / 1e9  # seconds
    median = float(np.median(deltas))
    if median <= 0:
        return 252.0
    return 365.25 * 24 * 3600 / median


def _trades_from_positions(positions: pd.Series, exec_price: pd.Series,
                           bar_returns_gross: pd.Series) -> pd.DataFrame:
    """Build a trade ledger. A trade starts when position changes from 0 to ±1
    (or flips sign) and ends on the next change.
    """
    pos = positions.values
    px = exec_price.values
    idx = positions.index
    trades = []
    entry_i = None
    entry_side = 0
    for i in range(len(pos)):
        cur = pos[i]
        prev = pos[i - 1] if i > 0 else 0
        if cur != prev:
            if entry_i is not None:
                # close prior trade at this bar's exec price
                exit_px = px[i]
                entry_px = px[entry_i]
                ret = entry_side * (exit_px / entry_px - 1)
                trades.append({
                    "entry_time": idx[entry_i], "exit_time": idx[i],
                    "side": "long" if entry_side == 1 else "short",
                    "entry_px": entry_px, "exit_px": exit_px,
                    "bars": i - entry_i, "return": ret,
                })
                entry_i = None
                entry_side = 0
            if cur != 0:
                entry_i = i
                entry_side = int(cur)
    # close any open trade at the last bar
    if entry_i is not None:
        i = len(pos) - 1
        if i > entry_i:
            exit_px = px[i]
            entry_px = px[entry_i]
            ret = entry_side * (exit_px / entry_px - 1)
            trades.append({
                "entry_time": idx[entry_i], "exit_time": idx[i],
                "side": "long" if entry_side == 1 else "short",
                "entry_px": entry_px, "exit_px": exit_px,
                "bars": i - entry_i, "return": ret,
            })
    return pd.DataFrame(trades)


def run_backtest(df: pd.DataFrame, positions: pd.Series,
                 fee_bps: float = 10.0, slippage_bps: float = 5.0,
                 position_size: float = 1.0,
                 leverage: float = 1.0) -> BacktestResult:
    """`positions` is the desired position (sign) computed using info up to &
    including each bar's close. We execute the change at the NEXT bar's open.

    `position_size` is the fraction of equity allocated per trade (0–1). For
    example, 0.05 means only 5% of capital is risked at a time (the rest is
    held in cash). `leverage` multiplies notional exposure on top of that and
    is typically only meaningful for futures.
    """
    if positions.empty:
        raise ValueError("empty positions")

    # Shift so the position held during bar t was decided at the close of bar t-1.
    sign = positions.shift(1).fillna(0).astype(int)
    held = sign.astype(float) * float(position_size) * float(leverage)
    op = df["open"].astype(float)
    bar_ret_gross = op.pct_change().fillna(0.0)

    # ── Cost accounting (still per-bar, since costs are paid at fills) ──────
    # Fees are charged on |notional change|, NOT on capital at risk. `held`
    # is in units of "fraction of equity in notional terms" so |Δheld| is
    # the notional turnover. Example: 5% capital × 10× lev = 0.5 notional;
    # opening costs 0.5 × 5 bps = 2.5 bps of equity per side.
    pos_change = held.diff().abs().fillna(held.abs())
    cost_per_turn = (fee_bps + slippage_bps) / 1e4
    costs = pos_change * cost_per_turn
    total_notional_turnover = float(pos_change.sum())
    total_fee_drag = float(costs.sum())

    # ── Trade ledger first (closed round-trips) ─────────────────────────────
    trades = _trades_from_positions(sign, op, bar_ret_gross)
    bpy = _bars_per_year(df.index)

    # ── Equity, computed PER TRADE not per bar ──────────────────────────────
    # Why: per-bar `prod(1 + L·r_t)` compounds leverage on every bar inside a
    # trade, blowing up under L > 1. The realistic close-to-close P&L of a
    # single trade is L × (exit/entry − 1), and only that should compound
    # across trades. This matches TradingView's strategy.entry semantics and
    # what an actual fill-by-fill broker statement looks like.
    leverage_factor = float(position_size) * float(leverage)
    cost_per_trade = 2.0 * leverage_factor * cost_per_turn  # entry + exit cost
    if not trades.empty:
        gross_per_trade = trades["return"].values * leverage_factor
        net_per_trade = gross_per_trade - cost_per_trade
        # Cannot lose more than 100% of equity (margin call hard cap).
        net_per_trade = np.maximum(net_per_trade, -1.0)
        compounded = np.cumprod(1.0 + net_per_trade)

        # Bar-indexed equity that STEPS at each trade exit and is flat
        # between trades (the chart becomes a stair-step curve — that's
        # the realised-P&L view, no MTM wiggles).
        equity = pd.Series(1.0, index=df.index)
        for i, exit_time in enumerate(trades["exit_time"]):
            equity.loc[equity.index >= exit_time] = float(compounded[i])

        # Per-bar return series is zero except on exit bars; metrics that
        # used to be bar-frequency (Sharpe/Sortino/Vol) now use the
        # per-trade return distribution annualised by trades-per-year.
        strat_ret = equity.pct_change().fillna(0.0)
        n_years = max(
            (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 86400),
            1e-9,
        )
        trades_per_year = len(net_per_trade) / n_years
        trade_ret_std = float(np.std(net_per_trade, ddof=1)) if len(net_per_trade) > 1 else 0.0
        trade_ret_mean = float(np.mean(net_per_trade))
        # Annotate the ledger with the leveraged/net numbers the equity is built from.
        trades = trades.copy()
        trades["leveraged_return"] = gross_per_trade
        trades["net_return"] = net_per_trade
    else:
        equity = pd.Series(1.0, index=df.index)
        strat_ret = pd.Series(0.0, index=df.index)
        trade_ret_std = 0.0
        trade_ret_mean = 0.0
        trades_per_year = 0.0

    # Metrics — all derived from the per-trade compounded equity series.
    total_return = float(equity.iloc[-1] - 1)
    n_years = (df.index[-1] - df.index[0]).total_seconds() / (365.25 * 86400)
    cagr = (
        float(equity.iloc[-1] ** (1 / n_years) - 1)
        if n_years and n_years > 0 and equity.iloc[-1] > 0
        else np.nan
    )
    # Per-trade vol / Sharpe / Sortino, annualised by trades/year.
    if trades_per_year > 0 and trade_ret_std > 0:
        vol = float(trade_ret_std * np.sqrt(trades_per_year))
        sharpe = float(trade_ret_mean / trade_ret_std * np.sqrt(trades_per_year))
        neg = (
            trades["net_return"][trades["net_return"] < 0]
            if "net_return" in trades.columns else pd.Series(dtype=float)
        )
        downside = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
        sortino = float(trade_ret_mean / downside * np.sqrt(trades_per_year)) if downside else np.nan
    else:
        vol = sharpe = sortino = np.nan
    rollmax = equity.cummax()
    dd = (equity / rollmax - 1)
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 and not pd.isna(cagr) else np.nan

    bh_ret = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)

    # Median bar duration in seconds (e.g. 4h bars → 14400). Used to convert
    # "bars held" into wall-clock time so the average hold reads in hours/days
    # regardless of timeframe.
    if len(df.index) >= 2:
        bar_seconds = float(np.median(np.diff(df.index.view("i8")) / 1e9))
    else:
        bar_seconds = 0.0

    if not trades.empty:
        # Use the LEVERAGED + NET-OF-COSTS return so the stats match what the
        # equity curve actually realises trade-by-trade.
        ret_col = "net_return" if "net_return" in trades.columns else "return"
        wins = trades[trades[ret_col] > 0][ret_col]
        losses = trades[trades[ret_col] <= 0][ret_col]
        win_rate = float(len(wins) / len(trades))
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else np.inf
        expectancy = float(trades[ret_col].mean())
        rr_ratio = float(avg_win / abs(avg_loss)) if avg_loss < 0 else np.nan
        avg_hold_bars = float(trades["bars"].mean())
        avg_hold_seconds = avg_hold_bars * bar_seconds
        # Per-side averages, useful when allow_short is on
        long_trades = trades[trades["side"] == "long"]
        short_trades = trades[trades["side"] == "short"]
        avg_hold_long_bars = float(long_trades["bars"].mean()) if not long_trades.empty else np.nan
        avg_hold_short_bars = float(short_trades["bars"].mean()) if not short_trades.empty else np.nan
    else:
        win_rate = avg_win = avg_loss = expectancy = 0.0
        profit_factor = rr_ratio = np.nan
        avg_hold_bars = avg_hold_seconds = np.nan
        avg_hold_long_bars = avg_hold_short_bars = np.nan

    exposure = float((held != 0).mean())

    metrics = {
        "Total Return": total_return,
        "CAGR": cagr,
        "Buy & Hold Return": bh_ret,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Volatility (ann.)": vol,
        "Exposure": exposure,
        "# Trades": int(len(trades)),
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "R:R": rr_ratio,
        "Expectancy / Trade": expectancy,
        "Avg Hold (bars)": avg_hold_bars,
        "Avg Hold (seconds)": avg_hold_seconds,
        "Avg Hold Long (bars)": avg_hold_long_bars,
        "Avg Hold Short (bars)": avg_hold_short_bars,
    }

    metrics["Position Size"] = float(position_size)
    metrics["Leverage"] = float(leverage)
    metrics["Fee bps/side"] = float(fee_bps)
    metrics["Slippage bps/side"] = float(slippage_bps)
    metrics["Cost bps/turn (per side)"] = float(fee_bps + slippage_bps)
    metrics["Notional Turnover (× equity)"] = total_notional_turnover
    metrics["Total Fee Drag"] = total_fee_drag

    return BacktestResult(equity=equity, returns=strat_ret, positions=sign,
                          trades=trades, bars_per_year=bpy, metrics=metrics)
