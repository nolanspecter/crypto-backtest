"""Strategy library. Each strategy takes an OHLCV DataFrame and parameter dict,
and returns a pandas Series of target positions in {-1, 0, 1} indexed by time.

Positions are intended to be applied at the NEXT bar's open (see backtest.py),
so signals can be computed using close-of-bar information.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ---------- Indicator helpers ----------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    m = sma(s, n)
    sd = s.rolling(n).std(ddof=0)
    return m, m + k * sd, m - k * sd


def keltner(df: pd.DataFrame, n: int = 20, k: float = 2.0):
    m = ema(df["close"], n)
    a = atr(df, n)
    return m, m + k * a, m - k * a


def donchian(df: pd.DataFrame, n: int = 20):
    upper = df["high"].rolling(n).max()
    lower = df["low"].rolling(n).min()
    return upper, lower


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """Return +1 (uptrend) / -1 (downtrend) for each bar."""
    a = atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    dir_ = pd.Series(index=df.index, dtype=float)
    upper_f = upper.copy()
    lower_f = lower.copy()
    close = df["close"].values
    u = upper.values
    l = lower.values
    d = np.zeros(len(df))
    d[0] = 1
    for i in range(1, len(df)):
        u[i] = min(u[i], u[i - 1]) if close[i - 1] <= u[i - 1] else u[i]
        l[i] = max(l[i], l[i - 1]) if close[i - 1] >= l[i - 1] else l[i]
        if close[i] > u[i - 1]:
            d[i] = 1
        elif close[i] < l[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
    return pd.Series(d, index=df.index)


# ---------- Strategies ----------
# Each returns positions in {-1, 0, +1}. allow_short controls whether to flip
# to short or just go flat on exit.

def _to_position(long_sig: pd.Series, short_sig: pd.Series, allow_short: bool) -> pd.Series:
    pos = pd.Series(0, index=long_sig.index, dtype=int)
    pos = pos.where(~long_sig, 1)
    if allow_short:
        pos = pos.where(~short_sig, -1)
    return pos


def strat_ema_cross(df, fast=20, slow=50, allow_short=False, **_):
    f, s = ema(df["close"], fast), ema(df["close"], slow)
    long_sig = f > s
    short_sig = f < s
    return _to_position(long_sig, short_sig, allow_short)


def strat_supertrend(df, period=10, mult=3.0, allow_short=False, **_):
    d = supertrend(df, period, mult)
    long_sig = d == 1
    short_sig = d == -1
    return _to_position(long_sig, short_sig, allow_short)


def strat_donchian(df, n=20, allow_short=False, **_):
    upper, lower = donchian(df, n)
    # Breakout entries; stay until opposite breakout (turtle-style).
    long_entry = df["close"] > upper.shift(1)
    short_entry = df["close"] < lower.shift(1)
    pos = pd.Series(np.nan, index=df.index)
    pos[long_entry] = 1
    if allow_short:
        pos[short_entry] = -1
    else:
        pos[short_entry] = 0
    pos = pos.ffill().fillna(0).astype(int)
    return pos


def strat_rsi_meanrev(df, n=14, oversold=30, overbought=70, exit_mid=50, allow_short=False, **_):
    r = rsi(df["close"], n)
    pos = pd.Series(0, index=df.index, dtype=int)
    state = 0
    out = []
    for v in r.values:
        if np.isnan(v):
            out.append(0)
            continue
        if state == 0:
            if v < oversold:
                state = 1
            elif allow_short and v > overbought:
                state = -1
        elif state == 1 and v >= exit_mid:
            state = 0
        elif state == -1 and v <= exit_mid:
            state = 0
        out.append(state)
    return pd.Series(out, index=df.index, dtype=int)


def strat_bbands_meanrev(df, n=20, k=2.0, allow_short=False, **_):
    m, up, lo = bollinger(df["close"], n, k)
    c = df["close"]
    pos = pd.Series(np.nan, index=df.index)
    pos[c < lo] = 1
    pos[c > m] = 0  # exit longs at midline
    if allow_short:
        pos2 = pd.Series(np.nan, index=df.index)
        pos2[c > up] = -1
        pos2[c < m] = 0
        # combine: prefer most recent signal
        combined = pos.combine_first(pos2)
        pos = combined
    return pos.ffill().fillna(0).astype(int)


def strat_macd(df, fast=12, slow=26, signal=9, allow_short=False, **_):
    line, sig, _ = macd(df["close"], fast, slow, signal)
    long_sig = line > sig
    short_sig = line < sig
    return _to_position(long_sig, short_sig, allow_short)


def strat_roc(df, n=20, thr=0.0, allow_short=False, **_):
    r = df["close"].pct_change(n) * 100
    long_sig = r > thr
    short_sig = r < -thr
    return _to_position(long_sig, short_sig, allow_short)


def strat_breakout_volume(df, n=20, vol_mult=1.5, allow_short=False, **_):
    upper, lower = donchian(df, n)
    vol_avg = df["volume"].rolling(n).mean()
    long_entry = (df["close"] > upper.shift(1)) & (df["volume"] > vol_mult * vol_avg)
    short_entry = (df["close"] < lower.shift(1)) & (df["volume"] > vol_mult * vol_avg)
    pos = pd.Series(np.nan, index=df.index)
    pos[long_entry] = 1
    if allow_short:
        pos[short_entry] = -1
    else:
        pos[short_entry] = 0
    return pos.ffill().fillna(0).astype(int)


def strat_atr_channel(df, n=20, k=2.0, allow_short=False, **_):
    mid = sma(df["close"], n)
    a = atr(df, n)
    upper, lower = mid + k * a, mid - k * a
    long_sig = df["close"] > upper.shift(1)
    short_sig = df["close"] < lower.shift(1)
    pos = pd.Series(np.nan, index=df.index)
    pos[long_sig] = 1
    pos[short_sig] = -1 if allow_short else 0
    return pos.ffill().fillna(0).astype(int)


def strat_keltner(df, n=20, k=2.0, allow_short=False, **_):
    mid, upper, lower = keltner(df, n, k)
    long_sig = df["close"] > upper.shift(1)
    short_sig = df["close"] < lower.shift(1)
    pos = pd.Series(np.nan, index=df.index)
    pos[long_sig] = 1
    pos[short_sig] = -1 if allow_short else 0
    return pos.ffill().fillna(0).astype(int)


def strat_squeeze(df, n=20, bb_k=2.0, kc_k=1.5, allow_short=False, **_):
    """TTM-style squeeze: BB inside KC = squeeze ON; trade direction on release using momentum sign."""
    _, bb_u, bb_l = bollinger(df["close"], n, bb_k)
    _, kc_u, kc_l = keltner(df, n, kc_k)
    squeeze_on = (bb_u < kc_u) & (bb_l > kc_l)
    release = squeeze_on.shift(1).fillna(False) & ~squeeze_on
    mom = df["close"] - sma(df["close"], n)
    pos = pd.Series(np.nan, index=df.index)
    pos[release & (mom > 0)] = 1
    if allow_short:
        pos[release & (mom < 0)] = -1
    # Exit when momentum flips through zero
    flip_down = (mom <= 0) & (mom.shift(1) > 0)
    flip_up = (mom >= 0) & (mom.shift(1) < 0)
    pos[flip_down] = pos[flip_down].where(pos[flip_down] != 1, 0)
    if allow_short:
        pos[flip_up] = pos[flip_up].where(pos[flip_up] != -1, 0)
    return pos.ffill().fillna(0).astype(int)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    func: Callable
    params: dict
    family: str
    description: str
    entry_rule: str            # what the backtester actually does to enter
    exit_rule: str             # what the backtester actually does to exit
    suggested_tp_sl: str       # common ATR/structure-based TP/SL for live trading


STRATEGIES: dict[str, StrategySpec] = {
    "EMA Cross": StrategySpec(
        "EMA Cross", strat_ema_cross,
        {"fast": (20, 5, 100, 1), "slow": (50, 10, 300, 1)},
        "Trend", "Classic moving-average crossover. Long when fast EMA > slow EMA.",
        entry_rule="Long when fast EMA crosses above slow EMA (short on opposite cross if shorting enabled).",
        exit_rule="Signal-based: flat/flip on the opposite cross. No hard stop in the model.",
        suggested_tp_sl="Stop: below slow EMA or 2×ATR(14) under entry. TP: 2–3R or trail with the slow EMA / Chandelier (3×ATR).",
    ),
    "Supertrend": StrategySpec(
        "Supertrend", strat_supertrend,
        {"period": (10, 5, 50, 1), "mult": (3.0, 1.0, 6.0, 0.5)},
        "Trend", "ATR-based trailing band; flips direction on close beyond band.",
        entry_rule="Long when Supertrend turns up (close crosses above the upper band).",
        exit_rule="Exit/flip when Supertrend turns down (close crosses below the lower band). The band itself is the trailing stop.",
        suggested_tp_sl="Stop: current Supertrend line (auto-trailing). TP: none — let the trail do the work, or scale out at 2R.",
    ),
    "Donchian Breakout": StrategySpec(
        "Donchian Breakout", strat_donchian,
        {"n": (20, 10, 100, 1)},
        "Trend", "Buy N-bar high breakout, sell N-bar low breakout (turtle-style).",
        entry_rule="Long on close above prior N-bar high. Short (if enabled) on close below prior N-bar low.",
        exit_rule="Stay long until an opposite N-bar low breakout (turtle reversal). No fixed stop in the model.",
        suggested_tp_sl="Classic Turtle: stop = 2×ATR(20) from entry; exit longs on the N/2-bar low (e.g. 10-bar low for N=20). No fixed TP.",
    ),
    "RSI Mean Reversion": StrategySpec(
        "RSI Mean Reversion", strat_rsi_meanrev,
        {"n": (14, 5, 50, 1), "oversold": (30, 10, 40, 1),
         "overbought": (70, 60, 90, 1), "exit_mid": (50, 40, 60, 1)},
        "Mean Rev", "Buy oversold RSI, exit back to midline.",
        entry_rule="Long when RSI(n) drops below `oversold`. Short (if enabled) when RSI rises above `overbought`.",
        exit_rule="Exit when RSI crosses back through `exit_mid` (default 50). No hard stop.",
        suggested_tp_sl="Stop: 1.5–2×ATR below entry or last swing low. TP: midline cross (built-in) or RSI ≥ 60 for longs.",
    ),
    "Bollinger Mean Reversion": StrategySpec(
        "Bollinger Mean Reversion", strat_bbands_meanrev,
        {"n": (20, 10, 60, 1), "k": (2.0, 1.0, 3.5, 0.1)},
        "Mean Rev", "Buy at lower band, exit at midline (fade extremes).",
        entry_rule="Long when close pierces lower band. Short (if enabled) when close pierces upper band.",
        exit_rule="Exit longs when close crosses back above the midline (SMA). Mirror for shorts.",
        suggested_tp_sl="Stop: close beyond 2.5σ (i.e. wider band) or 1.5×ATR below entry. TP: opposite band for stretch, or midline for the base case.",
    ),
    "MACD": StrategySpec(
        "MACD", strat_macd,
        {"fast": (12, 5, 30, 1), "slow": (26, 15, 60, 1), "signal": (9, 5, 20, 1)},
        "Momentum", "Long when MACD line is above its signal line.",
        entry_rule="Long when MACD line crosses above signal line (short on opposite cross if enabled).",
        exit_rule="Exit/flip on the opposite signal-line cross. No hard stop in the model.",
        suggested_tp_sl="Stop: 2×ATR(14) or below the swing low at entry. TP: trail with histogram flip, or 2–3R fixed.",
    ),
    "ROC Momentum": StrategySpec(
        "ROC Momentum", strat_roc,
        {"n": (20, 5, 100, 1), "thr": (0.0, -5.0, 10.0, 0.5)},
        "Momentum", "Long when N-bar rate-of-change exceeds threshold.",
        entry_rule="Long when N-bar ROC% > `thr`. Short (if enabled) when ROC% < -`thr`.",
        exit_rule="Exit when ROC drops back below `thr` (longs) or rises above -`thr` (shorts).",
        suggested_tp_sl="Stop: 2×ATR below entry. TP: scale out at +1R/+2R, trail rest with 3×ATR Chandelier.",
    ),
    "Breakout + Volume": StrategySpec(
        "Breakout + Volume", strat_breakout_volume,
        {"n": (20, 10, 100, 1), "vol_mult": (1.5, 1.0, 4.0, 0.1)},
        "Momentum", "Donchian breakout confirmed by above-average volume.",
        entry_rule="Long on close above prior N-bar high AND volume > vol_mult × avg(volume,N). Mirror for shorts.",
        exit_rule="Stay until the opposite breakout (no hard stop / no fixed TP in the model).",
        suggested_tp_sl="Stop: just below the breakout bar's low, or 2×ATR. TP: measured move (range height) or trail with N/2-bar low.",
    ),
    "ATR Channel Breakout": StrategySpec(
        "ATR Channel Breakout", strat_atr_channel,
        {"n": (20, 10, 60, 1), "k": (2.0, 1.0, 4.0, 0.1)},
        "Volatility", "Breakout of SMA ± k*ATR channel.",
        entry_rule="Long when close breaks above SMA + k·ATR. Short (if enabled) when close breaks below SMA − k·ATR.",
        exit_rule="Exit/flip on the opposite channel breakout. No hard stop in the model.",
        suggested_tp_sl="Stop: opposite channel edge (mean reversion) or 2×ATR from entry. TP: trail with the SMA, or take 2–3×ATR.",
    ),
    "Keltner Breakout": StrategySpec(
        "Keltner Breakout", strat_keltner,
        {"n": (20, 10, 60, 1), "k": (2.0, 1.0, 4.0, 0.1)},
        "Volatility", "Breakout of EMA ± k*ATR (Keltner) channel.",
        entry_rule="Long when close breaks above EMA + k·ATR. Short (if enabled) when close breaks below EMA − k·ATR.",
        exit_rule="Exit/flip on the opposite Keltner-edge breakout. No hard stop in the model.",
        suggested_tp_sl="Stop: middle EMA line or 2×ATR. TP: scale at upper-band tag, trail rest with the middle line.",
    ),
    "Bollinger Squeeze (TTM)": StrategySpec(
        "Bollinger Squeeze (TTM)", strat_squeeze,
        {"n": (20, 10, 60, 1), "bb_k": (2.0, 1.5, 3.0, 0.1), "kc_k": (1.5, 1.0, 3.0, 0.1)},
        "Volatility", "Trade direction of momentum release after a BB-inside-KC squeeze.",
        entry_rule="On the bar that BBands exit the Keltner channel (squeeze release): long if momentum (close − SMA) > 0, short (if enabled) if < 0.",
        exit_rule="Exit when momentum flips through zero (longs out when momentum ≤ 0; shorts out when ≥ 0).",
        suggested_tp_sl="Stop: opposite Keltner band or 1.5×ATR. TP: measured move = width of the squeeze projected from breakout, or trail with momentum-flip (built-in).",
    ),
}
