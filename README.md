# Crypto 4H Backtester

Streamlit app to backtest popular trading strategies on the top-100 crypto market-cap universe, using free Binance OHLCV data via CCXT.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then pick a symbol, a strategy, tweak parameters, and click **Run backtest**.

## What's included

**Universe:** Top 100 coins by market cap (CoinGecko), filtered to those with a spot pair against `USDT` (or `USDC`/`BUSD`) on Binance.

**Default timeframe:** 4H. Also supports 1H / 2H / 1D.

### Strategies — the most commonly cited "edges" on crypto 4H

| Family | Strategy | Idea |
|---|---|---|
| Trend | EMA Cross | Fast EMA over slow EMA = long |
| Trend | Supertrend | ATR trailing band; flip on close beyond band |
| Trend | Donchian Breakout | N-bar high/low breakout (turtle-style) |
| Mean Reversion | RSI Mean Reversion | Buy oversold, exit at midline |
| Mean Reversion | Bollinger Mean Reversion | Buy at lower band, exit at midline |
| Momentum | MACD | MACD line above signal |
| Momentum | ROC Momentum | Threshold on N-bar rate of change |
| Momentum | Breakout + Volume | Donchian breakout confirmed by volume spike |
| Volatility | ATR Channel Breakout | Close beyond SMA ± k·ATR |
| Volatility | Keltner Breakout | Close beyond EMA ± k·ATR |
| Volatility | Bollinger Squeeze (TTM) | Trade direction of release from BB-inside-KC squeeze |

Trend-following and breakout strategies historically show the strongest edge on crypto 4H because of momentum/regime persistence; mean reversion and squeeze releases tend to work in chop. Always sanity-check vs. buy-and-hold (shown on the equity chart).

## Backtest mechanics

- Signals computed on bar close; positions executed at the **next bar's open** (no lookahead).
- Per-side **fees** and **slippage** in bps, charged on every position change.
- Reports: Total return, CAGR vs. buy & hold, Sharpe / Sortino / Calmar, max drawdown, # trades, win rate, profit factor, expectancy, exposure.
- Optional shorting (off by default).
- Full trade ledger + equity curve CSV downloads.

## Files

- `app.py` — Streamlit UI
- `data.py` — Universe + OHLCV loaders (cached)
- `strategies.py` — Strategy library (each returns a position series in {-1,0,+1})
- `backtest.py` — Vectorized backtester with metrics
