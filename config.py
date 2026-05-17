"""Per-machine config loader.

Reads config.toml (next to this file) and exposes its values as plain Python
constants. Anyone cloning the repo edits config.toml — no code changes
needed to switch market, default symbol, default strategy, etc.

Secrets (API keys) are NEVER read from this file. They come from the
Streamlit form or BINANCE_API_KEY / BINANCE_API_SECRET env vars.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — Python <3.11 fallback
    import tomli as tomllib  # type: ignore

_CONFIG_PATH = Path(__file__).parent / "config.toml"


def _load() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {_CONFIG_PATH}. Copy config.toml.example or restore the "
            "file from the repo root."
        )
    with _CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


_cfg = _load()


def _section(name: str) -> dict:
    return _cfg.get(name, {}) or {}


# ─── App (general Streamlit) ────────────────────────────────────────────────
_app = _section("app")
APP_TITLE: str = _app.get("title", "📈 Crypto 4H Backtester")
APP_PAGE_ICON: str = _app.get("page_icon", "📈")
DEFAULT_MARKET: str = _app.get("market", "spot")
DEFAULT_QUOTE: str = _app.get("quote", "USDT")
UNIVERSE_SIZE: int = int(_app.get("universe_size", 100))
DEFAULT_TIMEFRAME: str = _app.get("timeframe", "4h")
DEFAULT_LOOKBACK_DAYS: int = int(_app.get("lookback_days", 365 * 2))

# ─── Single-strategy mode ───────────────────────────────────────────────────
_ss = _section("single_strategy")
DEFAULT_SYMBOL: str = _ss.get("default_symbol", "BTC/USDT")
DEFAULT_STRATEGY: str = _ss.get("default_strategy", "EMA Cross")

# ─── Tournament mode ────────────────────────────────────────────────────────
_t = _section("tournament")
TOURNAMENT_DEFAULT_N: int = int(_t.get("default_n_symbols", 5))
TOURNAMENT_RANK_METRIC: str = _t.get("rank_metric", "Total Return")

# ─── Live trader form ──────────────────────────────────────────────────────
_tr = _section("trader")
TRADER_MARKET: str = _tr.get("market", DEFAULT_MARKET)
TRADER_SYMBOL: str = _tr.get("symbol", DEFAULT_SYMBOL)
TRADER_TIMEFRAME: str = _tr.get("timeframe", DEFAULT_TIMEFRAME)
TRADER_STRATEGY: str = _tr.get("strategy", DEFAULT_STRATEGY)
TRADER_STRATEGY_PARAMS: dict = dict(_tr.get("strategy_params", {}) or {})
TRADER_NOTIONAL_PCT: float = float(_tr.get("notional_pct", 25.0))
TRADER_LEVERAGE: float = float(_tr.get("leverage", 1.0))
TRADER_ALLOW_SHORT: bool = bool(_tr.get("allow_short", False))
TRADER_BAR_BUFFER_SECONDS: int = int(_tr.get("bar_buffer_seconds", 5))

# ─── Paths ──────────────────────────────────────────────────────────────────
_p = _section("paths")
LOG_DIR: Path = Path(_p.get("log_dir", "/tmp")).expanduser()
KEY_STORE: Path = Path(
    _p.get("key_store", "~/.crypto-backtest/binance.json")
).expanduser()

# ─── UI ─────────────────────────────────────────────────────────────────────
_ui = _section("ui")
DEFAULT_AUTO_REFRESH: str = _ui.get("default_auto_refresh", "3s")
LOG_TAIL_LINES: int = int(_ui.get("log_tail_lines", 300))
