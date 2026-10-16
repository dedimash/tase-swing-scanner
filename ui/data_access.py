"""גישה לנתונים מתוך הממשק, עם cache של Streamlit ושגיאות בעברית."""
from __future__ import annotations

import json
import warnings

import pandas as pd
import streamlit as st

from core.config import RESULTS_DIR, load_config
from core.data import PriceStore
from core.events import load_events
from core.regime import breadth, market_event_days, market_regime
from core.universe import load_universe, scan_universe
from core.verdict import MarketContext

SCAN_YEARS = 2  # מספיק לממוצע 200 ימים ולאינדיקטורים


class DataUnavailable(Exception):
    """לא הצלחנו לקבל נתונים מהספק."""


def _store() -> PriceStore:
    u = load_universe()
    return PriceStore(currencies=dict(zip(u["symbol"], u["currency"])))


@st.cache_data(ttl=900, show_spinner=False)
def daily_prices(symbol: str) -> pd.DataFrame:
    try:
        frames = _store().daily([symbol], years=SCAN_YEARS)
    except Exception as exc:
        raise DataUnavailable(str(exc)) from exc
    df = frames.get(symbol)
    if df is None or df.empty:
        raise DataUnavailable(f"no data for {symbol}")
    return df


@st.cache_data(ttl=900, show_spinner=False)
def scan_prices() -> dict[str, pd.DataFrame]:
    """נתונים יומיים לכל המניות הנזילות (בקריאה מרוכזת)."""
    symbols = scan_universe()["symbol"].tolist()
    try:
        return _store().daily(symbols, years=SCAN_YEARS)
    except Exception as exc:
        raise DataUnavailable(str(exc)) from exc


@st.cache_data(ttl=3600, show_spinner=False)
def edge_cards() -> dict | None:
    path = RESULTS_DIR / "edge_cards.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=900, show_spinner=False)
def market_context() -> MarketContext:
    idx_sym = load_config()["data"]["index_symbol"]
    index_df = _store().daily([idx_sym], years=SCAN_YEARS).get(idx_sym, pd.DataFrame())
    frames = scan_prices()
    try:
        events = load_events()
    except Exception:
        events = {}
    return MarketContext(
        index_df=index_df,
        regime=market_regime(index_df) if not index_df.empty else pd.Series(dtype=object),
        breadth=breadth(frames),
        event_days=market_event_days(frames),
        events=events,
        edge=edge_cards() or {},
    )


@st.cache_data(ttl=900, show_spinner=False)
def us_change_pct(us_symbol: str) -> float | None:
    """שינוי באחוזים של המניה בניו יורק, מול הסגירה הקודמת שם."""
    if not us_symbol:
        return None
    import yfinance as yf

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = yf.Ticker(us_symbol).fast_info
            last, prev = info.get("lastPrice"), info.get("previousClose")
        return (last / prev - 1) * 100 if last and prev else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def trades(setup_key: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"trades_{setup_key}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["signal_date", "entry_date", "exit_date"])


def cached_frames(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """נתונים מה-cache המקומי בלבד (להרצת בקטסט מתוך הממשק)."""
    return _store().cached_daily(symbols)
