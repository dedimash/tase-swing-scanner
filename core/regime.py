"""מצב השוק: משטר לפי מדד ת"א 125, רוחב שוק, וימים עם מחזור חריג בכל השוק."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import load_config
from core.indicators import sma

UP, DOWN, MIXED = "עולה", "יורד", "מעורב"


def market_regime(index_df: pd.DataFrame) -> pd.Series:
    """עולה: המדד מעל SMA50 ומעל SMA200. יורד: מתחת לשניהם. אחרת: מעורב."""
    cfg = load_config()["regime"]
    close = index_df["close"]
    fast, slow = sma(close, cfg["fast_sma"]), sma(close, cfg["slow_sma"])
    regime = np.where((close > fast) & (close > slow), UP, np.where((close < fast) & (close < slow), DOWN, MIXED))
    out = pd.Series(regime, index=index_df.index, dtype=object)
    out[slow.isna()] = None
    return out


def regime_on(dates: pd.DatetimeIndex, regime: pd.Series) -> pd.Series:
    """המשטר שהיה ידוע בכל תאריך (הערך האחרון עד התאריך הזה)."""
    unique = pd.DatetimeIndex(dates).unique()
    known = regime.reindex(regime.index.union(unique)).ffill().reindex(unique)
    return pd.Series(known.reindex(dates).to_numpy(), index=dates)


def breadth(frames: dict[str, pd.DataFrame], period: int = 50, min_stocks: int = 20) -> pd.Series:
    """רוחב שוק: איזה חלק מהמניות נסגר מעל הממוצע של period ימים (0 עד 1)."""
    above = {}
    for sym, df in frames.items():
        if df is None or len(df) < period:
            continue
        s = sma(df["close"], period)
        above[sym] = (df["close"] > s).where(s.notna())
    if not above:
        return pd.Series(dtype=float)
    table = pd.DataFrame(above)
    counted = table.notna().sum(axis=1)
    share = table.sum(axis=1) / counted.replace(0, np.nan)
    return share[counted >= min_stocks]


def market_event_days(
    frames: dict[str, pd.DataFrame], spike: float = 2.5, share: float = 0.3, min_stocks: int = 20
) -> pd.DatetimeIndex:
    """ימים שבהם לחלק גדול מהמניות היה מחזור חריג באותו יום. בדרך כלל אלה ימי עדכון מדדים."""
    flags = {}
    for sym, df in frames.items():
        if df is None or len(df) < 25:
            continue
        avg = df["volume"].shift(1).rolling(20, min_periods=20).mean()
        flags[sym] = (df["volume"] > spike * avg).where(avg.notna() & (avg > 0))
    if not flags:
        return pd.DatetimeIndex([])
    table = pd.DataFrame(flags)
    counted = table.notna().sum(axis=1)
    frac = table.sum(axis=1) / counted.replace(0, np.nan)
    return pd.DatetimeIndex(frac[(counted >= min_stocks) & (frac >= share)].index)
