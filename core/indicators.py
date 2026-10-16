"""אינדיקטורים טכניים, ממומשים ישירות ב-pandas.

כל הפונקציות סיבתיות: הערך ביום t תלוי רק בנתונים עד יום t (כולל).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """EMA שמתחיל מ-SMA של period הערכים הראשונים (כמו ברוב הפלטפורמות)."""
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) < period:
        return pd.Series(out, index=series.index)
    start = valid[0]
    seed_end = start + period
    out[seed_end - 1] = values[start:seed_end].mean()
    alpha = 2.0 / (period + 1)
    for i in range(seed_end, len(values)):
        v = values[i]
        out[i] = out[i - 1] if np.isnan(v) else alpha * v + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


def _wilder(values: np.ndarray, period: int, first_index: int) -> np.ndarray:
    """החלקת Wilder: ממוצע פשוט של period הערכים הראשונים, ואחר כך (prev*(n-1)+x)/n."""
    out = np.full(len(values), np.nan)
    seed_end = first_index + period
    if seed_end > len(values):
        return out
    out[seed_end - 1] = values[first_index:seed_end].mean()
    for i in range(seed_end, len(values)):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff().to_numpy(dtype=float)
    gain = np.where(change > 0, change, 0.0)
    loss = np.where(change < 0, -change, 0.0)
    avg_gain = _wilder(gain, period, first_index=1)
    avg_loss = _wilder(loss, period, first_index=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    out = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, out)
    out = np.where((avg_loss == 0) & (avg_gain == 0), 50.0, out)
    out[np.isnan(avg_gain)] = np.nan
    return pd.Series(out, index=close.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    )
    tr = ranges.max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df).to_numpy(dtype=float)
    return pd.Series(_wilder(tr, period, first_index=0), index=df.index)


def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP תוך-יומי שמתאפס בתחילת כל יום מסחר."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    day = pd.DatetimeIndex(df.index).normalize()
    pv = (typical * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum()
    out = pv / vol.replace(0, np.nan)
    # בנר הראשון ביום, כשעוד אין מחזור, VWAP הוא המחיר הטיפוסי.
    return out.fillna(typical)


def avg_turnover(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df["turnover"].rolling(window, min_periods=window).mean()


def typical_turnover(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """מחזור כספי ביום רגיל: חציון ולא ממוצע, כדי שעסקת בלוק אחת לא תעוות את התמונה."""
    return df["turnover"].rolling(window, min_periods=window).median()


def typical_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df["volume"].rolling(window, min_periods=window).median()


def relative_strength(close: pd.Series, index_close: pd.Series, period: int = 63) -> pd.Series:
    """תשואה יחסית למדד על פני period ימים. חיובי = המניה חזקה מהמדד."""
    idx = index_close.reindex(close.index).ffill()
    return (close / close.shift(period)) / (idx / idx.shift(period)) - 1.0
