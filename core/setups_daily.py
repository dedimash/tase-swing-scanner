"""תבניות מסחר על נרות יומיים.

כל תבנית מחזירה טבלת איתותים: שורה לכל יום שבו התנאים התקיימו בסגירה.
הכניסה עצמה מתבצעת לכל המוקדם בפתיחה של יום המסחר הבא.

לכל תבנית יש גם watch(): כשהמניה קרובה לתבנית אבל עוד אין איתות,
הפונקציה מחזירה משפט בעברית שמסביר למה לחכות (בשביל הרמזור הצהוב).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from core.config import load_config
from core.indicators import atr, ema, relative_strength, rsi, sma, typical_turnover

SIGNAL_COLS = ["stop", "atr", "close", "target_r", "max_hold"]


class Setup(ABC):
    key: str = ""
    name_he: str = ""
    description_he: str = ""
    # תבניות שבהן מחזור גבוה הוא חלק מהאיתות. ביום שבו כל השוק במחזור חריג (עדכון מדדים) האיתות לא נספר.
    uses_volume_confirmation: bool = False

    def __init__(self, **overrides):
        defaults = load_config()["setups"][self.key]
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise ValueError(f"unknown parameters for {self.key}: {sorted(unknown)}")
        self.params = {**defaults, **overrides}

    @abstractmethod
    def detect(self, df: pd.DataFrame, index_close: pd.Series | None = None) -> pd.DataFrame:
        """מחזיר DataFrame לפי תאריך האיתות, עם העמודות SIGNAL_COLS לפחות.

        index_close: סגירות של מדד ת"א 125, לחישוב חוזק יחסי (לא חובה).
        """

    def watch(self, df: pd.DataFrame) -> str | None:
        """אם המניה קרובה לתבנית בנר האחרון: משפט שמסביר למה לחכות. אחרת None."""
        return None

    @staticmethod
    def _liquid(df: pd.DataFrame) -> pd.Series:
        uni = load_config()["universe"]
        return typical_turnover(df, uni["turnover_window_days"]) >= uni["min_avg_turnover_ils"]

    @staticmethod
    def _rs(df: pd.DataFrame, index_close: pd.Series | None, period: int = 63) -> pd.Series:
        if index_close is None:
            return pd.Series(np.nan, index=df.index)
        return relative_strength(df["close"], index_close, period)

    def _finish(self, out: pd.DataFrame, p: dict) -> pd.DataFrame:
        """עמודות משותפות לכל האיתותים."""
        out["target_r"] = _optional(p.get("target_r"))
        out["max_hold"] = int(p["max_hold_days"])
        out["breakeven_r"] = _optional(p.get("breakeven_r"))
        out["trail_atr"] = _optional(p.get("trail_atr"))
        out["exit_ma"] = _optional(p.get("exit_ma"))
        if "entry_trigger" not in out:
            out["entry_trigger"] = np.nan
        if "rank_score" not in out:
            out["rank_score"] = out["rs"] if "rs" in out else np.nan
        return out


def _optional(value) -> float:
    return np.nan if value is None else float(value)


def _close_location(df: pd.DataFrame) -> pd.Series:
    day_range = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["low"]) / day_range


class Breakout20(Setup):
    key = "breakout_20d"
    name_he = "פריצה למעלה"
    description_he = "המניה סגרה מעל השיא של התקופה האחרונה, במחזור גבוה מהרגיל ובמגמה עולה."
    uses_volume_confirmation = True

    def detect(self, df: pd.DataFrame, index_close: pd.Series | None = None) -> pd.DataFrame:
        p = self.params
        prior_high = df["high"].shift(1).rolling(p["lookback"], min_periods=p["lookback"]).max()
        avg_volume = df["volume"].shift(1).rolling(p["volume_window"], min_periods=p["volume_window"]).mean()
        trend = sma(df["close"], p["trend_sma"])
        a = atr(df, p["atr_period"])
        gap = df["open"] - df["close"].shift(1)
        rs = self._rs(df, index_close, p["rs_period"])

        condition = (
            (df["close"] > prior_high)
            & (df["volume"] >= p["volume_mult"] * avg_volume)
            & (df["close"] > trend)
            & (gap <= p["max_gap_atr"] * a.shift(1))
            & self._liquid(df)
            & a.notna()
        )
        if p["above_sma200"]:
            condition &= df["close"] > sma(df["close"], 200)
        if p["min_rs"] is not None:
            condition &= rs > p["min_rs"]
        if p["max_atr_contraction"] is not None:
            contraction = (atr(df, 10) / atr(df, 50)).shift(1)
            condition &= contraction <= p["max_atr_contraction"]
        if p["min_close_location"] is not None:
            condition &= _close_location(df) >= p["min_close_location"]

        structural_stop = df["low"].rolling(p["stop_lookback"], min_periods=1).min()
        widest_stop = df["close"] - p["max_stop_atr"] * a
        stop = np.maximum(structural_stop, widest_stop)

        out = pd.DataFrame(
            {
                "stop": stop,
                "atr": a,
                "close": df["close"],
                "prior_high": prior_high,
                "volume_ratio": df["volume"] / avg_volume,
                "rs": rs,
                "entry_trigger": df["high"] if p["entry_mode"] == "stop_above_high" else np.nan,
            }
        )[condition.fillna(False)]
        out["rank_score"] = out["rs"] if p["rank_by"] == "rs" else out["volume_ratio"]
        return self._finish(out, p)

    def watch(self, df: pd.DataFrame) -> str | None:
        p = self.params
        if len(df) < max(p["lookback"], p["trend_sma"]) + 1:
            return None
        level = df["high"].iloc[-p["lookback"] :].max()  # השיא שצריך לעבור מחר
        close = df["close"].iloc[-1]
        a = atr(df, p["atr_period"]).iloc[-1]
        in_trend = close > sma(df["close"], p["trend_sma"]).iloc[-1]
        if in_trend and close < level and level - close <= 1.0 * a:
            return f"פריצה: סגירה מעל {level:,.2f} ₪ ביום עם מחזור גבוה מהרגיל"
        return None


class PullbackEMA20(Setup):
    key = "pullback_ema20"
    name_he = "ירידה קטנה במניה שעולה"
    description_he = "מניה במגמה עולה ירדה קצת עד לממוצע של 20 הימים, במחזור נמוך, וחזרה לעלות."

    def detect(self, df: pd.DataFrame, index_close: pd.Series | None = None) -> pd.DataFrame:
        p = self.params
        c = df["close"]
        e20, e50, s200 = ema(c, 20), ema(c, 50), sma(c, 200)
        a = atr(df, p["atr_period"])
        days = p["pullback_days"]
        touched = df["low"].rolling(days, min_periods=days).min() <= e20 * (1 + p["touch_pct"] / 100.0)
        reversal = (c > df["open"]) & (c > c.shift(1)) & (_close_location(df) >= 0.5)
        pullback_volume = df["volume"].shift(1).rolling(days, min_periods=days).mean()
        normal_volume = df["volume"].shift(days + 1).rolling(20, min_periods=20).mean()

        condition = (
            (e20 > e50)
            & (c > s200)
            & touched
            & (c > e20)
            & reversal
            & (pullback_volume < normal_volume)
            & self._liquid(df)
            & a.notna()
        )
        structural_stop = df["low"].rolling(days + 1, min_periods=1).min()
        stop = np.maximum(structural_stop, c - p["max_stop_atr"] * a)
        out = pd.DataFrame(
            {"stop": stop, "atr": a, "close": c, "ema20": e20, "rs": self._rs(df, index_close)}
        )[condition.fillna(False)]
        return self._finish(out, p)

    def watch(self, df: pd.DataFrame) -> str | None:
        if len(df) < 201:
            return None
        c = df["close"]
        e20, e50, s200 = ema(c, 20).iloc[-1], ema(c, 50).iloc[-1], sma(c, 200).iloc[-1]
        a = atr(df, self.params["atr_period"]).iloc[-1]
        close = c.iloc[-1]
        if e20 > e50 and close > s200 and close > e20 + a:
            return f"ירידה קטנה לאזור {e20:,.2f} ₪ (הממוצע של 20 ימים), ואחריה יום של עלייה"
        return None


class RSI2Reversion(Setup):
    key = "rsi2_reversion"
    name_he = "ירידה חדה במניה חזקה"
    description_he = "מניה במגמה עולה ארוכה ירדה בחדות ביום-יומיים האחרונים. בדרך כלל מניות כאלה מתאוששות מהר."

    def detect(self, df: pd.DataFrame, index_close: pd.Series | None = None) -> pd.DataFrame:
        p = self.params
        c = df["close"]
        r = rsi(c, p["rsi_period"])
        a = atr(df, p["atr_period"])
        condition = (c > sma(c, 200)) & (r < p["rsi_max"]) & self._liquid(df) & a.notna()
        out = pd.DataFrame(
            {"stop": c - p["stop_atr"] * a, "atr": a, "close": c, "rsi2": r, "rs": self._rs(df, index_close)}
        )[condition.fillna(False)]
        # בבחירה בין כמה איתותים: הירידה החדה ביותר קודמת
        out["rank_score"] = -out["rsi2"]
        return self._finish(out, p)

    def watch(self, df: pd.DataFrame) -> str | None:
        if len(df) < 201:
            return None
        c = df["close"]
        r = rsi(c, self.params["rsi_period"]).iloc[-1]
        threshold = self.params["rsi_max"]
        if c.iloc[-1] > sma(c, 200).iloc[-1] and threshold <= r < 3 * threshold:
            return "עוד יום של ירידה חדה (המניה כבר ירדה קצת, בתוך מגמה עולה ארוכה)"
        return None


SETUPS: dict[str, type[Setup]] = {s.key: s for s in (Breakout20, PullbackEMA20, RSI2Reversion)}
