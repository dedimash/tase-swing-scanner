"""שכבת נתונים: הורדת מחירים, המרה מאגורות לשקלים, cache ודוח איכות נתונים.

כל DataFrame שיוצא מכאן בנוי באותה צורה:
  index  - DatetimeIndex (יומי: תאריך בלי אזור זמן; תוך-יומי: Asia/Jerusalem)
  עמודות - open, high, low, close, volume, turnover
  מחירים בשקלים, turnover = close * volume בשקלים.
"""
from __future__ import annotations

import json
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import cache_dir, load_config

PRICE_COLS = ["open", "high", "low", "close"]
OUT_COLS = ["open", "high", "low", "close", "volume", "turnover"]

# Yahoo מחזיר מניות ת"א באגורות. כל מטבע שמופיע כאן מחולק במקדם שלו.
SUBUNIT_DIVISORS = {"ILA": 100.0, "GBp": 100.0, "ZAc": 100.0}


def repair_unit_errors(df: pd.DataFrame) -> pd.DataFrame:
    """מתקן קפיצות של פי 100 שנובעות מבלבול בין אגורות לשקלים ב-Yahoo.

    הנתון האחרון נחשב נכון. עוברים מהסוף להתחלה, ובכל קפיצה של פי 60 עד 160
    (למעלה או למטה) מכפילים את כל ההיסטוריה שלפניה במקדם המתאים.
    זה מתקן גם שינוי רמה מתמשך וגם נר בודד חריג.
    """
    close = df["close"].to_numpy(dtype=float)
    if len(close) < 2:
        return df
    ratio = close[1:] / close[:-1]
    step = np.where((ratio > 60) & (ratio < 160), 100.0, np.where((ratio > 1 / 160) & (ratio < 1 / 60), 0.01, 1.0))
    if (step == 1.0).all():
        return df
    # factor[t] = מכפלת כל התיקונים שאחרי t
    factor = np.append(np.cumprod(step[::-1])[::-1], 1.0)
    out = df.copy()
    out[PRICE_COLS] = out[PRICE_COLS].mul(factor, axis=0)
    return out


def normalize_frame(raw: pd.DataFrame, currency: str | None) -> pd.DataFrame:
    """ממיר טבלת OHLCV גולמית של Yahoo לפורמט האחיד, בשקלים."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=OUT_COLS)
    df = raw.rename(columns=str.lower)
    missing = [c for c in PRICE_COLS + ["volume"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns from provider: {missing}")
    df = df[PRICE_COLS + ["volume"]].astype(float)
    df = df.dropna(subset=["close"])
    df = repair_unit_errors(df)
    divisor = SUBUNIT_DIVISORS.get(currency or "", 1.0)
    if divisor != 1.0:
        df[PRICE_COLS] = df[PRICE_COLS] / divisor
    df["volume"] = df["volume"].fillna(0.0)
    df["turnover"] = df["close"] * df["volume"]
    df.index.name = "date"
    return df[OUT_COLS]


class PriceProvider(ABC):
    """ממשק לספק נתונים. היום Yahoo, ובעתיד אפשר להחליף בלי לגעת בשאר הקוד."""

    @abstractmethod
    def daily(self, symbols: list[str], years: int) -> dict[str, pd.DataFrame]:
        ...

    @abstractmethod
    def intraday(self, symbols: list[str], interval: str, days: int) -> dict[str, pd.DataFrame]:
        ...


class YahooProvider(PriceProvider):
    def __init__(self, currencies: dict[str, str] | None = None):
        self._currencies = dict(currencies or {})
        self._currency_cache_path = cache_dir() / "currency.json"
        if self._currency_cache_path.exists():
            cached = json.loads(self._currency_cache_path.read_text(encoding="utf-8"))
            for k, v in cached.items():
                self._currencies.setdefault(k, v)

    def currency(self, symbol: str) -> str | None:
        if symbol in self._currencies:
            return self._currencies[symbol]
        import yfinance as yf

        try:
            cur = yf.Ticker(symbol).fast_info.get("currency")
        except Exception:
            cur = None
        if cur:
            self._currencies[symbol] = cur
            self._currency_cache_path.write_text(
                json.dumps({k: v for k, v in self._currencies.items() if v}, ensure_ascii=False),
                encoding="utf-8",
            )
        return cur

    def _download_once(self, symbols: list[str], **kwargs) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        chunk = load_config()["data"]["download_chunk_size"]
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), chunk):
            part = symbols[i : i + chunk]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    part, auto_adjust=True, group_by="ticker", progress=False, threads=True, **kwargs
                )
            for sym in part:
                if raw is None or raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    if sym not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[sym]
                else:
                    sub = raw
                sub = sub.dropna(how="all")
                if not sub.empty:
                    out[sym] = normalize_frame(sub, self.currency(sym))
            if i + chunk < len(symbols):
                time.sleep(1.0)
        return out

    def _download(self, symbols: list[str], retries: int = 2, **kwargs) -> dict[str, pd.DataFrame]:
        """מוריד, ומנסה שוב את מה שלא הגיע. Yahoo מגביל לפעמים בקשות משרתי ענן ("Too Many Requests")."""
        out = self._download_once(symbols, **kwargs)
        for attempt in range(retries):
            missing = [s for s in symbols if s not in out]
            # מניה בודדת בלי נתונים ברשימה ארוכה היא בדרך כלל מניה שנמחקה, לא חסימה. לא מחכים בשבילה.
            if not missing or (len(symbols) > 5 and len(missing) < 5):
                break
            time.sleep(3.0 * (attempt + 1))
            out.update(self._download_once(missing, **kwargs))
        return out

    def daily(self, symbols: list[str], years: int) -> dict[str, pd.DataFrame]:
        frames = self._download(symbols, period=f"{years}y", interval="1d")
        for df in frames.values():
            df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        return frames

    def intraday(self, symbols: list[str], interval: str, days: int) -> dict[str, pd.DataFrame]:
        return self._download(symbols, period=f"{days}d", interval=interval)


class PriceStore:
    """cache מקומי של קובצי parquet מעל ספק נתונים."""

    def __init__(self, provider: PriceProvider | None = None, currencies: dict[str, str] | None = None):
        self.provider = provider or YahooProvider(currencies)
        self.root = cache_dir() / "prices"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = (
            json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {}
        )

    def _path(self, symbol: str, interval: str) -> Path:
        safe = symbol.replace("^", "IDX_").replace("=", "_")
        return self.root / interval / f"{safe}.parquet"

    def _fresh(self, symbol: str, interval: str, ttl: timedelta, span: int) -> bool:
        """עדכני = נשמר לפני פחות מ-ttl, ומכסה לפחות span (שנים ביומי, ימים בתוך-יומי)."""
        entry = self.manifest.get(interval, {}).get(symbol)
        if not entry or not self._path(symbol, interval).exists():
            return False
        if isinstance(entry, str):  # פורמט ישן: רק זמן שמירה, של היסטוריה מלאה
            entry = {"ts": entry, "span": 10**6}
        if entry.get("span", 0) < span:
            return False
        return datetime.now() - datetime.fromisoformat(entry["ts"]) < ttl

    def _save(self, symbol: str, interval: str, df: pd.DataFrame, span: int) -> None:
        p = self._path(symbol, interval)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
        self.manifest.setdefault(interval, {})[symbol] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "span": span,
        }

    def _flush(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")

    def _get(self, symbols, interval, ttl, span, fetch) -> dict[str, pd.DataFrame]:
        symbols = list(dict.fromkeys(symbols))
        stale = [s for s in symbols if not self._fresh(s, interval, ttl, span)]
        if stale:
            fetched = fetch(stale)
            for sym, df in fetched.items():
                self._save(sym, interval, df, span)
            self._flush()
        out = {}
        for s in symbols:
            p = self._path(s, interval)
            if p.exists():
                out[s] = pd.read_parquet(p)
        return out

    def daily(self, symbols: list[str], years: int | None = None) -> dict[str, pd.DataFrame]:
        cfg = load_config()["data"]
        years = years or cfg["daily_history_years"]
        ttl = timedelta(hours=cfg["daily_ttl_hours"])
        return self._get(symbols, "1d", ttl, years, lambda s: self.provider.daily(s, years))

    def intraday(self, symbols: list[str], interval: str = "60m", days: int | None = None) -> dict[str, pd.DataFrame]:
        cfg = load_config()["data"]
        days = days or (729 if interval in ("60m", "1h") else 59)
        ttl = timedelta(minutes=cfg["intraday_ttl_minutes"])
        return self._get(symbols, interval, ttl, days, lambda s: self.provider.intraday(s, interval, days))

    def cached_daily(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """קורא רק מה-cache, בלי לפנות לספק."""
        out = {}
        for s in symbols:
            p = self._path(s, "1d")
            if p.exists():
                out[s] = pd.read_parquet(p)
        return out


def quality_report(frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """דוח איכות נתונים: ימים חסרים, קפיצות חשודות, ימים בלי מחזור, נתונים לא עדכניים."""
    max_jump = load_config()["quality"]["max_daily_jump_pct"] / 100.0
    last_market_day = calendar.max() if calendar is not None and len(calendar) else None
    rows = []
    for sym, df in frames.items():
        if df is None or df.empty:
            rows.append({"symbol": sym, "rows": 0, "issue": "אין נתונים"})
            continue
        ret = df["close"].pct_change().abs()
        jumps = ret[ret > max_jump]
        missing = 0
        if calendar is not None:
            cal = calendar[(calendar >= df.index.min()) & (calendar <= df.index.max())]
            missing = int(len(cal.difference(df.index)))
        stale_days = int((last_market_day - df.index.max()).days) if last_market_day is not None else 0
        issues = []
        if len(jumps):
            issues.append(f"{len(jumps)} קפיצות מעל {int(max_jump * 100)}%")
        if missing:
            issues.append(f"{missing} ימים חסרים")
        if stale_days > 3:
            issues.append(f"נתון אחרון לפני {stale_days} ימים")
        rows.append(
            {
                "symbol": sym,
                "rows": len(df),
                "first": df.index.min().date(),
                "last": df.index.max().date(),
                "missing_days": missing,
                "jumps": len(jumps),
                "max_jump_pct": round(float(ret.max() * 100), 1) if len(ret.dropna()) else np.nan,
                "zero_volume_days": int((df["volume"] == 0).sum()),
                "issue": "; ".join(issues) if issues else "",
            }
        )
    return pd.DataFrame(rows)
