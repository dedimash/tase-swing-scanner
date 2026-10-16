"""אירועים שמשפיעים על חלון ההחזקה: דוחות כספיים ואקס-דיבידנד.

המקור: מסך המניות של Yahoo (קריאה אחת לכל הבורסה). הכיסוי חלקי:
  - לחלק מהמניות יש תאריך דוח עתידי ידוע.
  - לאחרות יש רק תאריך הדוח הקודם, ואז מעריכים: הדוח הקודם + כרבעון.
  - אקס-דיבידנד מופיע רק כשהחברה כבר הודיעה עליו.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from core.config import cache_dir, load_config

EARNINGS_FIELDS = ("earningsTimestampStart", "earningsTimestampEnd", "earningsTimestamp", "earningsCallTimestampStart")


@dataclass
class StockEvents:
    earnings: date | None = None
    earnings_is_estimate: bool = True
    ex_dividend: date | None = None


def _to_date(ts) -> date | None:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    ts = float(ts)
    if ts > 1e11:  # מילישניות
        ts /= 1000.0
    return datetime.fromtimestamp(ts).date()


def parse_quote(quote: dict, today: date) -> StockEvents:
    cfg = load_config()["events"]
    dates = [d for d in (_to_date(quote.get(f)) for f in EARNINGS_FIELDS) if d]
    future = sorted(d for d in dates if d >= today)
    ev = StockEvents()
    if future:
        ev.earnings = future[0]
        ev.earnings_is_estimate = bool(quote.get("isEarningsDateEstimate", True))
    elif dates:
        guess = max(dates) + timedelta(days=cfg["estimate_days_after_last"])
        if guess >= today - timedelta(days=cfg["estimate_margin_days"]):
            ev.earnings, ev.earnings_is_estimate = guess, True
    for action in quote.get("corporateActions") or []:
        meta = action.get("meta", {})
        if meta.get("eventType") == "DIVIDEND":
            d = _to_date(meta.get("dateEpochMs"))
            if d and d >= today and (ev.ex_dividend is None or d < ev.ex_dividend):
                ev.ex_dividend = d
    return ev


def fetch_quotes() -> list[dict]:
    import yfinance as yf

    query = yf.EquityQuery("eq", ["exchange", "TLV"])
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for offset in range(0, 3000, 250):
            quotes = yf.screen(query, size=250, offset=offset, sortField="intradaymarketcap", sortAsc=False).get("quotes", [])
            rows += quotes
            if len(quotes) < 250:
                break
    return rows


def load_events(today: date | None = None) -> dict[str, StockEvents]:
    """אירועים לכל המניות, עם cache של כמה שעות. אם Yahoo לא זמין, מחזיר מה שנשמר."""
    today = today or date.today()
    path = cache_dir() / "events.json"
    ttl = timedelta(hours=load_config()["events"]["ttl_hours"])
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    fresh = cached and datetime.now() - datetime.fromisoformat(cached["fetched_at"]) < ttl
    if not fresh:
        try:
            quotes = fetch_quotes()
            data = {q["symbol"]: asdict(parse_quote(q, today)) for q in quotes if q.get("symbol")}
            cached = {"fetched_at": datetime.now().isoformat(timespec="seconds"), "events": data}
            path.write_text(json.dumps(cached, default=str), encoding="utf-8")
        except Exception:
            if not cached:
                return {}
    out = {}
    for sym, e in cached["events"].items():
        out[sym] = StockEvents(
            earnings=date.fromisoformat(e["earnings"]) if e.get("earnings") else None,
            earnings_is_estimate=bool(e.get("earnings_is_estimate", True)),
            ex_dividend=date.fromisoformat(e["ex_dividend"]) if e.get("ex_dividend") else None,
        )
    return out
