"""מודול סיכון: גודל פוזיציה, תקרות, אירועים בחלון ההחזקה, ואזהרות."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from core.config import load_config
from core.events import StockEvents

RED, YELLOW = "red", "yellow"


@dataclass
class PositionPlan:
    shares: int
    amount_ils: float
    risk_ils: float
    limited_by: list[str] = field(default_factory=list)


def plan_position(
    entry: float,
    stop: float,
    avg_volume_shares: float | None = None,
    capital: float | None = None,
    risk_pct: float | None = None,
) -> PositionPlan:
    """כמה מניות לקנות: סיכון קבוע מההון, בלי לעבור 25% מההון ובלי לעבור 1.5% מהמחזור היומי."""
    cfg = load_config()
    a, r = cfg["account"], cfg["risk"]
    capital = capital if capital is not None else a["capital_ils"]
    risk_pct = risk_pct if risk_pct is not None else a["risk_per_trade_pct"]
    risk_per_share = entry - stop
    if risk_per_share <= 0 or entry <= 0:
        return PositionPlan(0, 0.0, 0.0, ["מחיר היציאה בהפסד לא מתחת למחיר הכניסה"])
    limits = {
        "risk": capital * risk_pct / 100.0 / risk_per_share,
        "cap": capital * a["max_position_pct"] / 100.0 / entry,
    }
    if avg_volume_shares and avg_volume_shares > 0:
        limits["liquidity"] = avg_volume_shares * r["max_adv_pct"] / 100.0
    binding = min(limits, key=limits.get)
    shares = math.floor(limits[binding])
    notes = {
        "cap": f"הכמות הוגבלה ל-{a['max_position_pct']:.0f}% מההון",
        "liquidity": "הכמות הוגבלה בגלל מחזור מסחר נמוך יחסית",
    }
    return PositionPlan(
        shares=shares,
        amount_ils=round(shares * entry, 2),
        risk_ils=round(shares * risk_per_share, 2),
        limited_by=[notes[binding]] if binding in notes else [],
    )


def hold_window(as_of: date, max_hold_days: int) -> tuple[date, date]:
    """ימי המסחר שבהם העסקה עשויה להיות פתוחה: מהיום הבא ועד max_hold_days ימי מסחר (שני-שישי)."""
    days = pd.bdate_range(pd.Timestamp(as_of) + pd.Timedelta(days=1), periods=max_hold_days)
    return days[0].date(), days[-1].date()


def event_check(events: StockEvents | None, start: date, end: date) -> list[tuple[str, str]]:
    """אירועים בחלון ההחזקה. אדום: אירוע ידוע. צהוב: תאריך דוח משוער."""
    if events is None:
        return []
    margin = pd.Timedelta(days=load_config()["events"]["estimate_margin_days"])
    out = []
    if events.earnings:
        if not events.earnings_is_estimate and start <= events.earnings <= end:
            out.append((RED, f"דוח כספי צפוי ב-{events.earnings:%d.%m.%Y}, בתוך תקופת ההחזקה. לא נכנסים לפני דוח."))
        elif events.earnings_is_estimate and (
            pd.Timestamp(start) - margin <= pd.Timestamp(events.earnings) <= pd.Timestamp(end) + margin
        ):
            out.append((YELLOW, f"ייתכן שהדוח הכספי הבא יתפרסם בסביבות {events.earnings:%d.%m.%Y} (הערכה). כדאי לבדוק במאיה."))
    if events.ex_dividend and start <= events.ex_dividend <= end:
        out.append((RED, f"אקס-דיבידנד ב-{events.ex_dividend:%d.%m.%Y}, בתוך תקופת ההחזקה. המחיר יירד באותו יום בגובה הדיבידנד."))
    return out


def sector_warnings(candidates: pd.DataFrame, sector_col: str = "sector") -> dict[str, str]:
    """אם יש יותר מ-2 הזדמנויות באותו סקטור: אזהרה על כל אחת מהן מעבר לשתיים הראשונות."""
    limit = load_config()["risk"]["max_per_sector"]
    out = {}
    if candidates.empty or sector_col not in candidates:
        return out
    for sector, group in candidates.groupby(sector_col):
        if not sector or len(group) <= limit:
            continue
        for sym in group["symbol"].iloc[limit:]:
            out[sym] = f"כבר יש {limit} הזדמנויות טובות יותר באותו סקטור. כמה מניות מאותו סקטור הן כמעט אותה עסקה."
    return out
