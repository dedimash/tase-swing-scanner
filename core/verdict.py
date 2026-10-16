"""הרמזור: האם מניה עומדת עכשיו בתנאים לעסקה, ומה עושים.

ירוק  = יש תבנית פעילה (רק תבנית שעברה את הבדיקה), וכל השערים עברו.
צהוב  = אין סיבה ברורה לא לגעת, אבל גם אין נקודת כניסה. מסבירים למה לחכות.
אדום  = יש סיבה ברורה לא להיכנס עכשיו.

הכלי לא אומר "כדאי לקנות". הוא אומר אם המניה עומדת בכללים ומה הסיכון.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.config import load_config
from core.events import StockEvents
from core.indicators import sma, typical_turnover, typical_volume
from core.regime import DOWN, regime_on
from core.risk import RED, YELLOW, event_check, hold_window, plan_position
from core.scoring import score_signals
from core.setups_daily import SETUPS, Setup

GREEN = "green"
HEADLINES = {
    GREEN: "עומדת בתנאים לעסקה",
    YELLOW: "מניה במצב סביר, אבל אין עכשיו נקודת כניסה",
    RED: "לא עכשיו",
}
TZ = ZoneInfo("Asia/Jerusalem")
MIN_BARS = 220


@dataclass
class Verdict:
    symbol: str
    color: str
    headline: str
    reason: str
    as_of: date | None = None
    close: float | None = None
    setup_key: str | None = None
    setup_name: str | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    exit_rule: str | None = None
    shares: int = 0
    amount_ils: float = 0.0
    risk_ils: float = 0.0
    hold_until: date | None = None
    max_hold_days: int | None = None
    past_performance: str | None = None
    expectancy_r: float | None = None
    score: float | None = None
    score_parts: dict = field(default_factory=dict)
    waits: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    partial_today: bool = False


@dataclass
class MarketContext:
    """מה שמשותף לכל המניות: מדד, משטר, רוחב שוק, ימים חריגים, אירועים, ותוצאות הבקטסט."""
    index_df: pd.DataFrame
    regime: pd.Series
    breadth: pd.Series
    event_days: pd.DatetimeIndex
    events: dict[str, StockEvents]
    edge: dict  # תוכן results/edge_cards.json

    @property
    def regime_now(self) -> str | None:
        return None if self.regime.empty else self.regime.dropna().iloc[-1]

    @property
    def breadth_now(self) -> float | None:
        return None if self.breadth.empty else float(self.breadth.iloc[-1])


def active_setups(edge: dict) -> list[Setup]:
    """רק תבניות שעברו את בדיקת ה-walk-forward, עם הפרמטרים שנבחרו בתקופת הכיול."""
    out = []
    for key, info in (edge or {}).get("setups", {}).items():
        if info.get("passed") and key in SETUPS:
            out.append(SETUPS[key](**info.get("walk_forward", {}).get("best_params", {})))
    return out


def market_is_open(now: datetime | None = None) -> bool:
    cfg = load_config()["market_hours"]
    now = now or datetime.now(TZ)
    if now.weekday() in (5, 6):  # שבת או ראשון
        return False
    close_str = cfg["friday_close"] if now.weekday() == 4 else cfg["weekday_close"]
    opens = time.fromisoformat(cfg["open"])
    closes = time.fromisoformat(close_str)
    return opens <= now.time() <= closes


def drop_partial_bar(df: pd.DataFrame, now: datetime | None = None) -> tuple[pd.DataFrame, bool]:
    """בזמן המסחר הנר של היום עוד לא נסגר. האיתותים מחושבים לפי הנר האחרון שנסגר."""
    now = now or datetime.now(TZ)
    if len(df) and df.index[-1].date() == now.date() and market_is_open(now):
        return df.iloc[:-1], True
    return df, False


def _past_performance(edge: dict, key: str) -> tuple[str | None, float | None]:
    card = edge.get("setups", {}).get(key, {}).get("walk_forward", {}).get("combined", {})
    win = card.get("win_rate")
    if win is None:
        return None, None
    return f"בעבר התבנית הזו הצליחה ב-{round(win * 10)} מתוך 10 פעמים", card.get("expectancy_r")


def _exit_rule(setup: Setup) -> str | None:
    if setup.params.get("exit_ma"):
        return f"יוצאים בסגירה הראשונה מעל הממוצע של {int(setup.params['exit_ma'])} הימים האחרונים"
    return None


def evaluate(
    symbol: str,
    df: pd.DataFrame | None,
    ctx: MarketContext,
    us_change_pct: float | None = None,
    now: datetime | None = None,
) -> Verdict:
    def verdict(color, reason, **kw):
        return Verdict(symbol=symbol, color=color, headline=HEADLINES[color], reason=reason, **kw)

    if df is None or len(df) < MIN_BARS:
        return verdict(RED, "אין מספיק נתונים למניה הזו.")
    df, partial = drop_partial_bar(df, now)
    as_of = df.index[-1].date()
    close = float(df["close"].iloc[-1])
    base = {"as_of": as_of, "close": close, "partial_today": partial}
    warnings = []
    if partial:
        warnings.append("המסחר היום עוד לא נגמר. הבדיקה לפי הסגירה של יום המסחר הקודם.")
    if us_change_pct is not None and abs(us_change_pct) >= 1.0:
        direction = "עלתה" if us_change_pct > 0 else "ירדה"
        warnings.append(f"המניה נסחרת גם בניו יורק, ושם היא {direction} {abs(us_change_pct):.1f}% מאז. צפוי פער בפתיחה כאן.")

    uni = load_config()["universe"]
    turnover = typical_turnover(df, uni["turnover_window_days"]).iloc[-1]
    if not turnover >= uni["min_avg_turnover_ils"]:
        return verdict(RED, "מחזור המסחר במניה נמוך מדי. קשה לקנות ולמכור בה בלי להפסיד על המחיר.", warnings=warnings, **base)

    regime_now = regime_on(pd.DatetimeIndex([df.index[-1]]), ctx.regime).iloc[0] if not ctx.regime.empty else None
    if regime_now == DOWN:
        return verdict(RED, "השוק כולו במגמת ירידה. בתקופה כזו לא נכנסים לעסקאות חדשות.", warnings=warnings, **base)

    c = df["close"]
    s50, s200 = sma(c, 50).iloc[-1], sma(c, 200).iloc[-1]
    if close < s50 and close < s200:
        return verdict(RED, "המניה במגמת ירידה: היא מתחת לממוצע של 50 ימים ושל 200 ימים.", warnings=warnings, **base)

    index_close = ctx.index_df["close"] if ctx.index_df is not None and not ctx.index_df.empty else None
    candidates = []
    from core.backtest import allowed_regimes

    is_event_day = df.index[-1] in ctx.event_days
    for setup in active_setups(ctx.edge):
        sig = setup.detect(df, index_close)
        if sig.empty or sig.index[-1] != df.index[-1]:
            continue
        allowed = allowed_regimes(setup.params.get("regime", "any"))
        if allowed is not None and regime_now not in allowed:
            continue
        if setup.uses_volume_confirmation and is_event_day:
            warnings.append("היום היה מחזור חריג בכל השוק (כנראה עדכון מדדים), ולכן פריצה במחזור גבוה לא נחשבת.")
            continue
        scored = score_signals(sig.iloc[[-1]], df, index_close, ctx.regime, ctx.breadth)
        candidates.append((setup, sig.iloc[-1], scored.iloc[-1]))

    if candidates:
        def priority(item):
            return _past_performance(ctx.edge, item[0].key)[1] or 0.0

        setup, sig, scored = max(candidates, key=priority)
        hold_days = int(setup.params["max_hold_days"])
        start, end = hold_window(as_of, hold_days)
        problems = event_check(ctx.events.get(symbol), start, end)
        reds = [m for lvl, m in problems if lvl == RED]
        warnings += [m for lvl, m in problems if lvl == YELLOW]
        if reds:
            return verdict(RED, reds[0], setup_key=setup.key, setup_name=setup.name_he, warnings=warnings, **base)

        entry, stop = close, float(sig["stop"])
        target_r = sig.get("target_r", np.nan)
        target = entry + float(target_r) * (entry - stop) if pd.notna(target_r) else None
        avg_shares = typical_volume(df, 20).iloc[-1]
        plan = plan_position(entry, stop, avg_shares)
        warnings += plan.limited_by
        past, expectancy = _past_performance(ctx.edge, setup.key)
        return verdict(
            GREEN,
            setup.description_he,
            setup_key=setup.key, setup_name=setup.name_he,
            entry=round(entry, 2), stop=round(stop, 2), target=round(target, 2) if target else None,
            exit_rule=_exit_rule(setup),
            shares=plan.shares, amount_ils=plan.amount_ils, risk_ils=plan.risk_ils,
            hold_until=end, max_hold_days=hold_days,
            past_performance=past, expectancy_r=expectancy,
            score=float(scored["score"]),
            score_parts={k: round(float(scored[k]), 2) for k in scored.index if k != "score"},
            warnings=warnings, **base,
        )

    # מה לחכות לו: רק מתבניות שמותרות במצב השוק הנוכחי
    allowed_now = [
        s for s in active_setups(ctx.edge)
        if (lambda a: a is None or regime_now in a)(allowed_regimes(s.params.get("regime", "any")))
    ]
    waits = [w for w in (s.watch(df) for s in allowed_now) if w]
    if waits:
        return verdict(YELLOW, "המניה קרובה לאחת התבניות, אבל עוד אין איתות.", waits=waits, warnings=warnings, **base)
    return verdict(YELLOW, "אין כרגע תבנית מתאימה במניה הזו. אין בזה שום דבר רע, פשוט לא עכשיו.", warnings=warnings, **base)
