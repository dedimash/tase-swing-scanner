"""מסך "סרוק עכשיו": כל המניות הנזילות, ורשימה קצרה של מניות שעומדות עכשיו בתנאים לעסקה."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core.research_prompt import build_prompt
from core.risk import sector_warnings
from core.universe import display_name, scan_universe
from core.verdict import GREEN, TZ, YELLOW, evaluate
from ui.data_access import DataUnavailable, market_context, scan_prices, us_change_pct
from ui.nav import open_check
from ui.style import date_he
from ui.chat_prompt import render_chat_prompt
from ui.verdict_view import render_verdict

REGIME_TEXT = {"עולה": "עולה 📈", "מעורב": "מעורב ↔️", "יורד": "יורד 📉"}


def _market_box(ctx) -> None:
    regime = ctx.regime_now
    breadth = ctx.breadth_now
    parts = [f"**מצב השוק:** {REGIME_TEXT.get(regime, 'לא ידוע')}"]
    if breadth is not None:
        parts.append(f"{breadth * 100:.0f}% מהמניות נמצאות מעל הממוצע של 50 הימים האחרונים")
    st.info(" · ".join(parts))
    if ctx.index_note:
        st.caption(f"ⓘ {ctx.index_note}")
    if regime is None:
        st.warning(
            "לא הצלחתי לקבל כרגע את נתוני מדד ת\"א 125. בלי המדד אי אפשר לדעת את מצב השוק, "
            "ולכן כרגע לא יופיע רמזור ירוק. נסי שוב בעוד כמה דקות."
        )


def _run_scan() -> dict:
    ctx = market_context()
    frames = scan_prices()
    uni = scan_universe().set_index("symbol")
    now = datetime.now(TZ)
    verdicts = []
    progress = st.progress(0.0, text="בודקת מניות...")
    symbols = [s for s in uni.index if s in frames]
    for i, sym in enumerate(symbols):
        verdicts.append(evaluate(sym, frames[sym], ctx, now=now))
        if i % 10 == 0:
            progress.progress((i + 1) / len(symbols), text=f"בודקת מניות... {i + 1} מתוך {len(symbols)}")
    progress.empty()

    greens = [v for v in verdicts if v.color == GREEN]
    for v in greens:  # רק לירוקות: אזהרת פער ממניה דואלית
        us = uni.loc[v.symbol, "us_symbol"] if v.symbol in uni.index else ""
        change = us_change_pct(us) if us else None
        if change is not None and abs(change) >= 1.0:
            direction = "עלתה" if change > 0 else "ירדה"
            v.warnings.append(f"המניה נסחרת גם בניו יורק, ושם היא {direction} {abs(change):.1f}%. צפוי פער בפתיחה כאן.")
    greens.sort(key=lambda v: (-(v.expectancy_r or 0), -(v.score or 0)))
    table = pd.DataFrame({"symbol": [v.symbol for v in greens], "sector": [uni.loc[v.symbol, "sector"] for v in greens]})
    for sym, msg in sector_warnings(table).items():
        next(v for v in greens if v.symbol == sym).warnings.append(msg)

    yellows = [v for v in verdicts if v.color == YELLOW and v.waits]
    yellows.sort(key=lambda v: -float(uni.loc[v.symbol, "avg_turnover_ils"]))  # הנזילות קודם
    return {
        "at": now,
        "checked": len(verdicts),
        "greens": greens,
        "yellows": yellows[:15],
        "counts": pd.Series([v.color for v in verdicts]).value_counts().to_dict(),
    }


def render() -> None:
    st.title("סרוק עכשיו")
    try:
        with st.spinner("טוענת את מצב השוק..."):
            ctx = market_context()
    except DataUnavailable:
        st.warning("לא הצלחתי לקבל נתונים כרגע. נסי שוב בעוד כמה דקות.")
        return
    if not ctx.edge:
        st.warning("עוד אין תוצאות בדיקה לתבניות. צריך להריץ במחשב: python scripts/run_backtests.py")
        return
    _market_box(ctx)

    if st.button("🔎 סרוק עכשיו", type="primary", width="stretch"):
        try:
            with st.spinner("מביאה נתונים לכל המניות. בפעם הראשונה זה לוקח קצת זמן..."):
                st.session_state["scan"] = _run_scan()
        except DataUnavailable:
            st.warning("לא הצלחתי לקבל נתונים כרגע. נסי שוב בעוד כמה דקות.")

    result = st.session_state.get("scan")
    if not result:
        st.caption("לוחצים על הכפתור, והמערכת בודקת את כל המניות הנזילות בבורסה.")
        return

    names = scan_universe().set_index("symbol")
    greens, yellows = result["greens"], result["yellows"]
    frames = scan_prices() if greens else {}
    as_of = greens[0].as_of if greens else None
    st.caption(
        f"נבדקו {result['checked']} מניות · הסריקה בוצעה ב-{result['at']:%d.%m.%Y %H:%M}"
        + (f" · לפי הסגירה של {date_he(as_of)}" if as_of else "")
    )

    if not greens:
        st.success("אין היום מניות שעומדות בכל התנאים. זה בסדר גמור: לא חייבים לסחור כל יום.")
    else:
        st.subheader(f"נמצאו {len(greens)} הזדמנויות")
        for v in greens:
            row = names.loc[v.symbol] if v.symbol in names.index else None
            title = display_name(row) if row is not None else v.symbol
            with st.container(border=True):
                st.markdown(f"### {title}")
                render_verdict(v, compact=True)
                with st.expander("🔍 בדיקה בצ'אט לפני שנכנסים"):
                    us = row["us_symbol"] if row is not None else ""
                    name_en = row["name_en"] if row is not None else ""
                    render_chat_prompt(build_prompt(v, title, frames[v.symbol], us, name_en=name_en), key=v.symbol)
                if st.button("לפרטים ולגרף", key=f"open_{v.symbol}", width="stretch"):
                    open_check(v.symbol, title)

    if yellows:
        with st.expander(f"מניות שקרובות לנקודת כניסה ({len(yellows)})"):
            for v in yellows:
                row = names.loc[v.symbol] if v.symbol in names.index else None
                title = display_name(row) if row is not None else v.symbol
                st.markdown(f"**{title}**: " + "; ".join(v.waits))

