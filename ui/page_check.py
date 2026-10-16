"""מסך "בדוק מניה": חיפוש בעברית, רמזור, ומה עושים."""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from core.indicators import atr, ema, rsi, sma, typical_turnover
from core.research_prompt import build_prompt
from core.scoring import COMPONENT_NAMES_HE
from core.universe import display_name, search
from core.verdict import TZ, evaluate
from ui.data_access import DataUnavailable, daily_prices, market_context, us_change_pct
from ui.style import date_he, ils, ils_large
from ui.chat_prompt import render_chat_prompt
from ui.verdict_view import render_verdict, verdict_chart


def render() -> None:
    st.title("בדוק מניה")
    query = st.text_input("שם המניה", placeholder="למשל: לאומי", key="check_query")
    if not query.strip():
        st.caption("אפשר לכתוב שם בעברית, שם באנגלית או סימול.")
        return

    matches = search(query, limit=6)
    if matches.empty:
        st.info("לא מצאתי מניה בשם הזה. נסי לכתוב רק חלק מהשם.")
        return

    labels = {row["symbol"]: f"{display_name(row)} ({row['symbol'].removesuffix('.TA')})" for _, row in matches.iterrows()}
    wanted = st.session_state.pop("check_symbol", None)
    options = list(labels)
    symbol = wanted if wanted in labels else options[0]
    if len(options) > 1:
        symbol = st.radio("התכוונת ל:", options, index=options.index(symbol), format_func=labels.get)
    row = matches.set_index("symbol").loc[symbol]

    try:
        with st.spinner("מביאה נתונים..."):
            df = daily_prices(symbol)
            ctx = market_context()
    except DataUnavailable:
        st.warning("לא הצלחתי לקבל נתונים כרגע. נסי שוב בעוד כמה דקות.")
        return

    us = row.get("us_symbol") or ""
    verdict = evaluate(symbol, df, ctx, us_change_pct=us_change_pct(us) if us else None, now=datetime.now(TZ))

    st.subheader(display_name(row))
    if len(df) >= 2:
        last, prev = df.iloc[-1], df.iloc[-2]
        with st.container(key="grid_price"):
            c1, c2 = st.columns(2)
            c1.metric("מחיר אחרון", ils(last["close"]), f"{(last['close'] / prev['close'] - 1) * 100:+.1f}%")
            c2.metric("מחזור מסחר ביום רגיל", ils_large(typical_turnover(df, 20).iloc[-1]), help="החציון של 20 ימי המסחר האחרונים, כדי שעסקה גדולה אחת לא תעוות את התמונה.")
        st.caption(
            f"נכון לסגירה של {date_he(verdict.as_of or df.index[-1])} · "
            f"ממוצע 5 הימים האחרונים: {ils(sma(df['close'], 5).iloc[-1])}"
        )

    render_verdict(verdict)
    if ctx.index_note:
        st.caption(f"ⓘ {ctx.index_note}")
    if ctx.regime_now is None:
        st.caption("⚠ נתוני מדד ת\"א 125 לא התקבלו כרגע, ולכן לא יופיע רמזור ירוק. נסי שוב בעוד כמה דקות.")
    st.markdown("#### 🔍 בדיקה בצ'אט")
    render_chat_prompt(build_prompt(verdict, display_name(row), df, us, name_en=row.get("name_en") or ""), key=symbol)
    if not ctx.edge:
        st.caption("⚠ עוד אין תוצאות בדיקה לתבניות, ולכן אין רמזור ירוק. צריך להריץ: python scripts/run_backtests.py")

    with st.expander("פרטים למתקדמים"):
        show_lines = st.checkbox("להציג ממוצעים נעים על הגרף", value=False)
        if len(df) > 60:
            a = atr(df, 14).iloc[-1]
            close = df["close"].iloc[-1]
            st.write(
                f"RSI(2): {rsi(df['close'], 2).iloc[-1]:.0f} · RSI(14): {rsi(df['close'], 14).iloc[-1]:.0f} · "
                f"ATR(14): {ils(a)} ({a / close * 100:.1f}%) · EMA20: {ils(ema(df['close'], 20).iloc[-1])} · "
                f"EMA50: {ils(ema(df['close'], 50).iloc[-1])}"
            )
        if verdict.score is not None:
            parts = " · ".join(f"{COMPONENT_NAMES_HE.get(k, k)}: {v:.2f}" for k, v in verdict.score_parts.items())
            st.write(f"ציון: {verdict.score:.0f}/100 ({parts})")
            st.caption("בבדיקה על העבר, הציון לא ניבא אילו עסקאות יצליחו. הוא משמש רק לסדר את הרשימה.")
        ev = ctx.events.get(symbol)
        if ev and ev.earnings:
            kind = "הערכה" if ev.earnings_is_estimate else "תאריך ידוע"
            st.write(f"דוח כספי הבא: {date_he(ev.earnings)} ({kind})")
        if us:
            st.write(f"נסחרת גם בארה\"ב בסימול {us}")
    st.plotly_chart(verdict_chart(df, verdict, show_lines), width="stretch", config={"displayModeBar": False})
