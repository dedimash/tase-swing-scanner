"""תצוגת הרמזור, משותפת למסך הסריקה ולמסך בדיקת מניה."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.indicators import ema
from core.verdict import GREEN, RED, YELLOW, Verdict
from ui.style import date_he, ils

ICONS = {GREEN: "🟢", YELLOW: "🟡", RED: "🔴"}
BOX = {GREEN: st.success, YELLOW: st.warning, RED: st.error}

HELP = {
    "entry": "קונים בפתיחה של יום המסחר הבא. המחיר כאן הוא הסגירה האחרונה, אז מחר הוא יכול להיות קצת שונה.",
    "stop": "אם המחיר יורד עד לכאן, מוכרים. ככה ההפסד נשאר קטן ומתוכנן מראש.",
    "target": "אם המחיר מגיע עד לכאן, מוכרים ברווח.",
    "shares": "לפי ההון שלך והסיכון שנקבע לכל עסקה. ההון שהוגדר: ראו במסך מתקדם.",
    "risk": "זה הסכום שתפסידי אם המחיר יירד עד מחיר היציאה בהפסד.",
    "hold": "אם עד היום הזה העסקה לא הגיעה ליעד וגם לא ליציאה בהפסד, מוכרים בסגירה של אותו יום.",
}


def render_verdict(v: Verdict, compact: bool = False) -> None:
    BOX[v.color](f"{ICONS[v.color]} **{v.headline}**\n\n{v.reason}")

    if v.color == GREEN:
        with st.container(key=f"grid_{v.symbol}"):  # בטלפון: שני מספרים בשורה
            c1, c2 = st.columns(2)
            c1.metric("מחיר כניסה (בערך)", ils(v.entry), help=HELP["entry"])
            c2.metric("מחיר יציאה בהפסד", ils(v.stop), help=HELP["stop"])
            c1, c2 = st.columns(2)
            if v.target:
                c1.metric("מחיר יעד", ils(v.target), help=HELP["target"])
            else:
                c1.metric("מתי מוכרים ברווח", "לפי הכלל למטה", help=v.exit_rule)
            c2.metric("כמה מניות לקנות", f"{v.shares:,}", help=HELP["shares"])
            c1, c2 = st.columns(2)
            c1.metric("כמה כסף אפשר להפסיד", ils(v.risk_ils, 0), help=HELP["risk"])
            c2.metric("עד מתי להחזיק", date_he(v.hold_until), help=HELP["hold"])
        st.caption(f"סכום הקנייה: {ils(v.amount_ils, 0)} · {v.setup_name}")
        if v.exit_rule:
            st.write(f"**מתי מוכרים ברווח:** {v.exit_rule}.")
        if v.past_performance:
            st.write(f"**{v.past_performance}.**")
        if not compact:
            st.markdown("**מה עושים עכשיו:**")
            steps = [
                "בודקים את המחיר העדכני בפלטפורמת המסחר.",
                "מעתיקים את הטקסט לצ'אט (הכפתור הכחול למטה) ובודקים שאין סיבה לא להיכנס.",
                f"אם נכנסים: מגדירים מיד פקודת יציאה בהפסד במחיר {ils(v.stop)}.",
            ]
            st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))

    if v.waits:
        st.markdown("**למה לחכות:**")
        st.markdown("\n".join(f"- {w}" for w in v.waits))

    for w in v.warnings:
        st.caption(f"⚠ {w}")


def verdict_chart(df: pd.DataFrame, v: Verdict | None, show_lines: bool) -> go.Figure:
    recent = df.iloc[-125:]
    fig = go.Figure(
        go.Candlestick(
            x=recent.index, open=recent["open"], high=recent["high"], low=recent["low"], close=recent["close"],
            name="מחיר", increasing_line_color="#1a7f37", decreasing_line_color="#cf222e",
        )
    )
    if show_lines:
        fig.add_scatter(x=recent.index, y=ema(df["close"], 20).iloc[-125:], name="ממוצע 20 ימים", line=dict(width=1.5, color="#8250df"))
        fig.add_scatter(x=recent.index, y=ema(df["close"], 50).iloc[-125:], name="ממוצע 50 ימים", line=dict(width=1.5, color="#bf8700"))
    if v is not None and v.color == GREEN:
        for value, label, color in ((v.entry, "כניסה", "#1f6feb"), (v.stop, "יציאה בהפסד", "#cf222e"), (v.target, "יעד", "#1a7f37")):
            if value:
                fig.add_hline(y=value, line=dict(color=color, width=2, dash="dash"),
                              annotation_text=f"{label} {value:,.2f}", annotation_position="top left")
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
        yaxis_title="₪", legend=dict(orientation="h", y=1.08), hovermode="x unified",
        xaxis=dict(tickformat="%m/%Y", hoverformat="%d.%m.%Y"),
    )
    return fig
