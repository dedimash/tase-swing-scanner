"""עיצוב משותף: כיווניות מימין לשמאל, טקסט גדול, ועיצוב מספרים בעברית."""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

RTL_CSS = """
<style>
html, body, .stApp { font-size: 18px; }
.stApp, [data-testid="stSidebar"], [data-testid="stHeader"] { direction: rtl; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stMetric"] { text-align: right; }
.stApp input { direction: rtl; text-align: right; font-size: 1.1rem; }
.stApp button p { font-size: 1.1rem; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], .stApp pre, .stApp code { direction: ltr; }
[data-testid="stMetricValue"] { direction: rtl; text-align: right; }
[data-testid="stMetricDelta"] { direction: ltr; width: fit-content; margin-left: auto; }
[data-testid="stSlider"] [data-baseweb="slider"] { direction: ltr; }
.stApp label[data-baseweb="checkbox"] { gap: 0.6rem; }

/* ניווט: מסתירים את הכותרת של Streamlit, וסרגל הכפתורים נשאר למעלה גם בגלילה */
[data-testid="stHeader"] { display: none; }
[data-testid="stMainBlockContainer"] { padding-top: 0.75rem; }
/* Streamlit עוטף את הסרגל באלמנט בגובה שלו, ולכן ה-sticky הולך על העוטף */
[data-testid="stLayoutWrapper"]:has(> .st-key-navbar) {
  position: sticky; top: 0; z-index: 990; background: #ffffff;
  padding: 0.5rem 0 0.75rem; border-bottom: 1px solid #e5e7eb;
}
.st-key-navbar button { min-height: 3.4rem; border-radius: 0.75rem; }
.st-key-navbar button p { font-size: 1.25rem; font-weight: 600; }
.st-key-nav_scan_on button, .st-key-nav_check_on button, .st-key-nav_guide_on button {
  background: #e8f0fe; border: 2px solid #1f6feb; color: #1f6feb; cursor: default;
}

/* כפתורים ראשיים גדולים ונוחים ללחיצה */
.stApp button[kind="primary"] { min-height: 3rem; }
[data-testid="stPageLink"] p { font-size: 0.95rem; }

/* בטלפון: שני מספרים בשורה במקום אחד מתחת לשני */
@media (max-width: 640px) {
  [class*="st-key-grid_"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 0.75rem; }
  [class*="st-key-grid_"] [data-testid="stColumn"] { min-width: 0 !important; width: 50% !important; flex: 1 1 50% !important; }
  [data-testid="stMetricValue"] { font-size: 1.55rem; }
  [data-testid="stMetricLabel"] p { font-size: 0.9rem; }
  .stApp h1 { font-size: 2rem; }
  .st-key-navbar { flex-wrap: nowrap !important; gap: 0.4rem !important; }
  .st-key-navbar > div { flex: 1 1 0 !important; min-width: 0 !important; width: auto !important; }
  .st-key-navbar button { padding-left: 0.3rem; padding-right: 0.3rem; min-height: 3.2rem; }
  .st-key-navbar button p { font-size: 0.95rem; white-space: normal; line-height: 1.2; }
}
</style>
"""

DELAY_NOTICE = "המחירים עשויים להיות מושהים. לפני כניסה לעסקה יש לבדוק את המחיר בפלטפורמת המסחר."
ADVICE_NOTICE = "הכלי בודק עמידה בכללים ואינו ייעוץ השקעות."


def apply_style() -> None:
    st.markdown(RTL_CSS, unsafe_allow_html=True)


def footer_notices() -> None:
    st.divider()
    st.caption(f"ⓘ {DELAY_NOTICE}")
    st.caption(f"ⓘ {ADVICE_NOTICE}")


def ltr(text: str) -> str:
    """מבודד מספר (עם סימן +/-) כדי שלא יתהפך בתוך טקסט עברי."""
    return f"⁦{text}⁩"


def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def ils(x, decimals: int = 2) -> str:
    return "—" if _bad(x) else f"{x:,.{decimals}f} ₪"


def ils_large(x) -> str:
    """סכום גדול במילים: 217 מיליון ₪."""
    if _bad(x):
        return "—"
    if x >= 1e9:
        return f"{x / 1e9:,.1f} מיליארד ₪"
    if x >= 1e6:
        return f"{x / 1e6:,.1f} מיליון ₪"
    if x >= 1e3:
        return f"{x / 1e3:,.0f} אלף ₪"
    return f"{x:,.0f} ₪"


def pct(x, decimals: int = 1, signed: bool = False) -> str:
    if _bad(x):
        return "—"
    return ltr(f"{x:+.{decimals}f}%" if signed else f"{x:.{decimals}f}%")


def date_he(d) -> str:
    return "—" if d is None else pd.Timestamp(d).strftime("%d.%m.%Y")


def out_of_ten(win_rate) -> str:
    """'בעבר זה הצליח ב-6 מתוך 10 פעמים'."""
    if _bad(win_rate):
        return "אין מספיק נתונים"
    return f"בעבר זה הצליח ב-{round(win_rate * 10)} מתוך 10 פעמים"
