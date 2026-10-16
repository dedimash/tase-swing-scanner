"""ניווט: סרגל עליון עם שני כפתורים גדולים, וקישור קטן למסך "מתקדם" בתחתית.

app.py רושם כאן את הדפים. התפריט המובנה של Streamlit מוסתר.
"""
from __future__ import annotations

import streamlit as st

from ui.style import footer_notices

PAGES: dict = {}
MAIN = (("scan", "🔎 סריקה"), ("check", "📊 בדיקת מניה"), ("guide", "❓ הסבר"))


def nav_bar(current) -> None:
    """כפתורים גדולים בראש כל מסך. המסך הנוכחי מסומן במסגרת (ולא בכחול מלא,
    כדי שלא יתבלבל עם כפתור הפעולה של המסך)."""
    with st.container(horizontal=True, gap="small", key="navbar"):
        for name, label in MAIN:
            page = PAGES[name]
            active = current is not None and current.url_path == page.url_path
            key = f"nav_{name}_on" if active else f"nav_{name}"
            if st.button(label, key=key, width="stretch") and not active:
                st.switch_page(page)


def footer(current) -> None:
    footer_notices()
    if current is None or current.url_path != PAGES["advanced"].url_path:
        st.page_link(PAGES["advanced"], label="מתקדם: הגדרות ותוצאות בדיקה", icon="⚙️")


def open_check(symbol: str, name: str) -> None:
    """פותח את "בדוק מניה" על מניה מסוימת."""
    st.session_state["check_query"] = name
    st.session_state["check_symbol"] = symbol
    st.switch_page(PAGES["check"])
