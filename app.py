"""נקודת הכניסה של האפליקציה.  הרצה:  streamlit run app.py"""
import streamlit as st

from core.config import load_config
from ui import nav, page_advanced, page_check, page_guide, page_scan
from ui.style import apply_style

DEFAULT_TITLE = "סורק מניות"
TITLE = load_config().get("app", {}).get("title") or DEFAULT_TITLE

st.set_page_config(page_title=TITLE, page_icon="📈", layout="centered")
apply_style()

nav.PAGES["scan"] = st.Page(page_scan.render, title="סרוק עכשיו", url_path="scan", default=True)
nav.PAGES["check"] = st.Page(page_check.render, title="בדוק מניה", url_path="check")
nav.PAGES["guide"] = st.Page(page_guide.render, title="איך משתמשים", url_path="guide")
nav.PAGES["advanced"] = st.Page(page_advanced.render, title="מתקדם", url_path="advanced")

# התפריט המובנה מוסתר. במקומו: כותרת ראשית, כפתורים גדולים למעלה, וקישור קטן ל"מתקדם" למטה.
current = st.navigation(list(nav.PAGES.values()), position="hidden")
nav.app_title(TITLE)
nav.nav_bar(current)
current.run()
nav.footer(current)
