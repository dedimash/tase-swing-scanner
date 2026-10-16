"""כפתור "העתק פרומפט לצ'אט": העתקה בלחיצה אחת, קישורים לצ'אטים, והטקסט המלא כגיבוי."""
from __future__ import annotations

import html
import json

import streamlit as st
import streamlit.components.v1 as components

CHATS = (("פתיחת Claude", "https://claude.ai/new"), ("פתיחת ChatGPT", "https://chatgpt.com/"))

_BUTTON = """
<div dir="rtl" style="font-family: 'Source Sans Pro', Arial, sans-serif;">
  <button id="copy" style="width:100%; min-height:52px; border-radius:12px; border:0; cursor:pointer;
    background:#1f6feb; color:#fff; font-size:19px; font-weight:600;">📋 העתק את הטקסט לצ'אט</button>
  <textarea id="text" readonly style="position:absolute; left:-9999px; top:0;">{text}</textarea>
</div>
<script>
const button = document.getElementById("copy");
const area = document.getElementById("text");
const done = (ok) => {{
  button.textContent = ok ? "✅ הועתק! עכשיו מדביקים בצ'אט" : "לא הצלחתי להעתיק. אפשר להעתיק מהטקסט שלמטה";
  button.style.background = ok ? "#1a7f37" : "#9a6700";
}};
button.addEventListener("click", async () => {{
  let ok = false;
  try {{ await navigator.clipboard.writeText({js_text}); ok = true; }} catch (e) {{}}
  if (!ok) {{
    area.focus(); area.select(); area.setSelectionRange(0, area.value.length);
    try {{ ok = document.execCommand("copy"); }} catch (e) {{ ok = false; }}
  }}
  done(ok);
}});
</script>
"""


def render_chat_prompt(prompt: str, key: str) -> None:
    st.markdown(
        "1. לוחצים על הכפתור הכחול.\n"
        "2. פותחים את Claude או ChatGPT, ומוודאים שהחיפוש באינטרנט פעיל.\n"
        "3. מדביקים ושולחים, וקוראים את השורה האחרונה בתשובה.\n"
        "4. **אם כתוב שם \"כן\" או \"לא ניתן לוודא\", לא נכנסים.**"
    )
    js_text = json.dumps(prompt).replace("</", "<\\/")  # שלא ייסגר תג ה-script בטעות
    components.html(_BUTTON.format(text=html.escape(prompt), js_text=js_text), height=64)
    with st.container(horizontal=True, gap="small", key=f"chats_{key}"):
        for label, url in CHATS:
            st.link_button(label, url, width="stretch")
    with st.expander("להציג את הטקסט (אם ההעתקה לא עבדה)"):
        st.code(prompt, language=None, wrap_lines=True)
