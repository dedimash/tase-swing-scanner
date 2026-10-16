"""בדיקות תצוגה: הרמזור הירוק מוצג עם כל השדות, בלי שגיאות."""
from streamlit.testing.v1 import AppTest


def _green_app():
    from datetime import date

    from core.verdict import Verdict
    from ui.verdict_view import render_verdict

    v = Verdict(
        symbol="LUMI.TA", color="green", headline="עומדת בתנאים לעסקה", reason="מניה במגמה עולה ירדה בחדות.",
        as_of=date(2026, 10, 15), close=74.71, setup_key="rsi2_reversion", setup_name="ירידה חדה במניה חזקה",
        entry=74.71, stop=71.20, target=None, exit_rule="יוצאים בסגירה הראשונה מעל הממוצע של 5 הימים האחרונים",
        shares=213, amount_ils=15913.23, risk_ils=747.63, hold_until=date(2026, 10, 30), max_hold_days=10,
        past_performance="בעבר התבנית הזו הצליחה ב-7 מתוך 10 פעמים", warnings=["אזהרה לדוגמה"],
    )
    render_verdict(v)


def _chat_app():
    from ui.chat_prompt import render_chat_prompt

    render_chat_prompt("שורה ראשונה\n</script> בדיקה\nשורה אחרונה", key="TEST")


def test_chat_prompt_renders_with_fallback_text():
    at = AppTest.from_function(_chat_app).run(timeout=30)
    assert not at.exception
    assert any("שורה אחרונה" in c.value for c in at.code)


def _guide_app():
    from ui.page_guide import render

    render()


def test_guide_renders_with_real_numbers():
    at = AppTest.from_function(_guide_app).run(timeout=30)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "מתי לסרוק" in " ".join(h.value for h in at.subheader)
    assert "לא ניתן לוודא" in text
    assert len(at.expander) >= 5


def test_green_verdict_renders_all_fields():
    at = AppTest.from_function(_green_app).run(timeout=30)
    assert not at.exception
    labels = [m.label for m in at.metric]
    for label in ("מחיר כניסה (בערך)", "מחיר יציאה בהפסד", "כמה מניות לקנות", "כמה כסף אפשר להפסיד", "עד מתי להחזיק"):
        assert label in labels
    values = {m.label: m.value for m in at.metric}
    assert values["כמה מניות לקנות"] == "213"
    assert "71.20" in values["מחיר יציאה בהפסד"]
    text = " ".join(m.value for m in at.markdown)
    assert "מה עושים עכשיו" in text and "71.20" in text
    assert "7 מתוך 10" in text
