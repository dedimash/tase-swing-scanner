from datetime import date

import numpy as np
import pandas as pd

from core.indicators import typical_turnover
from core.research_prompt import build_prompt
from core.verdict import Verdict


def frame(n=260):
    close = np.linspace(60, 75, n)
    idx = pd.bdate_range(end="2026-10-15", periods=n)
    df = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 2e6}, index=idx)
    df["turnover"] = df["close"] * df["volume"]
    return df


GREEN = Verdict(
    symbol="LUMI.TA", color="green", headline="עומדת בתנאים לעסקה", reason="...",
    close=74.71, setup_key="rsi2_reversion", setup_name="ירידה חדה במניה חזקה",
    entry=74.71, stop=71.2, target=None, exit_rule="יוצאים בסגירה הראשונה מעל הממוצע של 5 הימים האחרונים",
    shares=213, amount_ils=15913.0, risk_ils=747.0, hold_until=date(2026, 10, 30), max_hold_days=10,
)
YELLOW = Verdict(
    symbol="LUMI.TA", color="yellow", headline="מניה במצב סביר, אבל אין עכשיו נקודת כניסה",
    reason="אין כרגע תבנית מתאימה.", close=74.71,
)
TODAY = date(2026, 10, 16)


def test_green_prompt_has_trade_numbers_and_rules():
    p = build_prompt(GREEN, "בנק לאומי לישראל", frame(), today=TODAY, name_en="Bank Leumi le-Israel B.M.")
    assert "16.10.2026" in p and "בנק לאומי לישראל" in p and "LUMI" in p
    assert "Bank Leumi le-Israel B.M." in p
    assert "74.71 ₪" in p and "71.20 ₪" in p and "30.10.2026" in p
    assert "ירידה חדה במניה חזקה" in p
    assert "ועד 10 ימי מסחר" in p and "עד עד" not in p
    assert "maya.tase.co.il" in p and "גלובס" in p
    assert "לא נמצא" in p and "אל תמליץ לקנות או למכור" in p
    assert "נמצא דגל אדום שמצדיק לא להיכנס עכשיו: כן / לא / לא ניתן לוודא" in p
    assert "עד 10 שורות" in p


def test_exact_date_windows():
    p = build_prompt(GREEN, "בנק לאומי לישראל", frame(), today=TODAY)
    assert "מ-02.10.2026 עד היום" in p          # שבועיים אחורה
    assert "מהיום עד 30.10.2026" in p           # עד סוף ההחזקה
    q = build_prompt(YELLOW, "בנק לאומי לישראל", frame(), today=TODAY)
    assert "מהיום עד 30.10.2026" in q           # שבועיים קדימה


def test_definitions_remove_ambiguity():
    p = build_prompt(GREEN, "בנק לאומי לישראל", frame(), today=TODAY)
    assert 'גיוס חוב שגרתי באג"ח לא נחשב' in p
    assert "קנייה עצמית של החברה לא נחשבת" in p
    assert "אירועים כלליים בשוק לא הופכים את התשובה" in p


def test_prompt_never_reveals_capital_or_position_size():
    p = build_prompt(GREEN, "בנק לאומי לישראל", frame(), today=TODAY)
    assert "213" not in p and "15,913" not in p and "747" not in p


def test_non_green_prompt_has_no_entry_or_stop():
    p = build_prompt(YELLOW, "בנק לאומי לישראל", frame(), today=TODAY)
    assert "כניסה משוערת" not in p and "יציאה בהפסד" not in p
    assert "אין כרגע תבנית מתאימה" in p
    assert "נמצא אירוע או דגל אדום שכדאי לדעת עליו לפני עסקה במניה הזו: כן / לא / לא ניתן לוודא" in p


def test_dual_listed_adds_us_question():
    p = build_prompt(YELLOW, "טבע", frame(), us_symbol="TEVA", today=TODAY)
    assert 'נסחרת גם בארה"ב (סימול TEVA)' in p


def test_trend_text_in_uptrend():
    p = build_prompt(YELLOW, "בנק לאומי לישראל", frame(), today=TODAY)
    assert "מגמה עולה" in p


def test_typical_turnover_ignores_single_block_trade():
    df = frame(40)
    df.loc[df.index[-5], "volume"] = 2e8  # עסקת בלוק אחת, פי 100
    df["turnover"] = df["close"] * df["volume"]
    typical = typical_turnover(df, 20).iloc[-1]
    assert typical < 2 * df["turnover"].iloc[-1]
    assert df["turnover"].tail(20).mean() > 5 * typical


def test_cannot_verify_option_and_coverage_rules():
    p = build_prompt(GREEN, "בנק לאומי לישראל", frame(), today=TODAY)
    assert '"לא ניתן לוודא": לא הצלחת לראות אף רשימת דיווחים' in p
    assert "אם ראית רק חלק מהתקופה" in p
    assert "בשורות של \"לא נמצא\" אין צורך בקישור" in p
    assert "כשהחברה עצמה מופיעה כקונה, זו קנייה עצמית" in p
    assert 'גלובס בנק לאומי לישראל דיווחים' in p
    assert "עד שני משפטים" in p
