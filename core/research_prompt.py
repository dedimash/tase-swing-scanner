"""פרומפט מחקר מוכן להדבקה בצ'אט (Claude או ChatGPT עם חיפוש באינטרנט).

הצ'אט בודק רק דבר אחד: האם יש סיבה לא להיכנס לעסקה. הוא לא מחליט לקנות.
הפונקציה מחזירה טקסט בלבד, כך שבעתיד אפשר לשלוח אותו גם ל-API בלי לשנות שום דבר אחר.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from core.indicators import rsi, sma, typical_turnover
from core.setups_daily import SETUPS
from core.verdict import GREEN, Verdict

NEWS_SITES = "גלובס, כלכליסט, דה מרקר, ביזפורטל"


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:,.2f} ₪"


def _trend_text(df: pd.DataFrame) -> str:
    c = df["close"]
    if len(c) < 200:
        return "אין מספיק היסטוריה"
    close, s50, s200 = c.iloc[-1], sma(c, 50).iloc[-1], sma(c, 200).iloc[-1]
    above = [name for name, level in (("50", s50), ("200", s200)) if close > level]
    if len(above) == 2:
        return "מעל הממוצע של 50 ימים ושל 200 ימים (מגמה עולה)"
    if not above:
        return "מתחת לממוצע של 50 ימים ושל 200 ימים (מגמה יורדת)"
    return f"מעל הממוצע של {above[0]} ימים בלבד (מגמה לא ברורה)"


def _market_data(df: pd.DataFrame) -> str:
    parts = [f"מגמה: {_trend_text(df)}"]
    if len(df) > 20:
        parts.append(f"RSI(14): {rsi(df['close'], 14).iloc[-1]:.0f}")
        turnover = typical_turnover(df, 20).iloc[-1]
        if pd.notna(turnover):
            parts.append(f"מחזור מסחר ביום רגיל: {turnover / 1e6:,.1f} מיליון ₪")
    return ". ".join(parts) + "."


def build_prompt(
    verdict: Verdict,
    name: str,
    df: pd.DataFrame,
    us_symbol: str = "",
    today: date | None = None,
    name_en: str = "",
) -> str:
    today = today or date.today()
    ticker = verdict.symbol.removesuffix(".TA")
    english = f"{name_en}, " if name_en and name_en != name else ""
    identity = f"{name} ({english}סימול {ticker}, בורסת תל אביב)"
    look_back = today - timedelta(days=14)
    lines: list[str] = []

    if verdict.color == GREEN:
        setup = SETUPS.get(verdict.setup_key)
        description = setup.description_he if setup else ""
        horizon = f"{verdict.max_hold_days} ימי מסחר" if verdict.max_hold_days else "כמה ימים"
        lines.append(f"היום {today:%d.%m.%Y}. אני שוקלת עסקת סווינג קצרה (מכמה שעות ועד {horizon}) במניית {identity}.")
        lines.append(f'המערכת שלי זיהתה את התבנית "{verdict.setup_name}": {description}')
        exit_line = f"מחיר יעד: {_fmt(verdict.target)}" if verdict.target else f"יציאה ברווח: {verdict.exit_rule}"
        lines.append(
            f"מחיר אחרון: {_fmt(verdict.close)}. כניסה משוערת: {_fmt(verdict.entry)}. "
            f"יציאה בהפסד: {_fmt(verdict.stop)}. {exit_line}. "
            f"החזקה עד {verdict.hold_until:%d.%m.%Y} לכל המאוחר."
        )
        look_ahead = verdict.hold_until
        final_question = "נמצא דגל אדום שמצדיק לא להיכנס עכשיו: כן / לא / לא ניתן לוודא"
    else:
        lines.append(
            f"היום {today:%d.%m.%Y}. אני בודקת את מניית {identity} "
            "לקראת עסקת סווינג קצרה אפשרית (מכמה שעות ועד כמה ימים)."
        )
        lines.append(f"מחיר אחרון: {_fmt(verdict.close)}. מה המערכת שלי אומרת: {verdict.headline}. {verdict.reason}")
        look_ahead = today + timedelta(days=14)
        final_question = "נמצא אירוע או דגל אדום שכדאי לדעת עליו לפני עסקה במניה הזו: כן / לא / לא ניתן לוודא"

    past = f"מ-{look_back:%d.%m.%Y} עד היום"
    ahead = f"מהיום עד {look_ahead:%d.%m.%Y}"
    lines.append(f"נתונים: {_market_data(df)}")
    lines.append("")
    lines.append("חפש באינטרנט ובדוק:")
    tasks = [
        f"דיווחים מיידיים של החברה {past}.",
        f"אירועים של החברה {ahead}: דוח כספי, אסיפת בעלי מניות, תאריך אקס-דיבידנד, פיצול, או אירוע מתוכנן אחר.",
        f"חדשות על החברה {past} שעלולות להשפיע על המניה בימים הקרובים."
        + (" אם המניה ירדה בחדות לאחרונה, האם יש ידיעה שמסבירה את הירידה?" if verdict.color == GREEN else ""),
        "דגלים אדומים: הנפקת מניות (גיוס חוב שגרתי באג\"ח לא נחשב), מכירה מהותית של בעלי עניין "
        "(קנייה עצמית של החברה לא נחשבת), הליך משפטי מהותי, הורדת דירוג, אזהרת רווח.",
    ]
    if us_symbol:
        tasks.append(f'המניה נסחרת גם בארה"ב (סימול {us_symbol}). מה קרה שם ביום-יומיים האחרונים?')
    tasks.append(f"אירועים כלליים בשוק {ahead} (למשל החלטת ריבית של בנק ישראל): רק לציין, בשורה נפרדת.")
    lines += [f"{i}. {t}" for i, t in enumerate(tasks, 1)]
    lines.append("")
    lines.append("איפה לחפש:")
    lines += [
        "- באתר מאיה של הבורסה (maya.tase.co.il). מאיה לא תמיד נפתח בחיפוש אוטומטי.",
        f'- אם מאיה לא נפתח: בדף המניה באתר גלובס יש את רשימת הדיווחים ואת עסקאות בעלי העניין. '
        f'אפשר למצוא אותו בחיפוש "גלובס {name} דיווחים".',
        "- בטבלת עסקאות בעלי העניין: כשהחברה עצמה מופיעה כקונה, זו קנייה עצמית ולא דגל אדום.",
        f"- אתרי חדשות כלכליות: {NEWS_SITES}.",
        "- העדף מקורות רשמיים ואתרי חדשות מוכרים. כשהמקור הוא אתר שאוסף מידע מאתרים אחרים, ציין את זה.",
    ]
    lines.append("")
    lines.append("כללים לתשובה:")
    lines += [
        "- לכל ממצא צרף קישור למקור ותאריך. בשורות של \"לא נמצא\" אין צורך בקישור.",
        '- אם לא מצאת מידע בנושא, כתוב "לא נמצא". אל תנחש ואל תשלים מהזיכרון.',
        "- אם ראית רק חלק מהתקופה (למשל רשימת דיווחים חלקית), כתוב את זה.",
        "- אל תמליץ לקנות או למכור, ואל תעריך לאן המחיר ילך.",
        "- כתוב בעברית פשוטה וקצרה. עד 10 שורות בטבלה.",
        "- דגל אדום \"מהותי\" = משהו שהחברה דיווחה עליו בדיווח מיידי, או שהופיע בכותרת של אתר חדשות כלכלי.",
    ]
    lines.append("")
    lines.append("התשובה בשורה האחרונה:")
    lines += [
        f'- "כן": נמצא דוח כספי או תאריך אקס-דיבידנד של החברה {ahead}, או דגל אדום מהותי מסעיף 4.',
        '- "לא ניתן לוודא": לא הצלחת לראות אף רשימת דיווחים של החברה (לא במאיה ולא בגלובס), '
        "או שיש ידיעה שנראית כמו דגל אדום ולא הצלחת לבדוק אותה.",
        '- "לא": בכל מקרה אחר. אירועים כלליים בשוק לא הופכים את התשובה ל"כן".',
    ]
    lines.append("")
    lines.append("מבנה התשובה:")
    lines.append("- טבלה קצרה: נושא | מה נמצא | תאריך | קישור.")
    lines.append(f'- שורה אחרונה: "{final_question}", ועד שני משפטים שמסבירים למה (כולל מה לא הצלחת לבדוק).')
    return "\n".join(lines)
