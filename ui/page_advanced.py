"""מסך "מתקדם": מעבדת בקטסט, איכות נתונים והגדרות."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.backtest import edge_card, portfolio, run_setup
from core.config import RESULTS_DIR, load_config
from core.setups_daily import SETUPS
from core.universe import load_universe
from ui.data_access import cached_frames, edge_cards, trades
from ui.style import date_he, ils, ltr, out_of_ten, pct

REASONS_HE = {
    "target": "הגיעה ליעד", "gap_target": "קפצה מעל היעד", "stop": "יציאה בהפסד",
    "gap_stop": "נפתחה מתחת לסטופ", "time": "נגמר זמן ההחזקה", "open": "עדיין פתוחה",
    "trail": "סטופ נגרר",
}


def _fmt_r(x) -> str:
    return "—" if x is None else ltr(f"{x:+.2f}R")


def _card_metrics(card: dict) -> None:
    c = st.columns(4)
    c[0].metric("עסקאות", f"{card.get('trades', 0):,}")
    c[1].metric("אחוז הצלחה", pct((card.get("win_rate") or 0) * 100, 0))
    c[2].metric("תוחלת לעסקה", _fmt_r(card.get("expectancy_r")))
    c[3].metric("Profit factor", card.get("profit_factor") or "—")


def _stats_table(groups: dict, label: str) -> pd.DataFrame:
    rows = []
    for k, v in groups.items():
        rows.append(
            {label: k, "עסקאות": v["trades"], "אחוז הצלחה": pct((v["win_rate"] or 0) * 100, 0),
             "תוחלת": _fmt_r(v["expectancy_r"]), "Profit factor": v["profit_factor"]}
        )
    return pd.DataFrame(rows)


def _equity_chart(book: dict, bench: dict | None) -> go.Figure:
    fig = go.Figure()
    curve = pd.DataFrame(book.get("curve", []), columns=["date", "equity"])
    if not curve.empty:
        curve["date"] = pd.to_datetime(curve["date"])
        fig.add_scatter(x=curve["date"], y=curve["equity"] / curve["equity"].iloc[0] * 100, name="התבנית (תיק)", line=dict(color="#1f6feb"))
    if bench:
        b = pd.DataFrame(bench["curve"], columns=["date", "close"])
        b["date"] = pd.to_datetime(b["date"])
        fig.add_scatter(x=b["date"], y=b["close"] / b["close"].iloc[0] * 100, name="מדד ת\"א 125", line=dict(color="#8c959f", dash="dot"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="100 = התחלה",
                      legend=dict(orientation="h", y=1.1), hovermode="x unified",
                      xaxis=dict(tickformat="%m/%Y", hoverformat="%d.%m.%Y"))
    return fig


def _lab() -> None:
    data = edge_cards()
    if not data:
        st.warning("אין עדיין תוצאות בקטסט. יש להריץ במחשב: python scripts/run_backtests.py")
        return
    st.caption(f"עודכן: {date_he(data['generated_at'])} · {data['universe']}")
    st.warning(f"⚠ {data['survivorship_note']}")
    bench = data.get("benchmark")

    st.markdown("#### כלל המעבר (נקבע מראש)")
    st.markdown(
        "תבנית מופיעה בסריקה רק אם: (1) עם הפרמטרים שנבחרו בתקופת הכיול התוחלת חיובית, "
        "(2) גם בתקופת הבדיקה התוחלת חיובית, (3) התוחלת בשתי התקופות יחד גדולה מפעמיים טעות התקן."
    )
    status = pd.DataFrame(
        [{"תבנית": v["name_he"], "עברה": "✅" if v.get("passed") else "❌",
          "תוחלת כוללת": _fmt_r(v.get("walk_forward", {}).get("combined", {}).get("expectancy_r")),
          "סיבה": "; ".join(v.get("fail_reasons", [])) or "—"} for v in data["setups"].values()]
    )
    st.dataframe(status, hide_index=True)
    all_book = data.get("portfolio_all_active")
    if all_book:
        c = st.columns(3)
        c[0].metric("כל התבניות שעברו יחד: תשואה שנתית", pct(all_book.get("cagr_pct")))
        c[1].metric("ירידה מקסימלית", pct(all_book.get("max_drawdown_pct")))
        c[2].metric("חלק מההון שהיה מושקע", pct(all_book.get("avg_exposure_pct"), 0))
        st.plotly_chart(_equity_chart(all_book, bench), width="stretch", config={"displayModeBar": False})

    key = st.selectbox(
        "תבנית", list(data["setups"]),
        format_func=lambda k: ("✅ " if data["setups"][k].get("passed") else "❌ ") + data["setups"][k]["name_he"],
    )
    s = data["setups"][key]
    card, book = s["full_period"], s["portfolio"]
    st.subheader(s["name_he"])
    st.write(s["description_he"])
    if not s.get("passed"):
        st.error("התבנית לא עברה את הבדיקה, ולכן היא לא מופיעה בסריקה. " + "; ".join(s.get("fail_reasons", [])))
    st.write(f"**{out_of_ten(card.get('win_rate'))}.**")

    st.markdown(f"#### כל התקופה ({date_he(card.get('period_start'))} עד {date_he(card.get('period_end'))})")
    _card_metrics(card)
    st.caption(
        f"רווח ממוצע בעסקה מוצלחת {_fmt_r(card.get('avg_win_r'))}, הפסד ממוצע בעסקה כושלת {_fmt_r(card.get('avg_loss_r'))}, "
        f"החזקה ממוצעת {card.get('avg_hold_days')} ימים. R = הסכום שהיה בסיכון בעסקה."
    )

    st.markdown("#### כתיק השקעות")
    st.caption(
        f"לכל היותר {book.get('max_positions')} עסקאות פתוחות במקביל, סיכון של {book.get('risk_pct')}% מההון בכל עסקה. "
        f"נלקחו {book.get('trades_taken')} עסקאות, {book.get('trades_skipped')} איתותים נדחו כי לא היה מקום פנוי."
    )
    c = st.columns(2)
    c[0].metric("תשואה שנתית ממוצעת", pct(book.get("cagr_pct")), help="התבנית, כתיק")
    c[1].metric("ירידה מקסימלית", pct(book.get("max_drawdown_pct")))
    if book.get("avg_exposure_pct") is not None:
        st.caption(
            f"בממוצע רק {pct(book['avg_exposure_pct'], 0)} מההון היה מושקע בפועל. "
            "השאר חיכה במזומן, ולכן ההשוואה למדד (שמושקע ב-100%) לא שוויונית."
        )
    if bench:
        c = st.columns(2)
        c[0].metric("מדד ת\"א 125: תשואה שנתית", pct(bench["cagr_pct"]))
        c[1].metric("מדד ת\"א 125: ירידה מקסימלית", pct(bench["max_drawdown_pct"]))
    st.plotly_chart(_equity_chart(book, bench), width="stretch", config={"displayModeBar": False})

    alt_key = "without_regime_filter" if "without_regime_filter" in s else "with_regime_filter"
    if alt_key in s:
        alt = s[alt_key]
        label = "בלי פילטר משטר שוק" if alt_key == "without_regime_filter" else "עם פילטר משטר שוק"
        st.markdown(f"#### השוואה: {label}")
        _card_metrics(alt["full_period"])
        st.caption(f"כתיק: תשואה שנתית {pct(alt['portfolio'].get('cagr_pct'))}, ירידה מקסימלית {pct(alt['portfolio'].get('max_drawdown_pct'))}")

    wf = s.get("walk_forward", {})
    if "test" in wf:
        st.markdown("#### בדיקה על נתונים שלא שימשו לכיול (walk-forward)")
        st.caption(f"הפרמטרים שנבחרו בתקופת הכיול: {wf['best_params']}")
        left, mid, right = st.columns(3)
        for col, label, part, period in (
            (left, "תקופת הכיול", wf["train"], wf["train_period"]),
            (mid, "תקופת הבדיקה", wf["test"], wf["test_period"]),
            (right, "יחד", wf.get("combined", {}), [wf["train_period"][0], wf["test_period"][1]]),
        ):
            col.write(f"**{label}**")
            col.metric("תוחלת לעסקה", _fmt_r(part.get("expectancy_r")))
            se = part.get("expectancy_se")
            col.caption(f"{part.get('trades', 0)} עסקאות" + (f", ±{se:.2f}" if se else ""))

    cal = s.get("calibration") or {}
    if cal.get("buckets"):
        st.markdown("#### האם הציון מנבא? (כיול)")
        st.caption(cal.get("note", ""))
        cal_df = pd.DataFrame(cal["buckets"]).rename(
            columns={"bucket": "עשירון", "trades": "עסקאות", "score_from": "ציון מ-", "score_to": "עד",
                     "avg_r": "תוצאה ממוצעת (R)", "win_rate": "אחוז הצלחה"}
        )
        st.dataframe(cal_df, hide_index=True)
        st.caption(f"מתאם דירוג (Spearman) בין הציון לתוצאה: {cal.get('spearman')} (טעות תקן {cal.get('spearman_se')})")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### לפי שנה")
        st.dataframe(_stats_table(card.get("by_year", {}), "שנה"), hide_index=True)
    with t2:
        st.markdown("#### לפי משטר שוק")
        st.dataframe(_stats_table(card.get("by_regime", {}), "משטר"), hide_index=True)

    st.markdown("#### סיבות יציאה")
    reasons = card.get("exit_reasons", {})
    st.write(" · ".join(f"{REASONS_HE.get(k, k)}: {v}" for k, v in reasons.items()))

    tr = trades(key)
    if not tr.empty:
        with st.expander(f"כל העסקאות ({len(tr):,})"):
            show = tr[["symbol", "signal_date", "entry_date", "exit_date", "entry", "stop", "target", "exit", "exit_reason", "r"]].copy()
            show["exit_reason"] = show["exit_reason"].map(REASONS_HE)
            show["r"] = show["r"].round(2)
            st.dataframe(show.sort_values("signal_date", ascending=False), hide_index=True)

    _experiments(key)
    _rerun(key, s)


EXPERIMENT_NAMES = {
    "baseline": "התבנית כמו שהיא",
    "above_sma200": "רק מעל ממוצע 200 ימים",
    "rs_positive": "רק מניות חזקות מהמדד",
    "contraction": "רק אחרי התכווצות תנודתיות",
    "strong_close": "רק סגירה ברבע העליון של היום",
    "stop_entry": "כניסה רק מעל הגבוה של יום הפריצה",
    "tighter_stop": "סטופ צמוד יותר (1 ATR)",
    "breakeven": "סטופ לנקודת הכניסה אחרי 1R",
    "trail": "סטופ נגרר בלי יעד",
    "rank_rs": "בחירה לפי חוזק יחסי",
}


def _experiments(key: str) -> None:
    path = RESULTS_DIR / f"experiments_{key.split('_')[0]}.json"
    if not path.exists():
        return
    log = json.loads(path.read_text(encoding="utf-8"))
    ideas = log.get("train_ideas")
    if not ideas:
        return
    with st.expander("ניסויי שיפור (נבדקו רק על תקופת הכיול)"):
        st.caption(
            f"כל רעיון נבדק לבד על {date_he(ideas['period'][0])} עד {date_he(ideas['period'][1])}. "
            "רעיון נחשב שיפור רק אם הוא משפר את התוחלת ביותר מפעמיים טעות התקן."
        )
        rows = []
        for name, r in ideas["results"].items():
            rows.append(
                {"רעיון": EXPERIMENT_NAMES.get(name, name), "עסקאות": r["trades"],
                 "אחוז הצלחה": pct((r["win_rate"] or 0) * 100, 0), "תוחלת": _fmt_r(r["expectancy_r"]),
                 "±": f"{r['expectancy_se']:.2f}", "תשואה שנתית": pct(r["cagr_pct"]),
                 "ירידה מקסימלית": pct(r["max_drawdown_pct"])}
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        base = ideas["results"]["baseline"]
        winners = [
            n for n, r in ideas["results"].items()
            if n != "baseline" and (r["expectancy_r"] or -9) - base["expectancy_r"] > 2 * base["expectancy_se"]
        ]
        if winners:
            st.success("רעיונות שעברו: " + ", ".join(EXPERIMENT_NAMES.get(n, n) for n in winners))
        else:
            st.warning("אף רעיון לא שיפר בצורה מובהקת. לכן לא נבדק שילוב על תקופת הבדיקה, והתבנית נשארה כמו שהיא.")


PARAM_LABELS = {
    "volume_mult": "מחזור ביום הפריצה (פי כמה מהממוצע)",
    "target_r": "יעד (פי כמה מהסיכון)",
    "max_hold_days": "ימי החזקה מקסימליים",
    "rsi_max": "RSI(2) מתחת ל-",
    "stop_atr": "סטופ (כמה ATR מתחת לסגירה)",
}
PARAM_CHOICES = {
    "volume_mult": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
    "target_r": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "max_hold_days": [3, 5, 7, 10, 15, 20],
    "rsi_max": [3, 5, 10, 15, 20],
    "stop_atr": [1.5, 2.0, 2.5, 3.0, 4.0],
}
REGIME_LABELS = {"up": "רק בשוק עולה", "not_down": "לא בשוק יורד", "any": "תמיד"}


def _rerun(key: str, s: dict) -> None:
    st.markdown("#### הרצה עם פרמטרים אחרים")
    params = s["params"]
    grid = load_config()["backtest"]["walk_forward"]["grids"].get(key, {})
    chosen = {}
    cols = st.columns(2)
    for i, name in enumerate(grid):
        choices = sorted(set(PARAM_CHOICES.get(name, grid[name])) | {params[name]})
        chosen[name] = cols[i % 2].select_slider(PARAM_LABELS.get(name, name), choices, value=params[name], key=f"rr_{key}_{name}")
    modes = list(REGIME_LABELS)
    chosen["regime"] = st.radio(
        "מתי לוקחים איתותים", modes, index=modes.index(params.get("regime", "any")),
        format_func=REGIME_LABELS.get, horizontal=True, key=f"rr_{key}_regime",
    )
    if not st.button("הרץ בקטסט", key=f"rr_{key}_go"):
        return
    u = load_universe()
    symbols = u.loc[u["in_ta125"] == 1, "symbol"].tolist()
    frames = cached_frames(symbols)
    idx_sym = load_config()["data"]["index_symbol"]
    index_df = cached_frames([idx_sym]).get(idx_sym)
    if len(frames) < 10 or index_df is None:
        st.warning("אין נתונים שמורים בשרת הזה. הרצה חדשה זמינה רק במחשב שבו רץ scripts/run_backtests.py.")
        return
    from core.backtest import backtest_window

    start, end = backtest_window(frames, load_config()["backtest"]["years"])
    with st.spinner("מריצה בקטסט..."):
        setup = SETUPS[key](**chosen)
        events = pd.DatetimeIndex(edge_cards().get("market_event_days", []))
        tr = run_setup(setup, frames, index_df, start, end, event_days=events)
        card = edge_card(tr)
        cfg = load_config()
        book = portfolio(tr, cfg["risk"]["max_open_positions"], cfg["account"]["risk_per_trade_pct"], cfg["account"]["capital_ils"])
    _card_metrics(card)
    st.caption(f"כתיק: תשואה שנתית {pct(book.get('cagr_pct'))}, ירידה מקסימלית {pct(book.get('max_drawdown_pct'))}")
    st.caption("⚠ שינוי פרמטרים עד שהתוצאה נראית טוב הוא התאמת יתר. תוצאה כזו כמעט תמיד חלשה יותר בעתיד.")


def _quality() -> None:
    data = edge_cards()
    st.markdown("#### מגבלות ידועות של הנתונים")
    st.markdown(
        "- **ימי ראשון חסרים:** ב-Yahoo אין נתונים לימי ראשון לפני 2026, כשהבורסה עוד נסחרה בהם. "
        "תנועה של יום ראשון מופיעה כפער בפתיחה של יום שני.\n"
        "- **בלבול אגורות ושקלים:** אצל חלק מהמניות ההיסטוריה ב-Yahoo קטנה פי 100. המערכת מזהה ומתקנת את זה אוטומטית.\n"
        "- **הטיית שורדים:** הבקטסט רץ על ההרכב הנוכחי של ת\"א 125.\n"
        "- **השהיה:** מחירי ת\"א ב-Yahoo בדרך כלל מושהים."
    )
    if data and data.get("data_quality_issues"):
        st.markdown("#### מניות עם קפיצה חריגה (מעל 25% ביום)")
        st.caption("אחרי התיקון האוטומטי. ברוב המקרים אלה תנועות אמיתיות, אבל כדאי לבדוק.")
        st.dataframe(pd.DataFrame(data["data_quality_issues"]), hide_index=True)


def _settings() -> None:
    cfg = load_config()
    st.caption("ההגדרות נקראות מהקובץ config.yaml. עריכה מתוך המסך תתווסף בשלב 2.")
    a, c = cfg["account"], cfg["costs"]
    st.write(f"**הון:** {ils(a['capital_ils'], 0)}")
    st.write(f"**סיכון לעסקה:** {a['risk_per_trade_pct']}%")
    st.write(f"**עמלה:** {c['commission_pct']}% (מינימום {ils(c['commission_min_ils'], 0)})")
    st.write(f"**החלקה:** {c['slippage_pct']}% מכל צד")
    st.write(f"**מחזור מינימלי לסריקה:** {ils(cfg['universe']['min_avg_turnover_ils'], 0)} ביום")


def render() -> None:
    st.title("מתקדם")
    lab, quality, settings = st.tabs(["מעבדת בקטסט", "איכות נתונים", "הגדרות"])
    with lab:
        _lab()
    with quality:
        _quality()
    with settings:
        _settings()
