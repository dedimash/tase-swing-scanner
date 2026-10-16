"""בקטסט לתבניות יומיות: סימולציית עסקאות, כרטיס יתרון ו-walk-forward.

כללי הסימולציה:
  - כניסה בפתיחה של היום שאחרי האיתות. אם הפתיחה מתחת לסטופ, אין כניסה.
  - יעד = כניסה + target_r * (כניסה - סטופ).
  - בכל יום בודקים קודם סטופ ואחר כך יעד (הנחה שמרנית כששניהם באותו נר).
  - גאף מעבר לסטופ או ליעד מתבצע במחיר הפתיחה.
  - אם לא נגענו בסטופ או ביעד, יוצאים בסגירה של יום ההחזקה האחרון.
  - עסקה אחת בכל רגע לכל מניה.
  - החלקה ועמלה מכל צד. R = רווח נקי / הסכום שהיה בסיכון.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import load_config
from core.regime import MIXED, UP, market_regime, regime_on
from core.setups_daily import Setup


@dataclass
class Costs:
    commission_pct: float = 0.0
    commission_min_ils: float = 0.0
    slippage_pct: float = 0.0

    @classmethod
    def from_config(cls) -> "Costs":
        c = load_config()["costs"]
        return cls(c["commission_pct"], c["commission_min_ils"], c["slippage_pct"])

    def commission(self, value: float) -> float:
        if self.commission_pct == 0 and self.commission_min_ils == 0:
            return 0.0
        return max(self.commission_min_ils, value * self.commission_pct / 100.0)


@dataclass
class Sizing:
    capital_ils: float = 100_000.0
    risk_pct: float = 0.75
    max_position_pct: float = 100.0

    @classmethod
    def from_config(cls) -> "Sizing":
        a = load_config()["account"]
        return cls(a["capital_ils"], a["risk_per_trade_pct"], a.get("max_position_pct", 100.0))

    def shares(self, entry: float, risk_per_share: float) -> int:
        """כמות מניות לפי סיכון קבוע, בלי לעבור את תקרת הפוזיציה (בלי מינוף)."""
        by_risk = self.capital_ils * self.risk_pct / 100.0 / risk_per_share
        by_cap = self.capital_ils * self.max_position_pct / 100.0 / entry
        return math.floor(min(by_risk, by_cap))


def _num(sig: pd.Series, key: str) -> float:
    value = sig.get(key, np.nan)
    return np.nan if value is None else float(value)


def simulate(symbol: str, df: pd.DataFrame, signals: pd.DataFrame, costs: Costs, sizing: Sizing) -> list[dict]:
    """מריץ את האיתותים של מניה אחת ומחזיר רשימת עסקאות.

    עמודות אופציונליות באיתות:
      entry_trigger - נכנסים רק אם המחיר מחר עובר את הרמה הזו (במחיר הגבוה מבין הפתיחה והרמה)
      target_r      - NaN = בלי יעד
      breakeven_r   - אחרי סגירה ברווח של X*R הסטופ עובר למחיר הכניסה
      trail_atr     - סטופ נגרר: X*ATR מתחת לסגירה הגבוהה מאז הכניסה
      exit_ma       - יציאה בסגירה הראשונה מעל ממוצע נע של X ימים
    """
    if signals.empty:
        return []
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    moving_averages: dict[int, np.ndarray] = {}
    dates = df.index
    n = len(df)
    slip = costs.slippage_pct / 100.0
    trades = []
    next_free = 0
    positions = dates.get_indexer(signals.index)
    for (sig_date, sig), i in zip(signals.iterrows(), positions):
        if i < 0 or i < next_free:
            continue
        e = i + 1
        if e >= n:
            continue  # איתות ביום האחרון: עדיין אין יום כניסה
        stop = float(sig["stop"])
        trigger = _num(sig, "entry_trigger")
        if np.isnan(trigger):
            entry_raw = o[e]
        elif h[e] >= trigger:
            entry_raw = max(o[e], trigger)
        else:
            continue  # המחיר לא עבר את רמת הכניסה: אין עסקה
        if not entry_raw > stop:
            continue
        risk_raw = entry_raw - stop
        target_r, breakeven_r, trail_atr = _num(sig, "target_r"), _num(sig, "breakeven_r"), _num(sig, "trail_atr")
        target = np.inf if np.isnan(target_r) else entry_raw + target_r * risk_raw
        trail_dist = np.nan if np.isnan(trail_atr) else trail_atr * float(sig["atr"])
        exit_ma = _num(sig, "exit_ma")
        ma = None
        if not np.isnan(exit_ma):
            period = int(exit_ma)
            if period not in moving_averages:
                moving_averages[period] = pd.Series(c).rolling(period, min_periods=period).mean().to_numpy()
            ma = moving_averages[period]
        last = e + int(sig["max_hold"]) - 1

        current_stop = stop
        best_close = entry_raw
        reason, exit_raw, j = "open", c[n - 1], n - 1
        for j in range(e, min(last, n - 1) + 1):
            moved = current_stop > stop
            if j > e and o[j] <= current_stop:
                reason, exit_raw = ("trail" if moved else "gap_stop"), o[j]
                break
            if l[j] <= current_stop:
                reason, exit_raw = ("trail" if moved else "stop"), current_stop
                break
            if j > e and o[j] >= target:
                reason, exit_raw = "gap_target", o[j]
                break
            if h[j] >= target:
                reason, exit_raw = "target", target
                break
            if ma is not None and c[j] > ma[j]:
                reason, exit_raw = "ma_exit", c[j]
                break
            if j == last:
                reason, exit_raw = "time", c[j]
                break
            # עדכון הסטופ לקראת מחר, לפי הסגירה של היום
            best_close = max(best_close, c[j])
            if not np.isnan(breakeven_r) and c[j] >= entry_raw + breakeven_r * risk_raw:
                current_stop = max(current_stop, entry_raw)
            if not np.isnan(trail_dist):
                current_stop = max(current_stop, best_close - trail_dist)

        entry = entry_raw * (1 + slip)
        exit_px = exit_raw * (1 - slip)
        risk_per_share = entry - stop
        shares = sizing.shares(entry, risk_per_share)
        if shares <= 0:
            continue
        fees = costs.commission(shares * entry) + costs.commission(shares * exit_px)
        pnl = shares * (exit_px - entry) - fees
        risk_ils = shares * risk_per_share
        rank = _num(sig, "rank_score")
        trades.append(
            {
                "symbol": symbol,
                "signal_date": sig_date,
                "entry_date": dates[e],
                "exit_date": dates[j],
                "entry": round(entry, 4),
                "stop": round(stop, 4),
                "target": round(target, 4) if np.isfinite(target) else np.nan,
                "exit": round(exit_px, 4),
                "exit_reason": reason,
                "hold_days": j - e + 1,
                "shares": shares,
                "pnl_ils": round(pnl, 2),
                "r": pnl / risk_ils,
                "risk_frac": risk_ils / sizing.capital_ils,
                "position_frac": shares * entry / sizing.capital_ils,
                "rank_score": float(sig.get("volume_ratio", 0.0)) if np.isnan(rank) else rank,
            }
        )
        next_free = j
    return trades


def allowed_regimes(mode: str | bool | None) -> set[str] | None:
    """up = רק שוק עולה, not_down = עולה או מעורב, any = תמיד (None)."""
    if mode is True:  # תאימות לאחור: regime_filter=True
        mode = "up"
    return {"up": {UP}, "not_down": {UP, MIXED}}.get(mode or "any")


def run_setup(
    setup: Setup,
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    costs: Costs | None = None,
    sizing: Sizing | None = None,
    event_days: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """מריץ תבנית על כל המניות. מסנן איתותים לפי תאריך האיתות בטווח [start, end).

    הפרמטר regime של התבנית קובע באילו משטרי שוק לוקחים איתותים (ראו allowed_regimes).
    event_days: ימים עם מחזור חריג בכל השוק. בתבניות שמסתמכות על מחזור, איתותים מהימים האלה לא נספרים.
    """
    costs = costs or Costs.from_config()
    sizing = sizing or Sizing.from_config()
    regime = market_regime(index_df) if index_df is not None and not index_df.empty else None
    allowed = allowed_regimes(setup.params.get("regime", "any"))
    skip_days = event_days if (setup.uses_volume_confirmation and event_days is not None) else None
    rows = []
    index_close = index_df["close"] if index_df is not None and not index_df.empty else None
    for sym, df in frames.items():
        if df is None or len(df) < 60:
            continue
        sig = setup.detect(df, index_close)
        if start is not None:
            sig = sig[sig.index >= start]
        if end is not None:
            sig = sig[sig.index < end]
        if allowed is not None and regime is not None and not sig.empty:
            sig = sig[regime_on(sig.index, regime).isin(allowed).to_numpy()]
        if skip_days is not None and not sig.empty:
            sig = sig[~sig.index.isin(skip_days)]
        rows += simulate(sym, df, sig, costs, sizing)
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades
    trades["setup"] = setup.key
    if regime is not None:
        trades["regime"] = regime_on(pd.DatetimeIndex(trades["signal_date"]), regime).to_numpy()
    return trades.sort_values("exit_date").reset_index(drop=True)


def portfolio(trades: pd.DataFrame, max_positions: int, risk_pct: float, capital: float = 100_000.0) -> dict:
    """סימולציית תיק: לכל היותר max_positions עסקאות פתוחות, סיכון קבוע מההון לכל עסקה.

    כשיש יותר איתותים ממקומות פנויים באותו יום, נבחרים האיתותים עם rank_score הגבוה ביותר.
    אם בעסקאות יש risk_frac (הסיכון בפועל אחרי תקרת הפוזיציה), משתמשים בו במקום risk_pct.
    """
    if trades is None or trades.empty:
        return {"trades_taken": 0}
    closed = trades[trades["exit_reason"] != "open"]
    order = closed.sort_values(["entry_date", "rank_score", "symbol"], ascending=[True, False, True])
    open_exits: list[pd.Timestamp] = []
    taken = []
    for _, t in order.iterrows():
        open_exits = [x for x in open_exits if x >= t["entry_date"]]
        if len(open_exits) < max_positions:
            open_exits.append(t["exit_date"])
            taken.append(t)
    if not taken:
        return {"trades_taken": 0}
    book = pd.DataFrame(taken).sort_values("exit_date")
    risk_frac = book["risk_frac"].to_numpy() if "risk_frac" in book else np.full(len(book), risk_pct / 100.0)
    growth = 1.0 + book["r"].to_numpy() * risk_frac
    equity = pd.Series(capital * np.cumprod(growth), index=pd.DatetimeIndex(book["exit_date"]))
    peak = np.maximum.accumulate(np.concatenate([[capital], equity.to_numpy()]))[1:]
    drawdown = equity.to_numpy() / peak - 1.0
    first_entry = pd.Timestamp(book["entry_date"].min())
    years = max((equity.index.max() - first_entry).days / 365.25, 1e-9)
    final = float(equity.iloc[-1])
    curve = equity.groupby(equity.index).last()

    exposure = None
    if "position_frac" in book:
        days = pd.bdate_range(first_entry, equity.index.max())
        invested = pd.Series(0.0, index=days)
        for _, t in book.iterrows():
            invested.loc[t["entry_date"] : t["exit_date"]] += t["position_frac"]
        exposure = round(float(invested.mean()) * 100, 1)

    streak = longest = 0
    for r in book["r"].to_numpy():
        streak = streak + 1 if r <= 0 else 0
        longest = max(longest, streak)

    return {
        "max_positions": max_positions,
        "risk_pct": risk_pct,
        "max_losing_streak": int(longest),
        "trades_taken": int(len(book)),
        "trades_skipped": int(len(closed) - len(book)),
        "final_multiple": round(final / capital, 3),
        "cagr_pct": round(((final / capital) ** (1 / years) - 1) * 100, 2),
        "max_drawdown_pct": round(float(drawdown.min()) * 100, 2),
        "avg_exposure_pct": exposure,
        "curve": [[str(d.date()), round(float(v), 0)] for d, v in curve.items()],
    }


def _stats(r: pd.Series) -> dict:
    wins, losses = r[r > 0], r[r <= 0]
    gross_loss = -losses.sum()
    return {
        "trades": int(len(r)),
        "win_rate": round(float((r > 0).mean()), 4) if len(r) else None,
        "expectancy_r": round(float(r.mean()), 4) if len(r) else None,
        # טעות התקן של התוחלת: הבדל קטן מפעמיים הערך הזה הוא כנראה מזל
        "expectancy_se": round(float(r.std(ddof=1) / np.sqrt(len(r))), 4) if len(r) > 1 else None,
        "avg_win_r": round(float(wins.mean()), 4) if len(wins) else None,
        "avg_loss_r": round(float(losses.mean()), 4) if len(losses) else None,
        "profit_factor": round(float(wins.sum() / gross_loss), 3) if gross_loss > 0 else None,
    }


def edge_card(trades: pd.DataFrame) -> dict:
    """סיכום סטטיסטי של התבנית. עסקאות שעדיין פתוחות לא נספרות."""
    if trades is None or trades.empty:
        return {"trades": 0}
    closed = trades[trades["exit_reason"] != "open"].sort_values("exit_date")
    if closed.empty:
        return {"trades": 0}
    r = closed["r"]
    cum = r.cumsum()
    drawdown = cum - cum.cummax().clip(lower=0)
    first = pd.Timestamp(closed["signal_date"].min())
    last = pd.Timestamp(closed["exit_date"].max())
    years = max((last - first).days / 365.25, 1e-9)
    card = _stats(r)
    card.update(
        {
            "total_r": round(float(r.sum()), 2),
            "max_drawdown_r": round(float(drawdown.min()), 2),
            "trades_per_year": round(len(r) / years, 1),
            "avg_hold_days": round(float(closed["hold_days"].mean()), 1),
            "exit_reasons": {k: int(v) for k, v in closed["exit_reason"].value_counts().items()},
            "period_start": str(first.date()),
            "period_end": str(last.date()),
            "by_year": {
                str(y): _stats(g["r"]) for y, g in closed.groupby(pd.DatetimeIndex(closed["signal_date"]).year)
            },
        }
    )
    if "regime" in closed.columns:
        card["by_regime"] = {str(k): _stats(g["r"]) for k, g in closed.groupby("regime", dropna=True)}
    return card


def backtest_window(frames: dict[str, pd.DataFrame], years: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = max(df.index.max() for df in frames.values() if df is not None and not df.empty)
    return end - pd.DateOffset(years=years), end + pd.Timedelta(days=1)


def passes(train: dict, test: dict, combined: dict) -> tuple[bool, list[str]]:
    """כלל המעבר, שנקבע מראש:
    1. תוחלת חיובית בתקופת הכיול (עם הפרמטרים שנבחרו).
    2. תוחלת חיובית בתקופת הבדיקה.
    3. התוחלת בשתי התקופות יחד גדולה מפעמיים טעות התקן.
    """
    reasons = []
    if not (train.get("expectancy_r") or 0) > 0:
        reasons.append("תוחלת לא חיובית בתקופת הכיול")
    if not (test.get("expectancy_r") or 0) > 0:
        reasons.append("תוחלת לא חיובית בתקופת הבדיקה")
    exp, se = combined.get("expectancy_r"), combined.get("expectancy_se")
    if exp is None or se is None or not exp > 2 * se:
        reasons.append("התוחלת הכוללת לא גדולה מספיק ביחס לטעות הסטטיסטית")
    return not reasons, reasons


def walk_forward(
    setup_cls: type[Setup],
    frames: dict[str, pd.DataFrame],
    index_df: pd.DataFrame | None = None,
    costs: Costs | None = None,
    sizing: Sizing | None = None,
    event_days: pd.DatetimeIndex | None = None,
) -> dict:
    """מכייל פרמטרים על תקופת הכיול ובודק אותם, בלי שינוי, על התקופה שאחריה. מחזיר גם אם התבנית עברה."""
    cfg = load_config()["backtest"]
    wf = cfg["walk_forward"]
    start, end = backtest_window(frames, cfg["years"])
    split = start + pd.DateOffset(years=wf["train_years"])
    grid = wf["grids"][setup_cls.key]
    keys = list(grid)

    def run(params, a, b):
        return run_setup(setup_cls(**params), frames, index_df, a, b, costs, sizing, event_days)

    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        card = edge_card(run(params, start, split))
        results.append({"params": params, "trades": card["trades"], "expectancy_r": card.get("expectancy_r")})

    eligible = [x for x in results if x["trades"] >= wf["min_train_trades"] and x["expectancy_r"] is not None]
    if not eligible:
        return {"error": "not enough trades in the training period", "grid": results, "passed": False,
                "fail_reasons": ["אין מספיק עסקאות בתקופת הכיול"]}
    best = max(eligible, key=lambda x: x["expectancy_r"])
    train_trades, test_trades = run(best["params"], start, split), run(best["params"], split, end)
    train, test = edge_card(train_trades), edge_card(test_trades)
    combined = edge_card(pd.concat([train_trades, test_trades], ignore_index=True))
    ok, reasons = passes(train, test, combined)
    return {
        "train_period": [str(start.date()), str(split.date())],
        "test_period": [str(split.date()), str((end - pd.Timedelta(days=1)).date())],
        "best_params": best["params"],
        "train": train,
        "test": test,
        "combined": combined,
        "passed": ok,
        "fail_reasons": reasons,
        "grid": results,
    }
