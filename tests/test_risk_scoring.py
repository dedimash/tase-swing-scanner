from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.events import StockEvents, parse_quote
from core.risk import RED, YELLOW, event_check, hold_window, plan_position, sector_warnings
from core.scoring import calibration, score_signals


def test_position_by_risk():
    # הון 100,000, סיכון 1% = 1,000. סיכון למניה 5 -> 200 מניות, 20,000 ₪ (מתחת לתקרה)
    plan = plan_position(100.0, 95.0, capital=100_000, risk_pct=1.0)
    assert plan.shares == 200
    assert plan.risk_ils == pytest.approx(1000.0)
    assert plan.limited_by == []


def test_position_capped_at_25_percent():
    plan = plan_position(100.0, 99.5, capital=100_000, risk_pct=1.0)
    assert plan.amount_ils <= 25_000
    assert plan.limited_by


def test_position_limited_by_liquidity():
    plan = plan_position(100.0, 95.0, avg_volume_shares=1000, capital=100_000, risk_pct=1.0)
    assert plan.shares == 15  # 1.5% מ-1000
    assert "מחזור" in plan.limited_by[0]


def test_invalid_stop():
    assert plan_position(100.0, 101.0).shares == 0


def test_hold_window_skips_weekend():
    start, end = hold_window(date(2026, 10, 16), 3)  # שישי
    assert start == date(2026, 10, 19) and end == date(2026, 10, 21)


def test_known_earnings_in_window_is_red():
    ev = StockEvents(earnings=date(2026, 10, 20), earnings_is_estimate=False)
    out = event_check(ev, date(2026, 10, 19), date(2026, 10, 23))
    assert out[0][0] == RED


def test_estimated_earnings_near_window_is_yellow():
    ev = StockEvents(earnings=date(2026, 10, 30), earnings_is_estimate=True)
    out = event_check(ev, date(2026, 10, 19), date(2026, 10, 23))
    assert out[0][0] == YELLOW


def test_ex_dividend_in_window_is_red():
    out = event_check(StockEvents(ex_dividend=date(2026, 10, 21)), date(2026, 10, 19), date(2026, 10, 23))
    assert out[0][0] == RED


def test_parse_quote_future_and_estimate():
    today = date(2026, 10, 16)
    ts = lambda d: pd.Timestamp(d).timestamp()  # noqa: E731
    known = parse_quote({"earningsTimestampStart": ts("2026-11-17"), "isEarningsDateEstimate": False}, today)
    assert known.earnings == date(2026, 11, 17) and not known.earnings_is_estimate
    guessed = parse_quote({"earningsTimestamp": ts("2026-08-12")}, today)
    assert guessed.earnings_is_estimate and guessed.earnings > today
    div = parse_quote({"corporateActions": [{"meta": {"eventType": "DIVIDEND", "dateEpochMs": ts("2026-10-29") * 1000}}]}, today)
    assert div.ex_dividend == date(2026, 10, 29)


def test_sector_warnings_after_two():
    c = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "sector": ["Banks", "Banks", "Banks", "Tech"]})
    w = sector_warnings(c)
    assert list(w) == ["C"]


def _frame(n=300):
    close = np.linspace(50, 80, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1e5}, index=idx)


def test_score_between_0_and_100_and_tighter_stop_scores_higher():
    df = _frame()
    d = df.index[-1]
    sig = pd.DataFrame({"close": [80.0, 80.0], "stop": [78.0, 72.0], "volume_ratio": [2.0, 2.0]}, index=[d, d])
    s = score_signals(sig, df, None, None, None)
    assert ((s["score"] >= 0) & (s["score"] <= 100)).all()
    assert s["score"].iloc[0] > s["score"].iloc[1]


def test_calibration_detects_positive_relation():
    rng = np.random.default_rng(0)
    score = rng.uniform(0, 100, 500)
    r = score / 50 - 1 + rng.normal(0, 0.5, 500)
    trades = pd.DataFrame({"score": score, "r": r, "exit_reason": "time"})
    cal = calibration(trades)
    assert cal["spearman"] > 0.5
    assert len(cal["buckets"]) == 10
    assert cal["buckets"][-1]["avg_r"] > cal["buckets"][0]["avg_r"]
