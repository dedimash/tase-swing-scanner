from datetime import date, datetime

import numpy as np
import pandas as pd

from core.events import StockEvents
from core.regime import market_regime
from core.verdict import GREEN, RED, TZ, YELLOW, MarketContext, drop_partial_bar, evaluate, market_is_open

EDGE = {
    "setups": {
        "rsi2_reversion": {
            "passed": True,
            "walk_forward": {"best_params": {}, "combined": {"win_rate": 0.67, "expectancy_r": 0.11}},
        },
        "pullback_ema20": {"passed": False, "walk_forward": {"best_params": {}}},
    }
}


def frame(close, end="2026-10-15", volume=200_000.0):
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range(end=end, periods=len(close))
    df = pd.DataFrame(
        {"open": close + 0.1, "high": close + 0.4, "low": close - 0.4, "close": close, "volume": volume}, index=idx
    )
    df["turnover"] = df["close"] * df["volume"]
    return df


def ctx(index_close=None, events=None):
    index_df = frame(index_close if index_close is not None else np.linspace(1000, 1500, 300))
    return MarketContext(
        index_df=index_df, regime=market_regime(index_df), breadth=pd.Series(dtype=float),
        event_days=pd.DatetimeIndex([]), events=events or {}, edge=EDGE,
    )


AFTER_CLOSE = datetime(2026, 10, 15, 20, 0, tzinfo=TZ)
SHARP_DROP = list(np.linspace(30, 60, 298)) + [57.0, 54.0]


def test_green_on_active_setup_signal():
    v = evaluate("X.TA", frame(SHARP_DROP), ctx(), now=AFTER_CLOSE)
    assert v.color == GREEN
    assert v.setup_key == "rsi2_reversion"
    assert v.shares > 0 and v.stop < v.entry
    assert v.target is None and "ממוצע" in v.exit_rule
    assert "מתוך 10" in v.past_performance


def test_failed_setup_never_green():
    edge_only_failed = {"setups": {"rsi2_reversion": {"passed": False, "walk_forward": {"best_params": {}}}}}
    c = ctx()
    c.edge = edge_only_failed
    assert evaluate("X.TA", frame(SHARP_DROP), c, now=AFTER_CLOSE).color != GREEN


def test_red_when_stock_in_downtrend():
    v = evaluate("X.TA", frame(np.linspace(60, 30, 300)), ctx(), now=AFTER_CLOSE)
    assert v.color == RED and "ירידה" in v.reason


def test_red_when_market_down():
    v = evaluate("X.TA", frame(SHARP_DROP), ctx(np.linspace(1500, 1000, 300)), now=AFTER_CLOSE)
    assert v.color == RED and "השוק" in v.reason


def test_red_when_illiquid():
    v = evaluate("X.TA", frame(SHARP_DROP, volume=100.0), ctx(), now=AFTER_CLOSE)
    assert v.color == RED and "מחזור" in v.reason


def test_red_when_known_earnings_in_window():
    events = {"X.TA": StockEvents(earnings=date(2026, 10, 19), earnings_is_estimate=False)}
    v = evaluate("X.TA", frame(SHARP_DROP), ctx(events=events), now=AFTER_CLOSE)
    assert v.color == RED and "דוח" in v.reason
    assert v.setup_name  # אומרים גם איזו תבנית הייתה


def test_yellow_when_close_to_setup():
    close = list(np.linspace(30, 60, 299)) + [59.7]  # ירידה קטנה: RSI(2) בערך 25, בין 10 ל-30
    v = evaluate("X.TA", frame(close), ctx(), now=AFTER_CLOSE)
    assert v.color == YELLOW and v.waits


def test_not_enough_data():
    assert evaluate("X.TA", frame(np.linspace(10, 11, 50)), ctx(), now=AFTER_CLOSE).color == RED


def test_partial_bar_dropped_during_trading():
    df = frame(SHARP_DROP, end="2026-10-15")
    during = datetime(2026, 10, 15, 12, 0, tzinfo=TZ)
    trimmed, partial = drop_partial_bar(df, during)
    assert partial and trimmed.index[-1] < df.index[-1]
    assert not drop_partial_bar(df, AFTER_CLOSE)[1]


def test_market_hours():
    assert market_is_open(datetime(2026, 10, 16, 12, 0, tzinfo=TZ))      # שישי בצהריים
    assert not market_is_open(datetime(2026, 10, 16, 15, 0, tzinfo=TZ))  # שישי אחרי 13:50
    assert not market_is_open(datetime(2026, 10, 18, 12, 0, tzinfo=TZ))  # ראשון
