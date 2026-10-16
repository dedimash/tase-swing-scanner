import numpy as np
import pandas as pd
import pytest

from core.backtest import Costs, Sizing, allowed_regimes, simulate
from core.regime import DOWN, MIXED, UP, breadth, market_event_days
from core.setups_daily import SETUPS, PullbackEMA20, RSI2Reversion

NO_COSTS = Costs(0.0, 0.0, 0.0)


def frame(close, volume=100_000.0, spread=0.3):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="B")
    df = pd.DataFrame(
        {"open": close - 0.1, "high": close + spread, "low": close - spread, "close": close, "volume": volume},
        index=idx,
    )
    df["turnover"] = df["close"] * df["volume"]
    return df


def uptrend_then_pullback():
    """250 ימים של עלייה, 3 ימי ירידה רגועה עד הממוצע, ויום היפוך."""
    base = list(np.linspace(30, 60, 250))
    dips = [59.0, 58.0, 57.8]
    df = frame(base + dips + [59.5])
    df.loc[df.index[250:253], "volume"] = 50_000.0
    df.loc[df.index[250:253], "low"] = [58.0, 57.0, 56.5]
    last = df.index[-1]
    df.loc[last, ["open", "high", "low", "close"]] = [58.0, 59.8, 57.9, 59.5]
    df["turnover"] = df["close"] * df["volume"]
    return df


def test_pullback_signal_on_reversal_day():
    df = uptrend_then_pullback()
    sig = PullbackEMA20().detect(df)
    assert df.index[-1] in sig.index
    row = sig.loc[df.index[-1]]
    assert row["stop"] < row["close"]


def test_pullback_needs_quiet_volume():
    df = uptrend_then_pullback()
    df.loc[df.index[250:253], "volume"] = 300_000.0
    df["turnover"] = df["close"] * df["volume"]
    assert df.index[-1] not in PullbackEMA20().detect(df).index


def test_rsi2_signal_after_sharp_drop_in_uptrend():
    close = list(np.linspace(30, 60, 250)) + [57.0, 54.0]
    df = frame(close)
    sig = RSI2Reversion().detect(df)
    assert df.index[-1] in sig.index
    assert np.isnan(sig.loc[df.index[-1], "target_r"])
    assert sig.loc[df.index[-1], "exit_ma"] == 5


def test_rsi2_no_signal_below_sma200():
    close = list(np.linspace(60, 30, 250)) + [28.0, 26.0]
    assert RSI2Reversion().detect(frame(close)).empty


def test_ma_exit_in_simulation():
    close = [10, 10, 10, 10, 10, 9, 9.5, 10.5, 11]
    df = frame(close, spread=0.05)
    sig = pd.DataFrame(
        {"stop": [5.0], "max_hold": [5], "atr": [0.5], "target_r": [np.nan], "exit_ma": [5.0]}, index=[df.index[5]]
    )
    t = simulate("X", df, sig, NO_COSTS, Sizing(100_000, 1.0))
    # כניסה ביום 6 (פתיחה 9.4). הסגירה הראשונה מעל ממוצע 5: יום 7 (10.5 > 9.8)
    assert t[0]["exit_reason"] == "ma_exit"
    assert t[0]["exit"] == pytest.approx(10.5)


def test_allowed_regimes():
    assert allowed_regimes("up") == {UP}
    assert allowed_regimes("not_down") == {UP, MIXED}
    assert allowed_regimes("any") is None
    assert allowed_regimes(True) == {UP}
    assert DOWN not in allowed_regimes("not_down")


def test_breadth_share_above_ma():
    up = frame(np.linspace(10, 20, 60))
    down = frame(np.linspace(20, 10, 60))
    b = breadth({"A": up, "B": up, "C": down}, period=10, min_stocks=3)
    assert b.iloc[-1] == pytest.approx(2 / 3)


def test_market_event_day_detected():
    frames = {}
    for k in range(10):
        df = frame(np.full(40, 50.0))
        df.loc[df.index[35], "volume"] = 400_000.0
        frames[f"S{k}"] = df
    days = market_event_days(frames, min_stocks=5)
    assert list(days) == [frames["S0"].index[35]]


def test_all_setups_registered_and_watch_safe_on_short_data():
    short = frame(np.linspace(10, 11, 30))
    for cls in SETUPS.values():
        assert cls().watch(short) is None
