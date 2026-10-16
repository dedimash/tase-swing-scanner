import numpy as np
import pandas as pd

from core.setups_daily import Breakout20


def synthetic(n=200, seed=1):
    rng = np.random.default_rng(seed)
    close = 50 + rng.normal(0.05, 1.0, n).cumsum()
    close = np.maximum(close, 5)
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + rng.uniform(0, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1, n)
    volume = rng.uniform(50_000, 150_000, n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2025-01-01", periods=n, freq="B"),
    )
    df["turnover"] = df["close"] * df["volume"]
    return df


def breakout_frame():
    """60 ימים שקטים ואז יום פריצה ברור עם מחזור כפול."""
    n = 80
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.linspace(40, 50, n)
    df = pd.DataFrame(
        {"open": close - 0.1, "high": close + 0.2, "low": close - 0.3, "close": close, "volume": 100_000.0},
        index=idx,
    )
    last = idx[-1]
    df.loc[last, ["open", "high", "low", "close", "volume"]] = [50.0, 53.0, 49.9, 52.8, 250_000.0]
    df["turnover"] = df["close"] * df["volume"]
    return df


def test_breakout_detected_on_clear_breakout_day():
    df = breakout_frame()
    sig = Breakout20().detect(df)
    assert df.index[-1] in sig.index
    row = sig.loc[df.index[-1]]
    assert row["stop"] < row["close"]
    assert row["close"] - row["stop"] <= 2.0 * row["atr"] + 1e-9


def test_no_signal_without_volume():
    df = breakout_frame()
    df.loc[df.index[-1], "volume"] = 100_000.0
    assert df.index[-1] not in Breakout20().detect(df).index


def test_no_signal_when_illiquid():
    df = breakout_frame()
    df["volume"] = df["volume"] / 1000
    df["turnover"] = df["close"] * df["volume"]
    assert Breakout20().detect(df).empty


def test_no_look_ahead():
    df = synthetic(300)
    full = Breakout20(volume_mult=1.0).detect(df)
    cut_date = df.index[200]
    partial = Breakout20(volume_mult=1.0).detect(df.loc[:cut_date])
    pd.testing.assert_frame_equal(full.loc[:cut_date], partial)


def test_no_look_ahead_with_all_filters():
    df = synthetic(320, seed=3)
    index_close = synthetic(320, seed=4)["close"]
    opts = dict(
        volume_mult=1.0, above_sma200=True, min_rs=-1.0, max_atr_contraction=5.0,
        min_close_location=0.0, entry_mode="stop_above_high", rank_by="rs",
    )
    full = Breakout20(**opts).detect(df, index_close)
    assert len(full) >= 2
    cut = full.index[1]  # חותכים בדיוק ביום של איתות: הוא צריך להופיע גם בלי נתונים מהעתיד
    partial = Breakout20(**opts).detect(df.loc[:cut], index_close.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut], partial, check_freq=False)


def test_unknown_parameter_rejected():
    try:
        Breakout20(not_a_param=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
