import numpy as np
import pandas as pd
import pytest

from core.indicators import atr, avg_turnover, ema, relative_strength, rsi, sma, true_range, vwap


def series(values):
    return pd.Series(values, index=pd.date_range("2026-01-05", periods=len(values), freq="B"), dtype=float)


def test_sma():
    s = sma(series([1, 2, 3, 4, 5]), 3)
    assert np.isnan(s.iloc[1])
    assert s.iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_ema_seeded_with_sma():
    # period=3, alpha=0.5. seed=(1+2+3)/3=2, next=0.5*4+0.5*2=3, next=0.5*5+0.5*3=4
    e = ema(series([1, 2, 3, 4, 5]), 3)
    assert np.isnan(e.iloc[1])
    assert e.iloc[2:].tolist() == pytest.approx([2.0, 3.0, 4.0])


def test_rsi_hand_computed():
    # שינויים: +1, -1, +1, +1. period=2.
    # ממוצע ראשון: רווח 0.5, הפסד 0.5 -> 50
    # אחר כך: רווח (0.5+1)/2=0.75, הפסד 0.25 -> RS=3 -> 75
    # אחר כך: רווח 0.875, הפסד 0.125 -> RS=7 -> 87.5
    r = rsi(series([1, 2, 1, 2, 3]), 2)
    assert np.isnan(r.iloc[1])
    assert r.iloc[2:].tolist() == pytest.approx([50.0, 75.0, 87.5])


def test_rsi_only_gains_is_100():
    assert rsi(series(range(1, 30)), 14).iloc[-1] == 100.0


def ohlc(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.date_range("2026-01-05", periods=len(rows), freq="B"))


def test_true_range_uses_previous_close():
    df = ohlc([[10, 11, 9, 10], [12, 13, 12, 12.5], [11, 11.5, 8, 9]])
    # יום 2: max(13-12, |13-10|, |12-10|) = 3 ; יום 3: max(3.5, |11.5-12.5|, |8-12.5|) = 4.5
    assert true_range(df).tolist() == pytest.approx([2.0, 3.0, 4.5])


def test_atr_wilder():
    df = ohlc([[10, 11, 9, 10], [12, 13, 12, 12.5], [11, 11.5, 8, 9]])
    # period=2: seed=(2+3)/2=2.5, then (2.5*1+4.5)/2=3.5
    a = atr(df, 2)
    assert np.isnan(a.iloc[0])
    assert a.iloc[1:].tolist() == pytest.approx([2.5, 3.5])


def test_vwap_resets_each_day():
    idx = pd.DatetimeIndex(
        ["2026-01-05 10:00", "2026-01-05 11:00", "2026-01-06 10:00"], tz="Asia/Jerusalem"
    )
    df = pd.DataFrame(
        {"high": [10, 12, 20], "low": [10, 12, 20], "close": [10, 12, 20], "volume": [100, 300, 50]}, index=idx
    )
    v = vwap(df)
    assert v.tolist() == pytest.approx([10.0, (10 * 100 + 12 * 300) / 400, 20.0])


def test_avg_turnover():
    df = pd.DataFrame({"turnover": [1.0, 2.0, 3.0]})
    assert avg_turnover(df, 2).tolist()[1:] == [1.5, 2.5]


def test_relative_strength():
    close = series([100, 110, 121])
    index = series([100, 100, 110])
    rs = relative_strength(close, index, 2)
    # מניה +21%, מדד +10% -> 1.21/1.10 - 1 = 0.1
    assert rs.iloc[-1] == pytest.approx(0.1)


def test_indicators_are_causal():
    rng = np.random.default_rng(0)
    close = series(100 + rng.normal(0, 1, 120).cumsum())
    full = rsi(close, 14)
    cut = rsi(close.iloc[:80], 14)
    assert full.iloc[:80].tolist() == pytest.approx(cut.tolist(), nan_ok=True)
