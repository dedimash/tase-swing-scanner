import pandas as pd
import pytest

from core.data import normalize_frame, quality_report


def raw(prices, volume=1000):
    idx = pd.date_range("2026-01-05", periods=len(prices), freq="B")
    return pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [volume] * len(prices)},
        index=idx,
    )


def test_agorot_converted_to_shekels():
    df = normalize_frame(raw([7471.0, 7500.0]), "ILA")
    assert df["close"].tolist() == pytest.approx([74.71, 75.0])
    # מחזור בשקלים = מחיר בשקלים * מספר מניות
    assert df["turnover"].iloc[0] == pytest.approx(74.71 * 1000)


def test_shekel_and_index_not_divided():
    df = normalize_frame(raw([4091.37]), "ILS")
    assert df["close"].iloc[0] == pytest.approx(4091.37)


def test_unknown_currency_not_divided():
    assert normalize_frame(raw([10.0]), None)["close"].iloc[0] == 10.0


def test_output_columns():
    assert list(normalize_frame(raw([100.0]), "ILA").columns) == ["open", "high", "low", "close", "volume", "turnover"]


def test_level_shift_of_100x_repaired_to_latest_units():
    # ההיסטוריה הישנה קטנה פי 100 (כמו NXSN לפני 18.5.2026)
    df = normalize_frame(raw([2.87, 2.85, 285.1, 290.0]), "ILA")
    assert df["close"].tolist() == pytest.approx([2.87, 2.85, 2.851, 2.90])


def test_single_spike_of_100x_repaired():
    df = normalize_frame(raw([100.0, 10050.0, 101.0]), "ILS")
    assert df["close"].tolist() == pytest.approx([100.0, 100.5, 101.0])


def test_real_large_move_not_touched():
    df = normalize_frame(raw([100.0, 140.0, 70.0]), "ILS")
    assert df["close"].tolist() == pytest.approx([100.0, 140.0, 70.0])


def test_download_retries_when_yahoo_rate_limits(monkeypatch, tmp_path):
    """הניסיון הראשון חוזר ריק (כמו חסימה של Yahoo), השני מצליח."""
    from core import data as data_mod

    monkeypatch.setattr(data_mod, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(data_mod.time, "sleep", lambda s: None)
    provider = data_mod.YahooProvider({"^TA125.TA": "ILS"})
    calls = []

    def fake_once(symbols, **kwargs):
        calls.append(list(symbols))
        if len(calls) == 1:
            return {}
        return {s: normalize_frame(raw([100.0, 101.0]), "ILS") for s in symbols}

    monkeypatch.setattr(provider, "_download_once", fake_once)
    out = provider._download(["^TA125.TA"], period="1y", interval="1d")
    assert "^TA125.TA" in out and len(calls) == 2


def test_no_retry_for_single_missing_stock_in_long_list(monkeypatch, tmp_path):
    from core import data as data_mod

    monkeypatch.setattr(data_mod, "cache_dir", lambda: tmp_path)
    provider = data_mod.YahooProvider({})
    symbols = [f"S{i}.TA" for i in range(10)]
    calls = []

    def fake_once(syms, **kwargs):
        calls.append(list(syms))
        return {s: normalize_frame(raw([100.0]), "ILS") for s in syms if s != "S3.TA"}

    monkeypatch.setattr(provider, "_download_once", fake_once)
    out = provider._download(symbols)
    assert len(out) == 9 and len(calls) == 1


def test_quality_report_flags_jump_and_missing_day():
    good = normalize_frame(raw([100.0, 101, 102, 103]), "ILS")
    jumpy = normalize_frame(raw([100.0, 140, 101, 102]), "ILS")
    gappy = good.drop(good.index[1])
    calendar = good.index
    rep = quality_report({"GOOD": good, "JUMP": jumpy, "GAP": gappy}, calendar).set_index("symbol")
    assert rep.loc["GOOD", "issue"] == ""
    assert rep.loc["JUMP", "jumps"] == 2
    assert rep.loc["GAP", "missing_days"] == 1
