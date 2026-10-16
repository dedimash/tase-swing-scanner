import numpy as np
import pandas as pd
import pytest

from core.backtest import Costs, Sizing, edge_card, portfolio, simulate

NO_COSTS = Costs(0.0, 0.0, 0.0)
SIZING = Sizing(100_000, 1.0)


def bars(rows):
    """rows: [open, high, low, close]"""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=pd.date_range("2026-01-05", periods=len(rows), freq="B"))
    df["volume"] = 1000.0
    df["turnover"] = df["close"] * df["volume"]
    return df


def signal(df, day, stop=95.0, target_r=2.0, max_hold=3):
    return pd.DataFrame({"stop": [stop], "target_r": [target_r], "max_hold": [max_hold]}, index=[df.index[day]])


def test_target_hit():
    df = bars([[99, 100, 98, 100], [100, 104, 99, 103], [103, 111, 102, 108]])
    t = simulate("X", df, signal(df, 0), NO_COSTS, SIZING)
    assert t[0]["exit_reason"] == "target"
    assert t[0]["exit"] == pytest.approx(110.0)
    assert t[0]["r"] == pytest.approx(2.0)


def test_stop_hit():
    df = bars([[99, 100, 98, 100], [100, 101, 94, 96]])
    t = simulate("X", df, signal(df, 0), NO_COSTS, SIZING)
    assert t[0]["exit_reason"] == "stop"
    assert t[0]["r"] == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open():
    df = bars([[99, 100, 98, 100], [100, 101, 99, 100], [90, 92, 89, 91]])
    t = simulate("X", df, signal(df, 0), NO_COSTS, SIZING)
    assert t[0]["exit_reason"] == "gap_stop"
    assert t[0]["r"] == pytest.approx(-2.0)


def test_stop_checked_before_target_in_same_bar():
    df = bars([[99, 100, 98, 100], [100, 120, 90, 100]])
    t = simulate("X", df, signal(df, 0), NO_COSTS, SIZING)
    assert t[0]["exit_reason"] == "stop"


def test_time_exit_at_close():
    df = bars([[99, 100, 98, 100], [100, 101, 99, 100], [100, 102, 99, 101], [101, 103, 100, 102], [102, 130, 101, 125]])
    t = simulate("X", df, signal(df, 0, max_hold=3), NO_COSTS, SIZING)
    assert t[0]["exit_reason"] == "time"
    assert t[0]["exit"] == pytest.approx(102.0)
    assert t[0]["hold_days"] == 3


def test_no_entry_when_open_below_stop():
    df = bars([[99, 100, 98, 100], [94, 96, 93, 95]])
    assert simulate("X", df, signal(df, 0), NO_COSTS, SIZING) == []


def test_signal_on_last_bar_is_not_a_trade():
    df = bars([[99, 100, 98, 100]])
    assert simulate("X", df, signal(df, 0), NO_COSTS, SIZING) == []


def test_open_trade_when_data_ends():
    df = bars([[99, 100, 98, 100], [100, 101, 99, 100]])
    t = simulate("X", df, signal(df, 0, max_hold=5), NO_COSTS, SIZING)
    assert t[0]["exit_reason"] == "open"
    assert edge_card(pd.DataFrame(t))["trades"] == 0


def test_slippage_reduces_r():
    df = bars([[99, 100, 98, 100], [100, 104, 99, 103], [103, 111, 102, 108]])
    t = simulate("X", df, signal(df, 0), Costs(0.0, 0.0, 0.1), SIZING)
    assert t[0]["r"] < 2.0


def test_min_commission_applied():
    df = bars([[99, 100, 98, 100], [100, 104, 99, 103], [103, 111, 102, 108]])
    free = simulate("X", df, signal(df, 0), NO_COSTS, SIZING)[0]
    paid = simulate("X", df, signal(df, 0), Costs(0.0, 10.0, 0.0), SIZING)[0]
    assert free["pnl_ils"] - paid["pnl_ils"] == pytest.approx(20.0)


def test_one_trade_at_a_time_per_symbol():
    df = bars([[99, 100, 98, 100], [100, 101, 99, 100], [100, 101, 99, 100], [100, 101, 99, 100], [100, 101, 99, 100]])
    sigs = pd.DataFrame({"stop": [95.0, 95.0], "target_r": [2.0, 2.0], "max_hold": [3, 3]}, index=df.index[[0, 1]])
    assert len(simulate("X", df, sigs, NO_COSTS, SIZING)) == 1


def sig_with(df, **cols):
    base = {"stop": 95.0, "target_r": 2.0, "max_hold": 5, "atr": 2.0}
    base.update(cols)
    return pd.DataFrame({k: [v] for k, v in base.items()}, index=[df.index[0]])


def test_entry_trigger_not_reached_means_no_trade():
    df = bars([[99, 100, 98, 100], [100, 100.5, 99, 100]])
    assert simulate("X", df, sig_with(df, entry_trigger=101.0), NO_COSTS, SIZING) == []


def test_entry_trigger_fills_at_trigger_or_gap_open():
    df = bars([[99, 100, 98, 100], [100, 103, 99, 102], [102, 103, 101, 102]])
    t = simulate("X", df, sig_with(df, entry_trigger=101.0, max_hold=2), NO_COSTS, SIZING)
    assert t[0]["entry"] == pytest.approx(101.0)
    df2 = bars([[99, 100, 98, 100], [102, 103, 101, 102], [102, 103, 101, 102]])
    t2 = simulate("X", df2, sig_with(df2, entry_trigger=101.0, max_hold=2), NO_COSTS, SIZING)
    assert t2[0]["entry"] == pytest.approx(102.0)


def test_no_target_holds_until_time():
    df = bars([[99, 100, 98, 100], [100, 130, 99, 125], [125, 126, 124, 125]])
    t = simulate("X", df, sig_with(df, target_r=np.nan, max_hold=2), NO_COSTS, SIZING)
    assert t[0]["exit_reason"] == "time"
    assert t[0]["r"] == pytest.approx(5.0)


def test_breakeven_moves_stop_to_entry():
    # כניסה 100, סטופ 95 (R=5). סגירה 106 >= 100+1R -> הסטופ עובר ל-100. למחרת נוגעים ב-99.
    df = bars([[99, 100, 98, 100], [100, 107, 99.5, 106], [104, 104, 99, 100]])
    t = simulate("X", df, sig_with(df, target_r=np.nan, breakeven_r=1.0), NO_COSTS, SIZING)
    assert t[0]["exit_reason"] == "trail"
    assert t[0]["r"] == pytest.approx(0.0)


def test_trailing_stop_follows_best_close():
    # ATR=2, trail 2*ATR=4. סגירה הכי גבוהה 112 -> סטופ 108.
    df = bars([[99, 100, 98, 100], [100, 106, 99, 105], [105, 113, 104, 112], [111, 111, 107, 108]])
    t = simulate("X", df, sig_with(df, target_r=np.nan, trail_atr=2.0), NO_COSTS, SIZING)
    assert t[0]["exit_reason"] == "trail"
    assert t[0]["exit"] == pytest.approx(108.0)


def test_position_capped_without_leverage():
    # סטופ צמוד מאוד: לפי סיכון היינו קונים 100% מההון ויותר. התקרה 25%.
    df = bars([[99, 100, 98, 100], [100, 101, 99.8, 100.5]])
    capped = Sizing(100_000, 1.0, 25.0)
    t = simulate("X", df, sig_with(df, stop=99.9, max_hold=1), NO_COSTS, capped)
    assert t[0]["position_frac"] <= 0.25 + 1e-9
    assert t[0]["risk_frac"] < 0.01


def test_portfolio_respects_max_positions_and_compounds():
    d = pd.to_datetime
    trades = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "entry_date": d(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-10"]),
            "exit_date": d(["2025-01-05", "2025-01-06", "2025-01-07", "2025-01-12"]),
            "exit_reason": ["target", "stop", "target", "stop"],
            "rank_score": [2.0, 3.0, 1.0, 1.0],
            "r": [2.0, -1.0, 2.0, -1.0],
        }
    )
    book = portfolio(trades, max_positions=2, risk_pct=1.0, capital=100.0)
    # C נדחית כי A ו-B פתוחות. נלקחות A, B, D.
    assert book["trades_taken"] == 3
    assert book["trades_skipped"] == 1
    assert book["final_multiple"] == pytest.approx(1.02 * 0.99 * 0.99, abs=1e-3)
    assert book["max_drawdown_pct"] == pytest.approx((0.99 * 0.99 - 1) * 100, abs=0.01)


def test_edge_card_numbers():
    trades = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2026-01-01"]),
            "exit_date": pd.to_datetime(["2025-01-05", "2025-02-05", "2025-03-05", "2026-01-05"]),
            "exit_reason": ["target", "stop", "stop", "target"],
            "hold_days": [3, 2, 1, 4],
            "r": [2.0, -1.0, -1.0, 2.0],
        }
    )
    card = edge_card(trades)
    assert card["trades"] == 4
    assert card["win_rate"] == 0.5
    assert card["expectancy_r"] == 0.5
    assert card["profit_factor"] == 2.0
    assert card["max_drawdown_r"] == -2.0
    assert set(card["by_year"]) == {"2025", "2026"}
