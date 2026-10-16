"""ניסוי שיפורים לתבנית הפריצה, בשיטה שמונעת התאמת יתר.

1. כל רעיון נבדק לבד, רק על תקופת הכיול (3 השנים הראשונות).
2. הרעיונות שעזרו בכיול משולבים.
3. השילוב נבדק פעם אחת על תקופת הבדיקה, שלא שימשה לבחירה.

הרצה:  python scripts/experiment_breakout.py train
        python scripts/experiment_breakout.py test --variant <name>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import Costs, Sizing, backtest_window, edge_card, portfolio, run_setup  # noqa: E402
from core.config import RESULTS_DIR, load_config  # noqa: E402
from core.data import PriceStore  # noqa: E402
from core.setups_daily import Breakout20  # noqa: E402
from core.universe import load_universe  # noqa: E402

# רעיונות שנקבעו מראש, לפני שראינו את התוצאות שלהם.
IDEAS = {
    "baseline": {},
    "above_sma200": {"above_sma200": True},
    "rs_positive": {"min_rs": 0.0},
    "contraction": {"max_atr_contraction": 1.0},
    "strong_close": {"min_close_location": 0.75},
    "stop_entry": {"entry_mode": "stop_above_high"},
    "tighter_stop": {"max_stop_atr": 1.0},
    "breakeven": {"breakeven_r": 1.0},
    "trail": {"trail_atr": 2.0, "target_r": None},
    "rank_rs": {"rank_by": "rs"},
}

LOG = RESULTS_DIR / "experiments_breakout.json"


def load_data():
    u = load_universe()
    u = u[u["in_ta125"] == 1]
    store = PriceStore(currencies=dict(zip(u["symbol"], u["currency"])))
    frames = store.cached_daily(u["symbol"].tolist())
    idx = load_config()["data"]["index_symbol"]
    return frames, store.cached_daily([idx])[idx]


def evaluate(overrides: dict, frames, index_df, start, end) -> dict:
    cfg = load_config()
    costs, sizing = Costs.from_config(), Sizing.from_config()
    trades = run_setup(Breakout20(**overrides), frames, index_df, start, end, costs, sizing)
    card = edge_card(trades)
    book = portfolio(trades, cfg["risk"]["max_open_positions"], sizing.risk_pct, sizing.capital_ils)
    return {
        "overrides": overrides,
        "trades": card.get("trades", 0),
        "win_rate": card.get("win_rate"),
        "expectancy_r": card.get("expectancy_r"),
        "expectancy_se": card.get("expectancy_se"),
        "profit_factor": card.get("profit_factor"),
        "cagr_pct": book.get("cagr_pct"),
        "max_drawdown_pct": book.get("max_drawdown_pct"),
        "avg_exposure_pct": book.get("avg_exposure_pct"),
    }


def show(rows: dict) -> None:
    print(f"{'variant':<16}{'trades':>7}{'win':>7}{'exp R':>8}{'±SE':>7}{'PF':>6}{'CAGR%':>7}{'DD%':>7}{'expo%':>7}")
    for name, r in rows.items():
        print(
            f"{name:<16}{r['trades']:>7}{(r['win_rate'] or 0):>7.2f}{(r['expectancy_r'] or 0):>8.3f}"
            f"{(r['expectancy_se'] or 0):>7.3f}{(r['profit_factor'] or 0):>6.2f}{(r['cagr_pct'] or 0):>7.1f}"
            f"{(r['max_drawdown_pct'] or 0):>7.1f}{(r['avg_exposure_pct'] or 0):>7.1f}"
        )


def save(section: str, payload: dict) -> None:
    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else {}
    log[section] = {"run_at": datetime.now().isoformat(timespec="seconds"), **payload}
    LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["train", "combo", "test"])
    parser.add_argument("--variant", default="", help="JSON of overrides for combo/test")
    args = parser.parse_args()

    frames, index_df = load_data()
    cfg = load_config()["backtest"]
    start, end = backtest_window(frames, cfg["years"])
    split = start + pd.DateOffset(years=cfg["walk_forward"]["train_years"])
    period = {"train": [str(start.date()), str(split.date())], "test": [str(split.date()), str(end.date())]}

    if args.phase == "train":
        rows = {name: evaluate(o, frames, index_df, start, split) for name, o in IDEAS.items()}
        show(rows)
        save("train_ideas", {"period": period["train"], "results": rows})
        return

    overrides = json.loads(args.variant)
    if args.phase == "combo":
        rows = {"baseline": evaluate({}, frames, index_df, start, split), "combo": evaluate(overrides, frames, index_df, start, split)}
        show(rows)
        save("train_combo", {"period": period["train"], "results": rows})
    else:
        rows = {"baseline": evaluate({}, frames, index_df, split, end), "combo": evaluate(overrides, frames, index_df, split, end)}
        show(rows)
        save("test_once", {"period": period["test"], "results": rows})


if __name__ == "__main__":
    main()
