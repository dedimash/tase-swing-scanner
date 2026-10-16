"""מריץ בקטסט ו-walk-forward לכל התבניות, ושומר את התוצאות ל-results/.

רץ מקומית (לא בשרת). האפליקציה רק קוראת את הקבצים שנוצרים כאן.
הרצה:  python scripts/run_backtests.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import (  # noqa: E402
    Costs,
    Sizing,
    backtest_window,
    edge_card,
    portfolio,
    run_setup,
    walk_forward,
)
from core.config import RESULTS_DIR, load_config  # noqa: E402
from core.data import PriceStore, quality_report  # noqa: E402
from core.regime import breadth, market_event_days, market_regime  # noqa: E402
from core.scoring import calibration, score_signals  # noqa: E402
from core.setups_daily import SETUPS  # noqa: E402
from core.universe import load_universe  # noqa: E402


def backtest_universe() -> pd.DataFrame:
    """מניות ת"א 125 לפי הרכב המדד הנוכחי. שימו לב: זה יוצר הטיית שורדים בבקטסט."""
    u = load_universe()
    return u[u["in_ta125"] == 1]


def attach_scores(setup, trades: pd.DataFrame, frames, index_df, regime, breadth_series) -> pd.DataFrame:
    """מוסיף לכל עסקה את הציון שהיה לאיתות ביום שבו הוא נוצר."""
    if trades.empty:
        return trades
    parts = []
    for sym in trades["symbol"].unique():
        df = frames[sym]
        sig = setup.detect(df, index_df["close"])
        sc = score_signals(sig, df, index_df["close"], regime, breadth_series)
        parts.append(pd.DataFrame({"symbol": sym, "signal_date": sc.index, "score": sc["score"].to_numpy()}))
    scores = pd.concat(parts).drop_duplicates(["symbol", "signal_date"])
    return trades.merge(scores, on=["symbol", "signal_date"], how="left")


def main() -> None:
    cfg = load_config()
    universe = backtest_universe()
    symbols = universe["symbol"].tolist()
    currencies = dict(zip(universe["symbol"], universe["currency"]))
    store = PriceStore(currencies=currencies)

    print(f"loading daily data for {len(symbols)} stocks + index")
    frames = store.daily(symbols)
    index_sym = cfg["data"]["index_symbol"]
    index_df = store.daily([index_sym])[index_sym]
    print(f"  got {len(frames)} stocks, index rows {len(index_df)}")

    quality = quality_report(frames, index_df.index)
    flagged = quality[quality["issue"] != ""]
    print(f"  data quality: {len(flagged)} stocks with issues")

    RESULTS_DIR.mkdir(exist_ok=True)
    costs, sizing = Costs.from_config(), Sizing.from_config()
    start, end = backtest_window(frames, cfg["backtest"]["years"])
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe": f"ת\"א 125 לפי ההרכב הנוכחי ({len(frames)} מניות)",
        "survivorship_note": "הבקטסט רץ על הרכב המדד של היום, ולכן כנראה אופטימי מדי.",
        "costs": costs.__dict__,
        "sizing": sizing.__dict__,
        "data_quality_issues": flagged.drop(columns=["first", "last"], errors="ignore").to_dict("records"),
        "setups": {},
    }

    max_pos = cfg["risk"]["max_open_positions"]
    bench = index_df["close"][(index_df.index >= start) & (index_df.index < end)]
    bench_years = (bench.index[-1] - bench.index[0]).days / 365.25
    output["benchmark"] = {
        "name": "מדד ת\"א 125 (קנה והחזק)",
        "cagr_pct": round(((bench.iloc[-1] / bench.iloc[0]) ** (1 / bench_years) - 1) * 100, 2),
        "max_drawdown_pct": round(float((bench / bench.cummax() - 1).min()) * 100, 2),
        "curve": [[str(d.date()), round(float(v), 2)] for d, v in bench.items()],
    }
    print(f"  benchmark TA-125: CAGR {output['benchmark']['cagr_pct']}%, max DD {output['benchmark']['max_drawdown_pct']}%")

    def summary(label: str, card: dict, book: dict) -> None:
        print(
            f"  {label}: {card['trades']} trades, win {card.get('win_rate')}, "
            f"expectancy {card.get('expectancy_r')}R, PF {card.get('profit_factor')} | "
            f"portfolio CAGR {book.get('cagr_pct')}%, max DD {book.get('max_drawdown_pct')}%"
        )

    event_days = market_event_days(frames)
    output["market_event_days"] = [str(d.date()) for d in event_days]
    print(f"  market-wide volume spike days: {len(event_days)}")
    regime_series = market_regime(index_df)
    breadth_series = breadth(frames)

    active_trades = []
    for key, setup_cls in SETUPS.items():
        print(f"running {key}")
        wf = walk_forward(setup_cls, frames, index_df, costs, sizing, event_days)
        # הפרמטרים הפעילים: מה שנבחר בתקופת הכיול. אם לא היו מספיק עסקאות, ברירת המחדל.
        setup = setup_cls(**wf.get("best_params", {}))
        trades = run_setup(setup, frames, index_df, start, end, costs, sizing, event_days)
        trades = attach_scores(setup, trades, frames, index_df, regime_series, breadth_series)
        trades.to_csv(RESULTS_DIR / f"trades_{key}.csv", index=False, encoding="utf-8")
        card = edge_card(trades)
        book = portfolio(trades, max_pos, sizing.risk_pct, sizing.capital_ils)
        if wf.get("passed"):
            active_trades.append(trades)

        entry = {
            "name_he": setup.name_he,
            "description_he": setup.description_he,
            "params": setup.params,
            "passed": bool(wf.get("passed")),
            "fail_reasons": wf.get("fail_reasons", []),
            "full_period": card,
            "portfolio": book,
            "walk_forward": wf,
            "calibration": calibration(trades),
        }
        summary("full period (active params)", card, book)
        cal = entry["calibration"]
        print(f"  score calibration: spearman {cal.get('spearman')} ±{cal.get('spearman_se')} -> {cal.get('note')}")

        mode = setup.params.get("regime", "any")
        if mode != "any":
            alt = run_setup(setup_cls(**wf.get("best_params", {}), regime="any"), frames, index_df, start, end, costs, sizing, event_days)
            alt_book = portfolio(alt, max_pos, sizing.risk_pct, sizing.capital_ils)
            entry["without_regime_filter"] = {
                "full_period": edge_card(alt),
                "portfolio": {k: v for k, v in alt_book.items() if k != "curve"},
            }
            summary("without regime filter", entry["without_regime_filter"]["full_period"], alt_book)

        if "test" in wf:
            print(f"  walk-forward best {wf['best_params']}")
            for part in ("train", "test", "combined"):
                c = wf[part]
                print(f"    {part:<8}: {c['trades']} trades, expectancy {c.get('expectancy_r')}R ±{c.get('expectancy_se')}")
        print(f"  PASSED: {wf.get('passed')}  {wf.get('fail_reasons')}")
        output["setups"][key] = entry

    if active_trades:
        merged = pd.concat(active_trades, ignore_index=True)
        book = portfolio(merged, max_pos, sizing.risk_pct, sizing.capital_ils)
        output["portfolio_all_active"] = book
        print(f"all passed setups together: CAGR {book.get('cagr_pct')}%, max DD {book.get('max_drawdown_pct')}%, exposure {book.get('avg_exposure_pct')}%")

    path = RESULTS_DIR / "edge_cards.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
