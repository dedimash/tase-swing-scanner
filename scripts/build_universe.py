"""בונה את data/universe.csv.

מקורות:
  - Yahoo: רשימת כל המניות בבורסת ת"א, שם באנגלית, מטבע, שווי שוק ומחזור.
  - data/hebrew_names.csv: שמות בעברית, כינויים, חברות בת"א 125 (מוויקיפדיה, 27.9.2026),
    הערכה של ת"א 35, וסימולים בארה"ב למניות דואליות.
  - Yahoo .info (רק למניות נזילות): סקטור וענף.

הרצה:  python scripts/build_universe.py           (בלי סקטורים, מהיר)
        python scripts/build_universe.py --sectors (כולל סקטורים, כמה דקות)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DATA_DIR, cache_dir, load_config  # noqa: E402
from core.universe import UNIVERSE_PATH  # noqa: E402

COLUMNS = [
    "symbol", "name_he", "aliases", "name_en", "currency", "financial_currency",
    "market_cap_ils", "avg_turnover_ils", "in_ta125", "ta35_approx", "dual_listed",
    "us_symbol", "sector", "industry", "sector_group", "name_source",
]


def fetch_yahoo_listing() -> pd.DataFrame:
    import yfinance as yf

    query = yf.EquityQuery("eq", ["exchange", "TLV"])
    rows = []
    for offset in range(0, 3000, 250):
        res = yf.screen(query, size=250, offset=offset, sortField="intradaymarketcap", sortAsc=False)
        quotes = res.get("quotes", [])
        rows += quotes
        if len(quotes) < 250:
            break
    df = pd.DataFrame(rows)
    divisor = df["currency"].map({"ILA": 100.0}).fillna(1.0)
    return pd.DataFrame(
        {
            "symbol": df["symbol"],
            "name_en": df["longName"].fillna(df["shortName"]),
            "currency": df["currency"],
            "financial_currency": df.get("financialCurrency"),
            "market_cap_ils": df["marketCap"],
            # הערכה ראשונית לפי ממוצע 3 חודשים. הסריקה עצמה מחשבת מחזור מנתוני המחירים.
            "avg_turnover_ils": (df["averageDailyVolume3Month"] * df["regularMarketPrice"] / divisor).round(0),
        }
    )


def sector_group(sector: str, industry: str) -> str:
    industry = (industry or "").lower()
    if "bank" in industry:
        return "bank"
    if "insurance" in industry:
        return "insurance"
    if sector == "Real Estate" or "real estate" in industry or "reit" in industry:
        return "real_estate"
    return "other" if sector else ""


def fetch_sectors(symbols: list[str]) -> dict[str, dict]:
    import yfinance as yf

    path = cache_dir() / "sectors.json"
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for i, sym in enumerate(symbols):
        if sym in cached:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = yf.Ticker(sym).info
            cached[sym] = {"sector": info.get("sector") or "", "industry": info.get("industry") or ""}
        except Exception as exc:  # Yahoo נכשל לפעמים על מניה בודדת
            print(f"  sector failed for {sym}: {exc}")
            continue
        if i % 20 == 0:
            path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
            print(f"  sectors {i + 1}/{len(symbols)}")
        time.sleep(0.4)
    path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    return cached


def build(with_sectors: bool) -> pd.DataFrame:
    listing = fetch_yahoo_listing()
    names = pd.read_csv(DATA_DIR / "hebrew_names.csv", encoding="utf-8", dtype=str).fillna("")
    unknown = sorted(set(names["symbol"]) - set(listing["symbol"]))
    if unknown:
        print("WARNING: Hebrew-name symbols not found on Yahoo:", unknown)

    df = listing.merge(names, on="symbol", how="left")
    df["name_he"] = df["name_he"].fillna("")
    df["aliases"] = df["aliases"].fillna("")
    df["us_symbol"] = df["us_symbol"].fillna("")
    df["in_ta125"] = df["in_ta125"].fillna("0").astype(int)
    df["ta35_approx"] = df["ta35_approx"].fillna("0").astype(int)
    df["dual_listed"] = (df["us_symbol"] != "").astype(int)
    df["name_source"] = df["source"].fillna("yahoo")

    df["sector"] = ""
    df["industry"] = ""
    if with_sectors:
        min_turnover = load_config()["universe"]["min_avg_turnover_ils"]
        liquid = df.loc[(df["avg_turnover_ils"] >= min_turnover) | (df["in_ta125"] == 1), "symbol"].tolist()
        print(f"fetching sectors for {len(liquid)} liquid stocks")
        sectors = fetch_sectors(liquid)
        df["sector"] = df["symbol"].map(lambda s: sectors.get(s, {}).get("sector", ""))
        df["industry"] = df["symbol"].map(lambda s: sectors.get(s, {}).get("industry", ""))
    df["sector_group"] = [sector_group(s, i) for s, i in zip(df["sector"], df["industry"])]

    df = df[COLUMNS].sort_values("avg_turnover_ils", ascending=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sectors", action="store_true", help="fetch sector/industry for liquid stocks")
    args = parser.parse_args()
    df = build(args.sectors)
    df.to_csv(UNIVERSE_PATH, index=False, encoding="utf-8")
    print(
        f"wrote {len(df)} stocks -> {UNIVERSE_PATH}\n"
        f"  with Hebrew name: {(df['name_he'] != '').sum()}\n"
        f"  TA-125: {df['in_ta125'].sum()}, TA-35 approx: {df['ta35_approx'].sum()}, dual-listed: {df['dual_listed'].sum()}"
    )


if __name__ == "__main__":
    main()
