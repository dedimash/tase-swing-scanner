"""רשימת המניות וחיפוש לפי שם בעברית, שם באנגלית או סימול."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache

import pandas as pd

from core.config import DATA_DIR, load_config

UNIVERSE_PATH = DATA_DIR / "universe.csv"

_FINAL_LETTERS = str.maketrans("ךםןףץ", "כמנפצ")
_QUOTES = re.compile(r"[\"'`׳״]+")
_SEPARATORS = re.compile(r"[.,\-–_()/\\]+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """מנרמל טקסט לחיפוש: בלי ניקוד, בלי גרשיים ומקפים, אותיות סופיות כרגילות."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().translate(_FINAL_LETTERS)
    text = _QUOTES.sub("", text)
    text = _SEPARATORS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _compact(text: str) -> str:
    return text.replace(" ", "")


@lru_cache(maxsize=1)
def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_PATH, encoding="utf-8", dtype={"us_symbol": "str", "aliases": "str"})
    for col in ("name_he", "aliases", "name_en", "us_symbol", "sector", "industry", "sector_group"):
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def scan_universe(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """המניות שנכנסות לסריקה: מחזור כספי ממוצע מעל הסף."""
    df = load_universe() if df is None else df
    min_turnover = load_config()["universe"]["min_avg_turnover_ils"]
    return df[df["avg_turnover_ils"] >= min_turnover]


def display_name(row: pd.Series) -> str:
    """השם בעברית, ואם אין אז באנגלית, ואם אין אז הסימול (גם כשהסימול הוא האינדקס של השורה)."""
    return row.get("name_he") or row.get("name_en") or row.get("symbol", row.name)


def _candidates(row: pd.Series) -> list[str]:
    names = [row["name_he"], row["name_en"], row["symbol"].removesuffix(".TA")]
    names += [a for a in str(row.get("aliases", "")).split(";") if a]
    return [n for n in (normalize(x) for x in names) if n]


def _score(query: str, candidate: str) -> float:
    if query == candidate or _compact(query) == _compact(candidate):
        return 100.0
    if candidate.startswith(query):
        return 92.0
    if any(word.startswith(query) for word in candidate.split()):
        return 88.0
    if len(query) >= 2 and (query in candidate or _compact(query) in _compact(candidate)):
        return 80.0
    return 70.0 * SequenceMatcher(None, query, candidate).ratio()


def search(query: str, limit: int = 8, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """מחזיר את המניות הכי מתאימות לשאילתה, מהמתאימה ביותר. מניות נזילות קודמות בשוויון."""
    df = load_universe() if df is None else df
    q = normalize(query)
    if not q:
        return df.iloc[0:0]
    scores = []
    for _, row in df.iterrows():
        scores.append(max((_score(q, c) for c in _candidates(row)), default=0.0))
    out = df.assign(score=scores)
    # כשיש התאמה טובה בשם, לא מציגים הצעות מקורבות שרק דומות באותיות.
    threshold = 80 if out["score"].max() >= 88 else 50
    out = out[out["score"] >= threshold]
    return out.sort_values(["score", "avg_turnover_ils"], ascending=[False, False]).head(limit)
