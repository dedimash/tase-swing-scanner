import pandas as pd

from core.universe import UNIVERSE_PATH, load_universe, normalize, search


def small_universe():
    return pd.DataFrame(
        {
            "symbol": ["LUMI.TA", "POLI.TA", "ICL.TA", "MVNE.TA"],
            "name_he": ["בנק לאומי לישראל", "בנק הפועלים", "איי.סי.אל גרופ", 'מבנה נדל"ן'],
            "aliases": ["לאומי;בנק לאומי", "פועלים;הפועלים", "כיל;כימיקלים לישראל", "מבנה"],
            "name_en": ["Bank Leumi le-Israel B.M.", "Bank Hapoalim B.M.", "ICL Group Ltd", "Mivne Real Estate"],
            "avg_turnover_ils": [2e8, 2e8, 3e7, 1e7],
        }
    )


def test_normalize_final_letters_and_quotes():
    assert normalize('מבנה נדל"ן') == normalize("מבנה נדלן")
    assert normalize("לאומי") == normalize("לאומי ")
    assert normalize("שלום") == "שלומ"


def test_search_hebrew_partial_name():
    assert search("לאומי", df=small_universe()).iloc[0]["symbol"] == "LUMI.TA"
    assert search("פועלים", df=small_universe()).iloc[0]["symbol"] == "POLI.TA"


def test_search_alias_symbol_and_english():
    u = small_universe()
    assert search("כיל", df=u).iloc[0]["symbol"] == "ICL.TA"
    assert search("lumi", df=u).iloc[0]["symbol"] == "LUMI.TA"
    assert search("leumi", df=u).iloc[0]["symbol"] == "LUMI.TA"
    assert search("נדלן", df=u).iloc[0]["symbol"] == "MVNE.TA"


def test_search_typo_still_finds():
    assert search("לאומיי", df=small_universe()).iloc[0]["symbol"] == "LUMI.TA"


def test_search_empty_query():
    assert search("  ", df=small_universe()).empty


def test_real_universe_leumi():
    if not UNIVERSE_PATH.exists():
        return
    load_universe.cache_clear()
    assert search("לאומי").iloc[0]["symbol"] == "LUMI.TA"
