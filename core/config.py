"""טעינת config.yaml ונתיבים משותפים."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=4)
def _read_config(path: str, mtime: float) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | None = None) -> dict:
    """ההגדרות מ-config.yaml. נקראות מחדש בכל פעם שהקובץ משתנה, גם בלי להפעיל מחדש את השרת."""
    cfg_path = Path(path) if path else CONFIG_PATH
    return _read_config(str(cfg_path), cfg_path.stat().st_mtime)


def cache_dir() -> Path:
    d = ROOT / load_config()["data"]["cache_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d
