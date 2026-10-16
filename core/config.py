"""טעינת config.yaml ונתיבים משותפים."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cache_dir() -> Path:
    d = ROOT / load_config()["data"]["cache_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d
