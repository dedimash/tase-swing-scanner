import os

from core.config import load_config


def test_config_reloads_when_file_changes(tmp_path):
    """שרת שרץ זמן רב (כמו ב-Streamlit Cloud) צריך לראות שינוי ב-config.yaml בלי הפעלה מחדש."""
    path = tmp_path / "config.yaml"
    path.write_text("app:\n  title: ישן\n", encoding="utf-8")
    os.utime(path, (1_000_000, 1_000_000))
    assert load_config(str(path))["app"]["title"] == "ישן"

    path.write_text("app:\n  title: חדש\n", encoding="utf-8")
    os.utime(path, (2_000_000, 2_000_000))
    assert load_config(str(path))["app"]["title"] == "חדש"


def test_real_config_has_title():
    assert load_config().get("app", {}).get("title")
