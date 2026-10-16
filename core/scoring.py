"""ציון 0 עד 100 לאיתות, וכיול שלו מול תוצאות העבר.

המשקלות נקבעו מראש ולא כוילו על הנתונים:
  איכות התבנית (80%): מגמה 25%, חוזק יחסי 20%, מחזור 15%, מרחק לסטופ 20%, מקום עד ההתנגדות 20%
  הקשר שוק (20%):     משטר שוק 50%, רוחב שוק 50%
(החלק הפונדמנטלי מהתוכנית עוד לא נבנה, ולכן המשקל שלו חולק בין השאר.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.indicators import relative_strength, sma
from core.regime import DOWN, MIXED, UP, regime_on

SETUP_WEIGHTS = {"trend": 0.25, "rs": 0.20, "volume": 0.15, "risk": 0.20, "room": 0.20}
SETUP_SHARE, CONTEXT_SHARE = 0.8, 0.2
COMPONENT_NAMES_HE = {
    "trend": "מגמה", "rs": "חוזק מול המדד", "volume": "מחזור", "risk": "מרחק לסטופ",
    "room": "מקום עד ההתנגדות", "regime": "מצב השוק", "breadth": "רוחב השוק",
}


def _clip(x) -> pd.Series:
    return pd.Series(x).clip(0.0, 1.0)


def score_signals(
    signals: pd.DataFrame,
    df: pd.DataFrame,
    index_close: pd.Series | None,
    regime: pd.Series | None,
    breadth_series: pd.Series | None,
) -> pd.DataFrame:
    """מחזיר לכל איתות את רכיבי הציון (0 עד 1) ואת הציון הכולל (0 עד 100)."""
    if signals.empty:
        return pd.DataFrame(index=signals.index, columns=[*SETUP_WEIGHTS, "regime", "breadth", "score"], dtype=float)
    dates = signals.index
    c = df["close"]
    s50, s200 = sma(c, 50), sma(c, 200)
    trend = ((c > s50).astype(float) + (s50 > s200).astype(float) + (c > s200).astype(float)) / 3.0
    rs = relative_strength(c, index_close, 63) if index_close is not None else pd.Series(np.nan, index=c.index)
    resistance = df["high"].shift(1).rolling(250, min_periods=60).max()

    close = signals["close"]
    risk_per_share = (close - signals["stop"]).clip(lower=1e-9)
    risk_pct = risk_per_share / close
    res = resistance.reindex(dates)
    room_r = (res - close) / risk_per_share
    room = np.where(res.isna() | (close >= res), 1.0, room_r / 2.0)
    volume = (
        (signals["volume_ratio"] - 1.0) / 2.0 if "volume_ratio" in signals else pd.Series(0.5, index=dates)
    )

    out = pd.DataFrame(index=dates)
    out["trend"] = trend.reindex(dates).fillna(0.5).to_numpy()
    out["rs"] = _clip(((rs.reindex(dates) + 0.15) / 0.30).fillna(0.5)).to_numpy()
    out["volume"] = _clip(volume.fillna(0.5)).to_numpy()
    out["risk"] = _clip(1.0 - (risk_pct - 0.02) / 0.08).to_numpy()
    out["room"] = _clip(room).to_numpy()
    if regime is not None and not regime.empty:
        labels = regime_on(dates, regime)
        out["regime"] = labels.map({UP: 1.0, MIXED: 0.5, DOWN: 0.0}).fillna(0.5).to_numpy()
    else:
        out["regime"] = 0.5
    if breadth_series is not None and not breadth_series.empty:
        out["breadth"] = breadth_series.reindex(breadth_series.index.union(dates)).ffill().reindex(dates).fillna(0.5).to_numpy()
    else:
        out["breadth"] = 0.5

    setup_part = sum(out[k] * w for k, w in SETUP_WEIGHTS.items())
    context_part = 0.5 * out["regime"] + 0.5 * out["breadth"]
    out["score"] = (100.0 * (SETUP_SHARE * setup_part + CONTEXT_SHARE * context_part)).round(1)
    return out


def calibration(trades: pd.DataFrame, buckets: int = 10) -> dict:
    """האם ציון גבוה יותר באמת הלך עם תוצאה טובה יותר? תוצאה ממוצעת (R) לפי עשירוני ציון."""
    data = trades.dropna(subset=["score", "r"])
    data = data[data["exit_reason"] != "open"]
    if len(data) < buckets * 10:
        return {"buckets": [], "spearman": None, "note": "אין מספיק עסקאות לכיול"}
    ranks = data["score"].rank(method="first")
    data = data.assign(bucket=pd.qcut(ranks, buckets, labels=False) + 1)
    table = data.groupby("bucket").agg(
        trades=("r", "size"), score_min=("score", "min"), score_max=("score", "max"),
        avg_r=("r", "mean"), win_rate=("r", lambda x: float((x > 0).mean())),
    )
    rho = float(data["score"].rank().corr(data["r"].rank()))
    se = 1.0 / np.sqrt(len(data) - 1)
    rows = [
        {"bucket": int(b), "trades": int(r.trades), "score_from": round(r.score_min, 1), "score_to": round(r.score_max, 1),
         "avg_r": round(r.avg_r, 3), "win_rate": round(r.win_rate, 3)}
        for b, r in table.iterrows()
    ]
    if rho > 2 * se:
        note = "ציון גבוה הלך עם תוצאות טובות יותר. אפשר להשתמש בציון לסדר את ההזדמנויות."
    elif rho < -2 * se:
        note = "ציון גבוה הלך דווקא עם תוצאות פחות טובות. אסור לסמוך על הציון."
    else:
        note = "לא נמצא קשר מובהק בין הציון לתוצאה. הציון משמש רק לסדר, לא כסימן לאיכות."
    return {"buckets": rows, "spearman": round(rho, 3), "spearman_se": round(se, 3), "note": note}
