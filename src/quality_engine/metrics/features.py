"""Stage 2: time-series summarization (reducing series to clustering features).

The output of calculator.py (Stage 1: a time series per metric) flows
through here and is reduced to summary scalars for clustering. Separate
responsibilities: calculator.py = financial logic, features.py = statistical
summarization.
"""

import numpy as np
import scipy.stats

from .calculator import (
    gross_margin, operating_margin, fcf_margin, revenue_growth,
    roic, reinvestment_rate, rnd_to_revenue,
)

METRICS = {
    "gross_margin": gross_margin,
    "operating_margin": operating_margin,
    "fcf_margin": fcf_margin,
    "revenue_growth": revenue_growth,
    "roic": roic,
    "reinvestment_rate": reinvestment_rate,
    "rnd_to_revenue": rnd_to_revenue,
}


def summarize_series(series: list[float]) -> dict:
    """Reduce a metric's time series to summary features for clustering.

    Clustering features (raw values; z-score normalization happens in M4):
      - level_last: last non-NaN value (current level)
      - trend:      OLS slope vs. time (direction + speed)
      - stability:  population std of the non-NaN values (volatility)

    META fields (for audit/filter; do NOT enter clustering distance —
    R² indicates measurement fit rather than quality, and mixing it in
    pollutes distance):
      - trend_r2:   linregress R² (how linear the trend is)
      - n_valid:    how many non-NaN points there were

    NaNs are dropped BUT the original period position (x-axis) is preserved
    so the time gap is not collapsed. Example: [nan, 0.43, 0.44, 0.46, 0.47]
    yields x=[1,2,3,4], y=[0.43,0.44,0.46,0.47].

    Thresholds: level_last >=1, stability >=2, trend >=3 non-NaN points;
    otherwise NaN.
    """
    arr = np.asarray(series, dtype="float64")
    valid_mask = ~np.isnan(arr)
    x = np.where(valid_mask)[0]
    y = arr[valid_mask]
    n_valid = len(y)

    level_last = y[-1] if n_valid >= 1 else np.nan
    stability = np.std(y, ddof=0) if n_valid >= 2 else np.nan

    if n_valid >= 3:
        result = scipy.stats.linregress(x, y)
        trend = result.slope
        trend_r2 = result.rvalue ** 2
    else:
        trend = np.nan
        trend_r2 = np.nan

    return {
        "level_last": level_last,
        "trend": trend,
        "stability": stability,
        "trend_r2": trend_r2,
        "n_valid": n_valid,
    }


def extract_features(cf) -> dict:
    """Extract clustering features from a company's CompanyFinancials.

    Computes every metric and summarizes it via summarize_series. The result
    has TWO compartments:
    - features: ENTERS clustering (level_last, trend, stability) — flat names
    - meta: does NOT enter clustering (trend_r2, n_valid) — audit/filter, in
      a separate compartment to physically prevent data leakage
    """
    features = {}
    meta = {}
    for name, fn in METRICS.items():
        series = fn(cf)
        summary = summarize_series(series)
        features[f"{name}_level_last"] = summary["level_last"]
        features[f"{name}_trend"] = summary["trend"]
        features[f"{name}_stability"] = summary["stability"]
        meta[f"{name}_trend_r2"] = summary["trend_r2"]
        meta[f"{name}_n_valid"] = summary["n_valid"]
    return {"features": features, "meta": meta}
