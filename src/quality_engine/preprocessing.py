"""Prepare the raw feature matrix for clustering: drop R&D, missing-data threshold, rank transform.

Operates ONLY on the feature matrix. The meta matrix (r2, n_valid) does not
enter here — the leakage separation is physically enforced.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

RND_PREFIX = "rnd_to_revenue"


def prepare_matrix(feature_df: pd.DataFrame, max_nan: int = 0) -> dict:
    """Prepare the raw feature matrix for clustering.

    Steps:
    1. Drop R&D features (rnd_to_revenue_*): NaN in ~65% of the universe,
       carries no information for general clustering (sector-conditional;
       may come back in a tech sub-universe later).
    2. Missing-data threshold: drop companies whose per-row NaN count
       exceeds max_nan. max_nan=0 (default) means only fully-clean
       companies remain (no imputation needed; K-means does not accept NaN).
    3. Universe-wide rank transform (pd.DataFrame.rank, method="average",
       pct=True): each feature is mapped to its universe-relative percentile
       in [0, 1]. Even RobustScaler could not tame structural outliers
       (e.g. MCK with z~31); rank fixes outliers at the root (highest = 1.0,
       lowest = 0.0), drops no company, and fabricates no data (rank is real
       information). MCK remains the highest-ROIC company (rank=1.0) but
       loses its overwhelming influence — clustering measures distance over
       rank, not absolute magnitude.

       NUANCE — RANK IS UNIVERSE-RELATIVE: when a new company is added the
       ENTIRE universe must be re-ranked. Stateless: there is no
       "fit then transform" behavior like RobustScaler/StandardScaler;
       that is why "scaler" in the return is None.

    Operates ONLY on the feature matrix — meta (r2, n_valid) does NOT enter
    here (leakage separation preserved).

    Returns (dict):
    - scaled: pd.DataFrame, kept companies × remaining features, rank [0,1]
    - scaler: None (rank is stateless; new data requires re-ranking the universe)
    - kept: list of kept tickers
    - dropped: list of dropped tickers (for audit/recovery)
    - feature_names: remaining feature names (order matters)
    """
    rnd_cols = [c for c in feature_df.columns if c.startswith(RND_PREFIX)]
    df = feature_df.drop(columns=rnd_cols)

    nan_per_row = df.isna().sum(axis=1)
    keep_mask = nan_per_row <= max_nan
    kept_df = df[keep_mask]
    kept = list(kept_df.index)
    dropped = list(df[~keep_mask].index)

    logger.info(
        f"Filter: {len(kept)} kept, {len(dropped)} dropped "
        f"(R&D removed, max_nan={max_nan}). Dropped: {dropped}"
    )

    # rank transform: each feature mapped to its universe-relative percentile [0,1]
    # method="average" (ties get the average rank), pct=True (0-1 normalized)
    scaled = kept_df.rank(method="average", pct=True)

    return {
        "scaled": scaled,
        "scaler": None,
        "kept": kept,
        "dropped": dropped,
        "feature_names": list(kept_df.columns),
    }
