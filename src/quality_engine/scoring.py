"""QScore: produces a quality score from the rank matrix.

Flow: direction correction -> factor groups (in-group mean, prevents
accidental weighting) -> factor weights (default EQUAL, avoids confirmation
bias) -> 0-100 -> quintile buckets (Q1=premium).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Features whose direction needs correcting (high rank = BAD -> invert with 1-rank):
# - all stability features (high std = volatile = bad)
# - reinvestment level + trend (capital-light thesis: low reinvestment = good)
INVERT_SUFFIXES = ("_stability",)
INVERT_EXACT = {"reinvestment_rate_level_last", "reinvestment_rate_trend"}

# 4 macro-factor groups (feature -> group). 18 features.
FACTOR_GROUPS = {
    "profitability": [
        "gross_margin_level_last", "operating_margin_level_last",
        "fcf_margin_level_last", "gross_margin_trend",
        "operating_margin_trend", "fcf_margin_trend",
    ],
    "capital_efficiency": [
        "roic_level_last", "roic_trend",
        "reinvestment_rate_level_last", "reinvestment_rate_trend",
    ],
    "growth": [
        "revenue_growth_level_last", "revenue_growth_trend",
    ],
    "stability": [
        "gross_margin_stability", "operating_margin_stability",
        "fcf_margin_stability", "roic_stability",
        "reinvestment_rate_stability", "revenue_growth_stability",
    ],
}


def _apply_directions(ranked: pd.DataFrame) -> pd.DataFrame:
    """Direction correction: invert 'low=good' features with 1-rank.
    After this, ALL features have high value = good (consistent direction).
    """
    df = ranked.copy()
    for col in df.columns:
        if col.endswith(INVERT_SUFFIXES) or col in INVERT_EXACT:
            df[col] = 1.0 - df[col]
    return df


def compute_qscore(
    ranked: pd.DataFrame, weights: dict = None, n_buckets: int = 5
) -> dict:
    """Produce a QScore from the rank matrix.

    Steps:
    1. Direction correction (_apply_directions): stability + reinvestment
       are inverted so that high = good for ALL features.
    2. In-group mean per macro-factor (factor score). This prevents
       'accidental weighting' — stability has 6 features but, as a single
       factor, gets 25% weight, not 6/18.
    3. Sum factors with weights. weights=None -> EQUAL weights (baseline,
       25% per factor). Default is equal to avoid confirmation bias.
    4. Min-Max scale to 0-100.
    5. Quintile buckets (n_buckets=5): Q1=highest (premium), Q5=lowest.

    ranked: the 'scaled' output of prepare_matrix (rank space, 0-1).
    weights: {factor: weight} or None (equal). Must sum to 1.

    Returns (dict):
    - qscore: pd.Series (ticker -> 0-100 score, high=good), sorted
    - factor_scores: pd.DataFrame (ticker × 4 factors, in-group means)
    - buckets: pd.Series (ticker -> Q1..Q5, Q1=premium)
    - weights: weights used
    """
    directed = _apply_directions(ranked)

    factor_scores = pd.DataFrame(index=directed.index)
    for group, cols in FACTOR_GROUPS.items():
        existing_cols = [c for c in cols if c in directed.columns]
        factor_scores[group] = directed[existing_cols].mean(axis=1)

    if weights is None:
        weights = {g: 1.0 / len(FACTOR_GROUPS) for g in FACTOR_GROUPS}
    raw = sum(factor_scores[g] * w for g, w in weights.items())

    qscore = 100 * (raw - raw.min()) / (raw.max() - raw.min())
    qscore = qscore.sort_values(ascending=False)
    qscore.name = "qscore"

    # pd.qcut assigns labels from low to high; since we want Q1=premium we
    # pass the labels reversed (lowest bin -> Q_n, highest bin -> Q1).
    bucket_labels = [f"Q{n_buckets - i}" for i in range(n_buckets)]
    buckets = pd.qcut(qscore, q=n_buckets, labels=bucket_labels)
    buckets.name = "bucket"

    logger.info(
        f"QScore: {len(qscore)} companies, {n_buckets} buckets. "
        f"Weights: {weights}"
    )
    logger.info(f"Bucket distribution:\n{buckets.value_counts().sort_index()}")

    return {
        "qscore": qscore,
        "factor_scores": factor_scores,
        "buckets": buckets,
        "weights": weights,
    }
