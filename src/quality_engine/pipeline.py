"""Universe -> feature/meta matrix orchestration.

The single-company flow (provider -> CompanyFinancials -> extract_features)
exists in parts; this module runs it across the universe and produces TWO
separate matrices: feature_df for clustering and meta_df for audit. The
separation is physical (leakage safeguard).
"""

import logging
import os
import time

import pandas as pd

from .data.base import DataProvider
from .metrics.features import extract_features

logger = logging.getLogger(__name__)


def build_feature_matrix(
    tickers: list[str],
    provider: DataProvider,
    delay: float = 0.0,
    save_dir: str | None = None,
) -> tuple:
    """Build feature and meta matrices for the given ticker list.

    Returns two DataFrames (data leakage separation preserved):
    - feature_df: rows=ticker, columns=21 clustering features (clustering uses this)
    - meta_df: rows=ticker, columns=14 meta (trend_r2, n_valid — audit/filter)

    Resilient: if a company fails (yfinance error, empty data) it is logged,
    skipped, and the loop continues (one company does not break the whole
    matrix).

    delay: wait between calls (seconds). 0 for small tests; ~0.2 is
    recommended for the full universe (~500 companies) to avoid rate limits.

    If save_dir is provided, the matrices are saved as feature_matrix.csv
    and meta_matrix.csv (index=ticker).
    """
    features_rows = []
    meta_rows = []
    basarisiz = []

    for ticker in tickers:
        try:
            cf = provider.get_financials(ticker)
            if not cf.period_end_dates:
                logger.warning(f"{ticker}: empty data, skipping")
                basarisiz.append(ticker)
                continue
            result = extract_features(cf)
            result["features"]["ticker"] = ticker
            result["meta"]["ticker"] = ticker
            features_rows.append(result["features"])
            meta_rows.append(result["meta"])
        except Exception as e:
            logger.warning(f"{ticker}: error ({type(e).__name__}: {e}), skipping")
            basarisiz.append(ticker)
        if delay > 0:
            time.sleep(delay)

    feature_df = pd.DataFrame(features_rows).set_index("ticker")
    meta_df = pd.DataFrame(meta_rows).set_index("ticker")

    logger.info(
        f"Matrix built: {len(features_rows)} succeeded, "
        f"{len(basarisiz)} failed. Failed: {basarisiz}"
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        feature_df.to_csv(os.path.join(save_dir, "feature_matrix.csv"))
        meta_df.to_csv(os.path.join(save_dir, "meta_matrix.csv"))
        logger.info(f"Matrices saved to: {save_dir}")

    return feature_df, meta_df
