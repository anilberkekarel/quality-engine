"""Engine entry point: fetch (or load cached) feature matrix -> prepare -> QScore -> output."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import argparse
import os
import random
import logging

import pandas as pd

from quality_engine.data.universe import get_sp500_tickers
from quality_engine.data.yfinance_provider import YFinanceProvider
from quality_engine.pipeline import build_feature_matrix
from quality_engine.preprocessing import prepare_matrix
from quality_engine.scoring import compute_qscore

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Quality Pattern Engine — screen S&P 500 companies for "
                    "sustainable quality and rank them by QScore."
    )
    parser.add_argument("--companies", type=int, default=200,
        help="Number of companies to analyze (random sample from S&P 500, "
             "max ~503). Default: 200.")
    parser.add_argument("--delay", type=float, default=0.3,
        help="Seconds between data fetches (rate-limit protection for the "
             "free yfinance source). Lower is faster but risks throttling; "
             "values below 0.2 are not recommended. Default: 0.3.")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducible company sampling. Default: 42.")
    args = parser.parse_args()

    # 1. universe + sampling
    all_tickers = get_sp500_tickers()
    n = min(args.companies, len(all_tickers))   # 503 cap
    if args.companies > len(all_tickers):
        print(f"Note: S&P 500 has {len(all_tickers)} companies; "
              f"capping at {len(all_tickers)}.")
    random.seed(args.seed)
    sample = random.sample(all_tickers, n)

    # 2. cache — embed company count in filename (avoid collisions)
    cache_path = f"_cache/feature_matrix_{n}.csv"
    if os.path.exists(cache_path):
        print(f"Loading cached matrix: {cache_path}")
        fdf = pd.read_csv(cache_path, index_col=0)
    else:
        print(f"Fetching {n} companies (delay={args.delay}s)... "
              f"this may take several minutes.")
        fdf, _ = build_feature_matrix(sample, YFinanceProvider(),
                                      delay=args.delay)
        os.makedirs("_cache", exist_ok=True)
        fdf.to_csv(cache_path)
        print(f"Saved: {cache_path}")

    print(f"Feature matrix: {fdf.shape[0]} companies, {fdf.shape[1]} features")

    # 3. prepare + score
    prep = prepare_matrix(fdf)
    print(f"Prepared: {len(prep['kept'])} kept, {len(prep['dropped'])} dropped")
    result = compute_qscore(prep["scaled"])
    qscore = result["qscore"]
    buckets = result["buckets"]

    # 4. output
    output_df = pd.DataFrame({"qscore": qscore.round(1), "bucket": buckets})
    print("\n=== QSCORE RANKING (all companies) ===")
    print(output_df.to_string())
    output_df.to_csv(f"_cache/qscore_output_{n}.csv")
    print(f"\nOutput saved: _cache/qscore_output_{n}.csv")
    print(f"Q1 (premium): {(buckets == 'Q1').sum()} companies")


if __name__ == "__main__":
    main()
