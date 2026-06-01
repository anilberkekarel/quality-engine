"""Engine entry point: reads feature matrix from cache -> prepare -> QScore -> output."""

import sys
import logging

import pandas as pd

from quality_engine.preprocessing import prepare_matrix
from quality_engine.scoring import compute_qscore

logging.basicConfig(level=logging.INFO, format="%(message)s")

CACHE_PATH = "_cache/feature_matrix.csv"


def main():
    try:
        fdf = pd.read_csv(CACHE_PATH, index_col=0)
    except FileNotFoundError:
        print(f"ERROR: {CACHE_PATH} not found. First build and save the matrix "
              f"via the pipeline (build_feature_matrix(..., save_dir='_cache')).")
        sys.exit(1)

    print(f"Feature matrix: {fdf.shape[0]} companies, {fdf.shape[1]} features")

    prep = prepare_matrix(fdf)
    print(f"Prepared: {len(prep['kept'])} companies kept, "
          f"{len(prep['dropped'])} dropped")

    result = compute_qscore(prep["scaled"])
    qscore = result["qscore"]
    buckets = result["buckets"]

    output_df = pd.DataFrame({
        "qscore": qscore.round(1),
        "bucket": buckets,
    })
    print("\n=== QSCORE RANKING (all companies) ===")
    print(output_df.to_string())

    output_df.to_csv("_cache/qscore_output.csv")
    print(f"\nOutput saved: _cache/qscore_output.csv")
    print(f"Q1 (premium): {(buckets == 'Q1').sum()} companies")


if __name__ == "__main__":
    main()
