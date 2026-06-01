"""K-means clustering + scores for selecting K.

Two distinct responsibilities: find_optimal_k produces the scores (a human
picks), and cluster fits with the chosen K. Mathematical max != financial
meaning: silhouette typically favors K=2, but that coarse binary split is
not useful — so we do not hard-code the selection.
"""

import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def find_optimal_k(scaled_df: pd.DataFrame, k_min: int = 2, k_max: int = 8) -> dict:
    """Compute clustering scores across K values (does NOT pick K).

    Two scores (paired validation):
    - inertia (Elbow): in-cluster tightness. Always decreases with K; look
      for the "elbow".
    - silhouette: cluster separation (-1..+1). Higher = better separated.

    THIS FUNCTION DOES NOT PICK THE FINAL K. It returns the scores; a human
    reviews them and chooses with financial judgment (mathematical max !=
    financial meaning; e.g. silhouette generally favors K=2 but that coarse
    binary split is not useful).
    """
    sonuc = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_df)
        sonuc[k] = {
            "inertia": km.inertia_,
            "silhouette": silhouette_score(scaled_df, labels),
        }
        logger.info(
            f"K={k}: inertia={km.inertia_:.1f}, "
            f"silhouette={sonuc[k]['silhouette']:.3f}"
        )
    return sonuc


def cluster(scaled_df: pd.DataFrame, k: int) -> dict:
    """Final K-means clustering with the chosen K.

    Returns:
    - labels: pd.Series (ticker -> cluster id)
    - centers: cluster centers (DataFrame, in the scaled space)
    - model: the fitted KMeans (to assign new companies later)
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_df)
    labels_series = pd.Series(labels, index=scaled_df.index, name="cluster")
    centers = pd.DataFrame(km.cluster_centers_, columns=scaled_df.columns)
    logger.info(
        f"Clustering done: K={k}, cluster distribution:\n"
        f"{labels_series.value_counts().sort_index()}"
    )
    return {"labels": labels_series, "centers": centers, "model": km}
