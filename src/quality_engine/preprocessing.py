"""Ham feature matrisini clustering'e hazırlama: R&D düşür, NaN eşiği, z-score.

Sadece FEATURE matrisini işler. Meta matrisi (r2, n_valid) buraya
girmez — leakage ayrımı fiziksel korunur.
"""

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

RND_PREFIX = "rnd_to_revenue"


def prepare_matrix(feature_df: pd.DataFrame, max_nan: int = 0) -> dict:
    """Ham feature matrisini clustering'e hazırlar.

    Adımlar:
    1. R&D feature'larını düşür (rnd_to_revenue_*): evrenin ~%65'inde NaN,
       genel clustering için bilgi taşımıyor (sektör-koşullu; tech-alt-evrende
       ileride geri gelebilir).
    2. Eksik-veri eşiği: satır başına NaN sayısı > max_nan olan şirketleri at.
       max_nan=0 (varsayılan) = sadece tam-temiz şirketler kalır (imputation
       gereksiz, K-means NaN kabul etmez).
    3. Evren-geneli z-score (sklearn StandardScaler): her feature ortalama 0,
       std 1. Clustering ölçek-duyarlı, bu zorunlu.

    SADECE feature matrisini işler — meta (r2, n_valid) buraya GİRMEZ
    (leakage ayrımı korunur).

    Döndürür (dict):
    - scaled: pd.DataFrame, kalan şirketler × kalan feature, z-score'lu
    - scaler: fit edilmiş StandardScaler (ileride yeni veri için aynı ölçek)
    - kept: kalan ticker listesi
    - dropped: elenen ticker listesi (denetim/geri-kazanım için)
    - feature_names: kalan feature isimleri (sıra önemli)
    """
    rnd_cols = [c for c in feature_df.columns if c.startswith(RND_PREFIX)]
    df = feature_df.drop(columns=rnd_cols)

    nan_per_row = df.isna().sum(axis=1)
    keep_mask = nan_per_row <= max_nan
    kept_df = df[keep_mask]
    kept = list(kept_df.index)
    dropped = list(df[~keep_mask].index)

    logger.info(
        f"Filtre: {len(kept)} tutuldu, {len(dropped)} elendi "
        f"(R&D düşürüldü, max_nan={max_nan}). Elenen: {dropped}"
    )

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(kept_df)
    scaled = pd.DataFrame(scaled_values, index=kept_df.index, columns=kept_df.columns)

    return {
        "scaled": scaled,
        "scaler": scaler,
        "kept": kept,
        "dropped": dropped,
        "feature_names": list(kept_df.columns),
    }
