"""Ham feature matrisini clustering'e hazırlama: R&D düşür, NaN eşiği, rank dönüşümü.

Sadece FEATURE matrisini işler. Meta matrisi (r2, n_valid) buraya
girmez — leakage ayrımı fiziksel korunur.
"""

import logging

import pandas as pd

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
    3. Evren-geneli rank dönüşümü (pd.DataFrame.rank, method="average",
       pct=True): her feature evren-içi yüzdelik sırasına [0,1] çevrilir.
       RobustScaler bile yapısal uçları (MCK gibi z~31) ehlileştiremedi; rank
       uçları kökten çözer (en yüksek = 1.0, en düşük = 0.0), hiçbir şirketi
       atmaz, veri uydurmaz (sıra gerçek bilgidir). MCK en yüksek ROIC'li
       kalır (rank=1.0) ama ezici etkisi gider — clustering mesafeyi sıra
       üzerinden ölçer, mutlak büyüklük üzerinden değil.

       NÜANS — RANK EVRENE GÖRELİDİR: yeni şirket eklenince TÜM evren yeniden
       rank'lenmeli. Stateless: RobustScaler/StandardScaler'ın "fit ettim,
       sonra transform ederim" davranışı YOK; bu yüzden return'deki "scaler"
       None.

    SADECE feature matrisini işler — meta (r2, n_valid) buraya GİRMEZ
    (leakage ayrımı korunur).

    Döndürür (dict):
    - scaled: pd.DataFrame, kalan şirketler × kalan feature, rank [0,1]
    - scaler: None (rank stateless; yeni veri için tüm evren yeniden rank'lenir)
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

    # rank dönüşümü: her feature evren-içi yüzdelik sıra [0,1]
    # method="average" (eşit değerlere ortalama sıra), pct=True (0-1 normalize)
    scaled = kept_df.rank(method="average", pct=True)

    return {
        "scaled": scaled,
        "scaler": None,
        "kept": kept,
        "dropped": dropped,
        "feature_names": list(kept_df.columns),
    }
