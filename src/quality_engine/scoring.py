"""QScore: rank matrisinden quality skoru üretir.

Akış: yön düzeltmesi -> faktör grupları (grup-içi ortalama, accidental
weighting'i önler) -> faktör ağırlıkları (varsayılan EŞİT, confirmation
bias'tan kaçınma) -> 0-100 -> quintile sepetler (Q1=premium).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Yön düzeltmesi gereken feature'lar (yüksek rank = KÖTÜ -> 1-rank ile çevir):
# - tüm stability'ler (yüksek std = oynak = kötü)
# - reinvestment level + trend (capital-light tezi: düşük yatırım = iyi)
INVERT_SUFFIXES = ("_stability",)
INVERT_EXACT = {"reinvestment_rate_level_last", "reinvestment_rate_trend"}

# 4 makro-faktör grubu (feature -> grup). 18 feature.
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
    """Yön düzeltmesi: 'düşük=iyi' feature'ları 1-rank ile çevir.
    Böylece TÜM feature'larda yüksek değer = iyi olur (tutarlı yön).
    """
    df = ranked.copy()
    for col in df.columns:
        if col.endswith(INVERT_SUFFIXES) or col in INVERT_EXACT:
            df[col] = 1.0 - df[col]
    return df


def compute_qscore(
    ranked: pd.DataFrame, weights: dict = None, n_buckets: int = 5
) -> dict:
    """Rank matrisinden QScore üretir.

    Adımlar:
    1. Yön düzeltmesi (_apply_directions): stability + reinvestment ters
       çevrilir, tüm feature'larda yüksek = iyi olur.
    2. Her makro-faktör için grup-içi ortalama (faktör skoru). Bu,
       'accidental weighting'i önler — istikrar 6 feature ama tek faktör
       olarak %25 ağırlık alır, 6/18 değil.
    3. Faktörleri ağırlıkla topla. weights=None -> EŞİT ağırlık (baseline,
       her faktör %25). confirmation bias'tan kaçınmak için varsayılan eşit.
    4. Min-Max ile 0-100'e oturt.
    5. Quintile sepetler (n_buckets=5): Q1=en yüksek (premium), Q5=en düşük.

    ranked: prepare_matrix'in 'scaled' çıktısı (rank uzayı, 0-1).
    weights: {faktör: ağırlık} veya None (eşit). Toplamı 1 olmalı.

    Döndürür (dict):
    - qscore: pd.Series (ticker -> 0-100 skor, yüksek=iyi), sıralı
    - factor_scores: pd.DataFrame (ticker × 4 faktör, grup-içi ortalamalar)
    - buckets: pd.Series (ticker -> Q1..Q5, Q1=premium)
    - weights: kullanılan ağırlıklar
    """
    directed = _apply_directions(ranked)

    factor_scores = pd.DataFrame(index=directed.index)
    for grup, cols in FACTOR_GROUPS.items():
        mevcut = [c for c in cols if c in directed.columns]
        factor_scores[grup] = directed[mevcut].mean(axis=1)

    if weights is None:
        weights = {g: 1.0 / len(FACTOR_GROUPS) for g in FACTOR_GROUPS}
    raw = sum(factor_scores[g] * w for g, w in weights.items())

    qscore = 100 * (raw - raw.min()) / (raw.max() - raw.min())
    qscore = qscore.sort_values(ascending=False)
    qscore.name = "qscore"

    # pd.qcut etiketleri düşükten yükseğe atar; Q1=premium istediğimiz için
    # etiketleri tersten veriyoruz (en düşük bin -> Q_n, en yüksek bin -> Q1).
    bucket_labels = [f"Q{n_buckets - i}" for i in range(n_buckets)]
    buckets = pd.qcut(qscore, q=n_buckets, labels=bucket_labels)
    buckets.name = "bucket"

    logger.info(
        f"QScore: {len(qscore)} şirket, {n_buckets} sepet. "
        f"Ağırlık: {weights}"
    )
    logger.info(f"Sepet dağılımı:\n{buckets.value_counts().sort_index()}")

    return {
        "qscore": qscore,
        "factor_scores": factor_scores,
        "buckets": buckets,
        "weights": weights,
    }
