"""K-means clustering + K seçimi için skorlar.

İki ayrı sorumluluk: find_optimal_k skorları üretir (insan seçer),
cluster nihai K ile fit eder. Matematiksel max ≠ finansal anlam:
silhouette genelde K=2'yi favori eder ama bu kaba ikilik işe yaramaz —
bu yüzden seçimi koda gömmüyoruz.
"""

import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def find_optimal_k(scaled_df: pd.DataFrame, k_min: int = 2, k_max: int = 8) -> dict:
    """Farklı K değerleri için clustering skorlarını hesaplar (K SEÇMEZ).

    İki skor (ikili doğrulama):
    - inertia (Elbow): küme-içi sıkılık. K arttıkça hep düşer; "dirsek" aranır.
    - silhouette: küme ayrışması (-1..+1). Yüksek = iyi ayrışmış.

    NİHAİ K'YI BU FONKSİYON SEÇMEZ. Skorları döndürür, insan inceler ve
    finansal anlamlılıkla seçer (matematiksel max != finansal anlam; örn.
    silhouette genelde K=2'yi favori eder ama bu kaba ikilik işe yaramaz).
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
    """Seçilen K ile final K-means clustering.

    Döndürür:
    - labels: pd.Series (ticker -> küme no)
    - centers: küme merkezleri (DataFrame, z-score uzayında)
    - model: fit edilmiş KMeans (ileride yeni şirket atamak için)
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_df)
    labels_series = pd.Series(labels, index=scaled_df.index, name="cluster")
    centers = pd.DataFrame(km.cluster_centers_, columns=scaled_df.columns)
    logger.info(
        f"Clustering tamam: K={k}, küme dağılımı:\n"
        f"{labels_series.value_counts().sort_index()}"
    )
    return {"labels": labels_series, "centers": centers, "model": km}
