"""Evren -> feature/meta matrisleri orkestrasyonu.

Tek şirket akışı (provider -> CompanyFinancials -> extract_features) zaten
parça parça var; bu modül onu evren üzerinde döndürür ve İKİ ayrı matris
çıkarır: clustering'e giren feature_df ve denetim için meta_df. Ayrım
fizikseldir (leakage önlemi).
"""

import logging
import time

import pandas as pd

from .data.base import DataProvider
from .metrics.features import extract_features

logger = logging.getLogger(__name__)


def build_feature_matrix(
    tickers: list[str], provider: DataProvider, delay: float = 0.0
) -> tuple:
    """Verilen ticker listesi için feature ve meta matrislerini kurar.

    İki DataFrame döndürür (data leakage ayrımı korunur):
    - feature_df: satır=ticker, sütun=21 clustering feature (clustering buraya girer)
    - meta_df: satır=ticker, sütun=14 meta (trend_r2, n_valid — denetim/filtre)

    Resilient: bir şirket çökerse (yfinance hatası, boş veri) loglanır,
    atlanır, döngü devam eder (tek şirket tüm matrisi çökertmez).

    delay: her çağrı arası bekleme (saniye). Küçük testte 0, tüm evrende
    (~500 şirket) rate-limit için 0.2 gibi bir değer önerilir.
    """
    features_rows = []
    meta_rows = []
    basarisiz = []

    for ticker in tickers:
        try:
            cf = provider.get_financials(ticker)
            if not cf.period_end_dates:
                logger.warning(f"{ticker}: boş veri, atlanıyor")
                basarisiz.append(ticker)
                continue
            result = extract_features(cf)
            result["features"]["ticker"] = ticker
            result["meta"]["ticker"] = ticker
            features_rows.append(result["features"])
            meta_rows.append(result["meta"])
        except Exception as e:
            logger.warning(f"{ticker}: hata ({type(e).__name__}: {e}), atlanıyor")
            basarisiz.append(ticker)
        if delay > 0:
            time.sleep(delay)

    feature_df = pd.DataFrame(features_rows).set_index("ticker")
    meta_df = pd.DataFrame(meta_rows).set_index("ticker")

    logger.info(
        f"Matris kuruldu: {len(features_rows)} başarılı, "
        f"{len(basarisiz)} başarısız. Başarısız: {basarisiz}"
    )

    return feature_df, meta_df
