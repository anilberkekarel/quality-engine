"""Aşama 2: zaman serisi özetleme (clustering feature'larına indirgeme).

calculator.py Aşama 1'in çıktısı (her metrik için zaman serisi) buradan
geçer ve clustering'e girecek özet skalerlere indirilir. Ayrı sorumluluk:
calculator.py finansal mantık, features.py istatistiksel özetleme.
"""

import numpy as np
import scipy.stats


def seri_ozetle(seri: list[float]) -> dict:
    """Bir metriğin zaman serisini clustering için özet feature'lara indirir.

    Clustering feature'ları (ham değer; z-score normalizasyonu M4'te):
      - level_last: son dolu değer (mevcut seviye)
      - trend:      zamana karşı OLS eğimi (yön + hız)
      - stability:  dolu değerlerin population std'si (oynaklık)

    META alanları (denetim/filtre için; clustering mesafe hesabına GİRMEZ —
    R² kaliteyi değil ölçüm uyumunu gösterir, mesafeye sokmak kirletir):
      - trend_r2:   linregress R² (trend ne kadar doğrusal)
      - n_valid:    kaç dolu nokta vardı

    NaN'ler atılır AMA orijinal dönem pozisyonu (x ekseni) korunur — zaman
    aralığı çarpılmasın diye. Örn [nan, 0.43, 0.44, 0.46, 0.47] için
    x=[1,2,3,4], y=[0.43,0.44,0.46,0.47].

    Eşikler: level_last >=1, stability >=2, trend >=3 dolu nokta; aksi NaN.
    """
    arr = np.asarray(seri, dtype="float64")
    valid_mask = ~np.isnan(arr)
    x = np.where(valid_mask)[0]
    y = arr[valid_mask]
    n_valid = len(y)

    level_last = y[-1] if n_valid >= 1 else np.nan
    stability = np.std(y, ddof=0) if n_valid >= 2 else np.nan

    if n_valid >= 3:
        result = scipy.stats.linregress(x, y)
        trend = result.slope
        trend_r2 = result.rvalue ** 2
    else:
        trend = np.nan
        trend_r2 = np.nan

    return {
        "level_last": level_last,
        "trend": trend,
        "stability": stability,
        "trend_r2": trend_r2,
        "n_valid": n_valid,
    }
