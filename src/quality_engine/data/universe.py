"""S&P 500 evren tanımı — veri sağlayıcıdan AYRI sorumluluk.

Evren (hangi şirketler aday) ile finansal veri (yfinance/FMP) farklı
katmanlardır: aynı evren üzerinde sağlayıcı değiştirilebilsin diye.
"""

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CACHE_PATH = Path(__file__).parent / "_cache" / "sp500_tickers.csv"


def get_sp500_tickers(use_cache: bool = True) -> list[str]:
    """Güncel S&P 500 ticker listesini döndürür.

    UYARI — SURVIVORSHIP BIAS: Bu liste BUGÜNKÜ S&P 500 üyeleridir,
    yani 'hayatta kalanlar'. Endeksten düşen/batan/satın alınan şirketler
    YOK. Mevcut aday taraması için geçerlidir, ANCAK tarihsel validation
    (Katman 2 — 'motor 2015'te X'i yakalar mıydı') için UYGUN DEĞİLDİR;
    o test için tarihsel üyelik verisi gerekir. Bu listeyi validation'da
    kullanma.

    Cache: use_cache=True ve cache dosyası varsa ondan okur (deterministik,
    tekrarlanabilir evren — aynı çalışmada aynı liste). Yoksa Wikipedia'dan
    çeker ve cache'e yazar. Listeyi yenilemek için cache dosyasını sil.

    Ticker temizliği: Wikipedia 'BRK.B', 'BF.B' gibi nokta formatı verir;
    yfinance tire ister ('BRK-B', 'BF-B'). Nokta -> tire çevrilir.
    """
    if use_cache and CACHE_PATH.exists():
        logger.info("S&P 500 cache'ten okunuyor: %s", CACHE_PATH)
        df = pd.read_csv(CACHE_PATH)
        return df["ticker"].tolist()

    logger.info("S&P 500 Wikipedia'dan çekiliyor: %s", WIKIPEDIA_SP500_URL)
    # Wikipedia pd.read_html'in default User-Agent'ına 403 dönüyor; gerçek
    # tarayıcı UA göndermek için requests üzerinden indirip metni geçiriyoruz.
    headers = {"User-Agent": "Mozilla/5.0 (research/educational use)"}
    response = requests.get(WIKIPEDIA_SP500_URL, headers=headers, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    df = tables[0]
    if "Symbol" not in df.columns:
        raise ValueError(
            f"Wikipedia S&P 500 tablo şeması değişmiş: 'Symbol' sütunu yok. "
            f"Mevcut sütunlar: {list(df.columns)}. Sessiz yanlış liste "
            f"döndürmemek için durduruldu — parser'ı güncelle."
        )

    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(CACHE_PATH, index=False)

    logger.warning(
        "SURVIVORSHIP BIAS: get_sp500_tickers BUGÜNKÜ üyeleri döndürür; "
        "tarihsel validation için kullanma."
    )
    return tickers
