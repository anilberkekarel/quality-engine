"""S&P 500 universe definition — SEPARATE responsibility from the data provider.

Universe (which companies are candidates) and financial data (yfinance/FMP)
are different layers so that the provider can be swapped while keeping the
universe definition stable.
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
    """Return the current S&P 500 ticker list.

    WARNING — SURVIVORSHIP BIAS: this list contains TODAY's S&P 500 members,
    i.e. the 'survivors'. Companies that were dropped from the index, went
    bankrupt, or were acquired are NOT here. It is valid for current
    candidate screening BUT NOT for historical validation (Tier 2 — "would
    the engine have caught X in 2015?"); that test requires historical
    membership data. Do not use this list in validation.

    Cache: if use_cache=True and the cache file exists, read from it
    (deterministic, reproducible universe — same list within the same run).
    Otherwise scrape from Wikipedia and write the cache. Delete the cache
    file to refresh the list.

    Ticker cleanup: Wikipedia returns dot format like 'BRK.B', 'BF.B';
    yfinance expects hyphens ('BRK-B', 'BF-B'). Dots are converted to
    hyphens.
    """
    if use_cache and CACHE_PATH.exists():
        logger.info("Reading S&P 500 from cache: %s", CACHE_PATH)
        df = pd.read_csv(CACHE_PATH)
        return df["ticker"].tolist()

    logger.info("Fetching S&P 500 from Wikipedia: %s", WIKIPEDIA_SP500_URL)
    # Wikipedia returns 403 to pd.read_html's default User-Agent; we
    # download via requests with a real browser UA and pass the text in.
    headers = {"User-Agent": "Mozilla/5.0 (research/educational use)"}
    response = requests.get(WIKIPEDIA_SP500_URL, headers=headers, timeout=15)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    df = tables[0]
    if "Symbol" not in df.columns:
        raise ValueError(
            f"Wikipedia S&P 500 table schema has changed: 'Symbol' column "
            f"is missing. Current columns: {list(df.columns)}. Stopped to "
            f"avoid silently returning the wrong list — update the parser."
        )

    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(CACHE_PATH, index=False)

    logger.warning(
        "SURVIVORSHIP BIAS: get_sp500_tickers returns TODAY's members; "
        "do not use for historical validation."
    )
    return tickers
