"""
news_pipeline/scraper.py
========================
Financial news scraper for TradePulse — Phase 3.

Responsibilities
----------------
1. Accept a stock ticker symbol and scrape the latest headlines from
   Google News RSS feeds (public, no authentication required).
2. Parse the RSS XML response with ``BeautifulSoup4`` and extract each
   item's title, publication date, and source link.
3. Return a clean list of dictionaries ready for downstream sentiment
   analysis.

Usage
-----
::

    from news_pipeline import NewsScraper

    scraper = NewsScraper()
    news = scraper.fetch_news("RELIANCE")
    print(news)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
_DEFAULT_MAX_ITEMS = 5
_REQUEST_TIMEOUT = 15  # seconds
_USER_AGENT = (
    "Mozilla/5.0 (compatible; TradePulse/1.0; "
    "+https://github.com/Rithik1219/TradePulse)"
)


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class NewsScraper:
    """Scrapes recent financial news headlines via Google News RSS.

    Parameters
    ----------
    max_items : int, optional
        Maximum number of headlines to return per ticker.  Defaults to 5.
    timeout : int, optional
        HTTP request timeout in seconds.  Defaults to 15.

    Attributes
    ----------
    max_items : int
        Configured headline limit.
    timeout : int
        Configured request timeout.
    """

    def __init__(
        self,
        max_items: int = _DEFAULT_MAX_ITEMS,
        timeout: int = _REQUEST_TIMEOUT,
    ) -> None:
        if max_items < 1:
            raise ValueError(
                f"max_items must be at least 1. Got {max_items}."
            )
        self.max_items = max_items
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_url(self, ticker: str) -> str:
        """Construct the Google News RSS search URL for a ticker.

        The query appends ``"stock"`` to bias results towards financial
        news rather than general mentions of the company name.
        """
        query = quote_plus(f"{ticker} stock")
        return _GOOGLE_NEWS_RSS_URL.format(query=query)

    @staticmethod
    def _parse_pub_date(raw_date: Optional[str]) -> str:
        """Best-effort parse of the RSS ``pubDate`` field.

        Returns an ISO-8601 string when parsing succeeds, or the raw
        string (stripped) as a fallback.
        """
        if not raw_date:
            return ""
        raw_date = raw_date.strip()
        try:
            dt = datetime.strptime(raw_date, "%a, %d %b %Y %H:%M:%S %Z")
            return dt.isoformat()
        except ValueError:
            return raw_date

    def _parse_feed(self, xml_text: str, ticker: str) -> List[Dict[str, str]]:
        """Parse RSS XML and return a list of headline dictionaries."""
        soup = BeautifulSoup(xml_text, "xml")
        items = soup.find_all("item", limit=self.max_items)

        results: List[Dict[str, str]] = []
        for item in items:
            title_tag = item.find("title")
            pub_date_tag = item.find("pubDate")
            link_tag = item.find("link")

            headline = title_tag.get_text(strip=True) if title_tag else ""
            timestamp = self._parse_pub_date(
                pub_date_tag.get_text(strip=True) if pub_date_tag else None
            )
            link = link_tag.get_text(strip=True) if link_tag else ""

            results.append(
                {
                    "ticker": ticker,
                    "headline": headline,
                    "timestamp": timestamp,
                    "link": link,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_news(self, ticker: str) -> List[Dict[str, str]]:
        """Scrape the latest news headlines for a single stock ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g. ``"RELIANCE"``, ``"TCS"``).

        Returns
        -------
        list[dict]
            Each dictionary contains:

            * ``ticker``    – the queried ticker symbol.
            * ``headline``  – the news headline text.
            * ``timestamp`` – publication date (ISO-8601 when parseable).
            * ``link``      – URL to the full article.

        Raises
        ------
        ValueError
            If *ticker* is empty or not a string.
        requests.RequestException
            On network / HTTP errors (logged as a warning; returns ``[]``
            so the pipeline can continue with partial data).
        """
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("ticker must be a non-empty string.")

        ticker = ticker.strip().upper()
        url = self._build_url(ticker)

        logger.info("Fetching news for ticker '%s' …", ticker)

        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": _USER_AGENT},
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(
                "Failed to fetch news for '%s': %s", ticker, exc
            )
            return []

        news_items = self._parse_feed(response.text, ticker)
        logger.info(
            "Scraped %d headline(s) for '%s'.", len(news_items), ticker
        )
        return news_items

    def fetch_news_bulk(self, tickers: List[str]) -> List[Dict[str, str]]:
        """Scrape news for multiple tickers and concatenate the results.

        Parameters
        ----------
        tickers : list[str]
            Iterable of ticker symbols.

        Returns
        -------
        list[dict]
            Flat list of headline dictionaries across all tickers.
        """
        if not tickers:
            raise ValueError("tickers list must not be empty.")

        all_news: List[Dict[str, str]] = []
        for ticker in tickers:
            all_news.extend(self.fetch_news(ticker))
        return all_news
