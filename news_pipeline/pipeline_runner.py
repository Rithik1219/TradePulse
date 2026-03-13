"""
news_pipeline/pipeline_runner.py
================================
Main execution script for the TradePulse news sentiment pipeline — Phase 3.

Responsibilities
----------------
1. Define a list of stock tickers (hardcoded for demonstration).
2. Invoke :class:`NewsScraper` to fetch the latest headlines for each ticker.
3. Pass the scraped results through :class:`SentimentAnalyzer` to obtain
   structured sentiment scores via the Gemini LLM.
4. Print the final ``pandas.DataFrame`` to the console.

Usage
-----
::

    python -m news_pipeline.pipeline_runner
"""

from __future__ import annotations

import logging
import sys

from news_pipeline.scraper import NewsScraper
from news_pipeline.llm_sentiment import SentimentAnalyzer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS = ["TCS", "RELIANCE"]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the end-to-end news sentiment pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # --- Step 1: Scrape news ------------------------------------------------
    logger.info("Starting news scraper for tickers: %s", TICKERS)
    scraper = NewsScraper()
    news_items = scraper.fetch_news_bulk(TICKERS)

    if not news_items:
        logger.warning("No news items scraped. Exiting.")
        sys.exit(0)

    logger.info("Total headlines scraped: %d", len(news_items))

    # --- Step 2: LLM sentiment analysis -------------------------------------
    logger.info("Running LLM sentiment analysis …")
    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_news(news_items)

    # --- Step 3: Display results --------------------------------------------
    print("\n" + "=" * 80)
    print("TradePulse — News Sentiment Report")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
