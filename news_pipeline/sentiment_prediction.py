"""
news_pipeline/sentiment_prediction.py
====================================
Phase 11 local sentiment and prediction pipeline.

Pipeline steps
--------------
1. Retrieve active portfolio tickers (placeholder implementation).
2. Scrape latest financial headlines for those tickers.
3. Score each headline with a local FinBERT model.
4. Ask local Ollama (Mistral) for a concise, two-sentence market summary.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

from news_pipeline.scraper import NewsScraper

logger = logging.getLogger(__name__)

_DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"
_OLLAMA_API_URL = "http://localhost:11434/api/generate"
_OLLAMA_TIMEOUT_SECONDS = 25

_FINBERT_PIPELINE = None


def get_active_portfolio_tickers() -> List[str]:
    """Return active portfolio tickers.

    Placeholder for Angel One SmartAPI integration in production.
    """
    return ["RELIANCE.NS", "TCS.NS"]


def _normalize_ticker_for_news_query(ticker: str) -> str:
    """Normalize exchange-formatted tickers for human/news search usage."""
    return ticker.split(".", maxsplit=1)[0].strip().upper()


def scrape_latest_financial_headlines(
    tickers: List[str],
    max_items_per_ticker: int = 4,
    timeout: int = 10,
) -> List[Dict[str, str]]:
    """Scrape latest financial headlines for the provided portfolio tickers."""
    if not tickers:
        return []

    scraper = NewsScraper(max_items=max_items_per_ticker, timeout=timeout)
    all_headlines: List[Dict[str, str]] = []

    for raw_ticker in tickers:
        query_ticker = _normalize_ticker_for_news_query(raw_ticker)
        scraped_items = scraper.fetch_news(query_ticker)
        for item in scraped_items:
            item["ticker"] = raw_ticker
        all_headlines.extend(scraped_items)

    return all_headlines


def _get_finbert_pipeline():
    """Lazily load the local FinBERT pipeline."""
    global _FINBERT_PIPELINE

    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE

    from transformers import pipeline  # imported lazily to avoid hard import failure

    _FINBERT_PIPELINE = pipeline(
        "text-classification",
        model=_DEFAULT_FINBERT_MODEL,
        tokenizer=_DEFAULT_FINBERT_MODEL,
    )
    return _FINBERT_PIPELINE


def _empty_sentiment_result(headline_count: int) -> Dict[str, Any]:
    return {
        "aggregate": {
            "bullish": 0.0,
            "bearish": 0.0,
            "neutral": 1.0 if headline_count else 0.0,
            "headline_count": headline_count,
        },
        "per_headline": [],
    }


def analyze_sentiment_with_finbert(
    headlines: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Run FinBERT on headlines and return strict bullish/bearish/neutral probabilities."""
    if not headlines:
        return _empty_sentiment_result(headline_count=0)

    texts = [item.get("headline", "").strip() for item in headlines]

    try:
        classifier = _get_finbert_pipeline()
        predictions = classifier(texts)
    except Exception:
        logger.exception("FinBERT inference failed; using neutral fallback.")
        return _empty_sentiment_result(headline_count=len(headlines))

    per_headline: List[Dict[str, Any]] = []
    sums = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
    label_map = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}

    for item, prediction in zip(headlines, predictions, strict=False):
        predicted_label = str(prediction.get("label", "neutral")).lower()
        confidence = float(prediction.get("score", 0.0))

        bucket = label_map.get(predicted_label, "neutral")
        sentiment_vector = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
        sentiment_vector[bucket] = confidence

        sums["bullish"] += sentiment_vector["bullish"]
        sums["bearish"] += sentiment_vector["bearish"]
        sums["neutral"] += sentiment_vector["neutral"]

        per_headline.append(
            {
                "ticker": item.get("ticker", ""),
                "headline": item.get("headline", ""),
                "source_link": item.get("link", ""),
                "published_at": item.get("timestamp", ""),
                "sentiment": sentiment_vector,
                "predicted_label": bucket,
            }
        )

    headline_count = len(per_headline)
    aggregate = {
        "bullish": round(sums["bullish"] / headline_count, 4),
        "bearish": round(sums["bearish"] / headline_count, 4),
        "neutral": round(sums["neutral"] / headline_count, 4),
        "headline_count": headline_count,
    }
    return {"aggregate": aggregate, "per_headline": per_headline}


def generate_mistral_market_summary(
    headlines: List[Dict[str, str]],
    aggregate_sentiment: Dict[str, float],
    timeout: int = _OLLAMA_TIMEOUT_SECONDS,
) -> str:
    """Generate a two-sentence summary using local Mistral (Ollama)."""
    if not headlines:
        return (
            "No recent ticker-specific headlines were found. "
            "The portfolio sentiment remains neutral until new data arrives."
        )

    top_headlines = "\n".join(
        f"- {item.get('ticker', '')}: {item.get('headline', '')}"
        for item in headlines[:8]
    )
    prompt = (
        "You are a financial assistant.\n"
        "Given the headlines and FinBERT aggregate probabilities below, "
        "write exactly 2 concise sentences explaining the expected market bias.\n\n"
        f"Headlines:\n{top_headlines}\n\n"
        f"FinBERT aggregate: {aggregate_sentiment}\n"
    )

    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(_OLLAMA_API_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        summary = str(data.get("response", "")).strip()
        if summary:
            return summary
    except requests.Timeout:
        logger.warning("Ollama summary generation timed out.")
    except requests.RequestException:
        logger.exception("Ollama summary generation request failed.")
    except ValueError:
        logger.exception("Invalid JSON from Ollama summary generation.")

    return (
        "Local summary generation is temporarily unavailable due to timeout or service error. "
        "FinBERT probabilities are provided and can be used directly for dashboard decisions."
    )


def run_local_news_prediction_pipeline() -> Dict[str, Any]:
    """Run the full local ticker-news sentiment and summary pipeline."""
    tickers = get_active_portfolio_tickers()
    headlines = scrape_latest_financial_headlines(tickers=tickers)
    sentiment = analyze_sentiment_with_finbert(headlines=headlines)
    summary = generate_mistral_market_summary(
        headlines=headlines,
        aggregate_sentiment=sentiment["aggregate"],
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "headlines": headlines,
        "sentiment": sentiment,
        "market_summary": summary,
        "meta": {
            "engine": "local_finbert_plus_mistral",
            "ollama_url": _OLLAMA_API_URL,
        },
    }
