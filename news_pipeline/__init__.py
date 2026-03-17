# news_pipeline package — Real-time news scraping and LLM sentiment engine for TradePulse
from .scraper import NewsScraper

try:
    from .llm_sentiment import SentimentAnalyzer
except Exception:  # pragma: no cover - optional dependency path
    SentimentAnalyzer = None

__all__ = ["NewsScraper", "SentimentAnalyzer"]
