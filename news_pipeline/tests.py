from unittest import TestCase
from unittest.mock import patch

import requests

from news_pipeline.sentiment_prediction import (
    analyze_sentiment_with_finbert,
    generate_mistral_market_summary,
    get_active_portfolio_tickers,
    run_local_news_prediction_pipeline,
)


class SentimentPredictionTests(TestCase):
    def test_get_active_portfolio_tickers_placeholder(self):
        tickers = get_active_portfolio_tickers()
        self.assertEqual(tickers, ["RELIANCE.NS", "TCS.NS"])

    @patch("news_pipeline.sentiment_prediction._get_finbert_pipeline")
    def test_finbert_output_mapped_to_probability_buckets(self, mock_classifier_factory):
        mock_classifier_factory.return_value = lambda texts: [
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.7},
            {"label": "neutral", "score": 0.6},
        ]
        headlines = [
            {"ticker": "RELIANCE.NS", "headline": "Reliance announces expansion"},
            {"ticker": "TCS.NS", "headline": "TCS misses quarterly guidance"},
            {"ticker": "TCS.NS", "headline": "Markets remain range-bound"},
        ]

        result = analyze_sentiment_with_finbert(headlines)
        aggregate = result["aggregate"]

        self.assertEqual(aggregate["headline_count"], 3)
        self.assertAlmostEqual(aggregate["bullish"], 0.2667, places=4)
        self.assertAlmostEqual(aggregate["bearish"], 0.2333, places=4)
        self.assertAlmostEqual(aggregate["neutral"], 0.2, places=4)

    @patch("news_pipeline.sentiment_prediction.requests.post")
    def test_mistral_summary_timeout_falls_back_gracefully(self, mock_post):
        mock_post.side_effect = requests.Timeout()
        summary = generate_mistral_market_summary(
            headlines=[{"ticker": "RELIANCE.NS", "headline": "Sample headline"}],
            aggregate_sentiment={"bullish": 0.5, "bearish": 0.2, "neutral": 0.3},
            timeout=1,
        )
        self.assertIn("temporarily unavailable", summary)

    @patch("news_pipeline.sentiment_prediction.generate_mistral_market_summary")
    @patch("news_pipeline.sentiment_prediction.analyze_sentiment_with_finbert")
    @patch("news_pipeline.sentiment_prediction.scrape_latest_financial_headlines")
    @patch("news_pipeline.sentiment_prediction.get_active_portfolio_tickers")
    def test_pipeline_returns_combined_payload(
        self,
        mock_tickers,
        mock_scrape,
        mock_sentiment,
        mock_summary,
    ):
        mock_tickers.return_value = ["RELIANCE.NS"]
        mock_scrape.return_value = [{"ticker": "RELIANCE.NS", "headline": "News"}]
        mock_sentiment.return_value = {
            "aggregate": {"bullish": 0.5, "bearish": 0.2, "neutral": 0.3, "headline_count": 1},
            "per_headline": [],
        }
        mock_summary.return_value = "Two sentence summary."

        payload = run_local_news_prediction_pipeline()
        self.assertEqual(payload["tickers"], ["RELIANCE.NS"])
        self.assertIn("generated_at", payload)
        self.assertEqual(payload["market_summary"], "Two sentence summary.")
