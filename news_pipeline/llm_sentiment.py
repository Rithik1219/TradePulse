"""
news_pipeline/llm_sentiment.py
===============================
LLM-powered sentiment analysis engine for TradePulse — Phase 3.

Responsibilities
----------------
1. Connect to the Google Gemini API via the official ``google-genai`` SDK
   using an API key loaded from the ``.env`` file.
2. Send each news headline to the LLM with a strict prompt that forces a
   structured JSON response containing a **sentiment score** (float,
   −1.0 to 1.0) and an **urgency** label (``"low"`` / ``"medium"`` /
   ``"high"``).
3. Parse the LLM response, validate it, and compile results into a
   clean ``pandas.DataFrame``.

Environment Variables Required
------------------------------
``GEMINI_API_KEY`` — Google Gemini API key.

Usage
-----
::

    from news_pipeline import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_news(news_items)
    print(df)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gemini-2.0-flash"

_SYSTEM_PROMPT = (
    "You are a financial sentiment analysis engine. "
    "You will receive a single stock-market news headline. "
    "Analyze the headline and respond with ONLY a valid JSON object — "
    "no markdown, no explanation, no extra text.\n\n"
    "The JSON object MUST have exactly two keys:\n"
    '  "sentiment_score": a float between -1.0 (highly bearish) '
    "and 1.0 (highly bullish).\n"
    '  "urgency": a string, one of "low", "medium", or "high".\n\n'
    "Example response:\n"
    '{"sentiment_score": 0.65, "urgency": "medium"}\n'
)

_VALID_URGENCY_VALUES = {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class SentimentAnalyzer:
    """Extracts structured sentiment scores from news headlines via an LLM.

    Parameters
    ----------
    env_path : str or None, optional
        Explicit path to the ``.env`` file.  When *None* (the default)
        ``python-dotenv`` searches upward from the working directory.
    model_name : str, optional
        Gemini model identifier.  Defaults to ``"gemini-2.0-flash"``.

    Attributes
    ----------
    model_name : str
        The Gemini model used for inference.
    """

    def __init__(
        self,
        env_path: Optional[str] = None,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        load_dotenv(dotenv_path=env_path)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Missing required environment variable: GEMINI_API_KEY. "
                "Please set it in your .env file."
            )

        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract the first JSON object from *text*.

        Handles cases where the LLM wraps the JSON in markdown fences
        or adds surrounding prose.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = cleaned.strip()

        # Attempt direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fall back: find the first { … } block
        match = re.search(r"\{.*?}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _validate_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalise the parsed LLM output.

        Returns a dict with guaranteed ``sentiment_score`` (float,
        clamped to [−1, 1]) and ``urgency`` (one of low/medium/high).
        """
        score = float(result.get("sentiment_score", 0.0))
        score = max(-1.0, min(1.0, score))

        urgency = str(result.get("urgency", "low")).lower().strip()
        if urgency not in _VALID_URGENCY_VALUES:
            urgency = "low"

        return {"sentiment_score": score, "urgency": urgency}

    def _score_headline(self, headline: str) -> Dict[str, Any]:
        """Send a single headline to the LLM and return the parsed score.

        Returns default-neutral values when the API call fails or the
        response cannot be parsed so the pipeline is never interrupted.
        """
        default = {"sentiment_score": 0.0, "urgency": "low"}

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=f"Headline: {headline}",
                config={
                    "system_instruction": _SYSTEM_PROMPT,
                    "temperature": 0.0,
                },
            )
            raw_text = response.text or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed for headline '%s': %s", headline, exc)
            return default

        parsed = self._extract_json(raw_text)
        if parsed is None:
            logger.warning(
                "Could not parse JSON from LLM response: %s", raw_text
            )
            return default

        return self._validate_result(parsed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_news(self, news_items: List[Dict[str, str]]) -> pd.DataFrame:
        """Analyse a list of scraped news items and return a scored DataFrame.

        Parameters
        ----------
        news_items : list[dict]
            Each dictionary must contain at least a ``"headline"`` key.
            Typically produced by :py:meth:`NewsScraper.fetch_news`.

        Returns
        -------
        pd.DataFrame
            Original news fields augmented with ``sentiment_score``
            (float) and ``urgency`` (str) columns.

        Raises
        ------
        ValueError
            If *news_items* is empty or not a list.
        """
        if not isinstance(news_items, list) or not news_items:
            raise ValueError("news_items must be a non-empty list of dicts.")

        scored_rows: List[Dict[str, Any]] = []
        for item in news_items:
            headline = item.get("headline", "")
            logger.info("Scoring headline: %s", headline[:80])

            sentiment = self._score_headline(headline)
            row = {**item, **sentiment}
            scored_rows.append(row)

        df = pd.DataFrame(scored_rows)

        logger.info(
            "Sentiment analysis complete — %d headline(s) scored.", len(df)
        )
        return df
