"""
ml_core/predictor.py
====================
Inference engine for TradePulse.

Loads the persisted Hybrid Ensemble artifacts (``FeaturePreprocessor``,
``XGBEngine``, ``LSTMModel``, ``MetaLearner``) from ``saved_models/``
and exposes a single ``predict_signal`` method that accepts live OHLCV
data and returns the Meta-Learner's final probability.

Usage
-----
::

    from ml_core.predictor import TradePulsePredictor

    predictor = TradePulsePredictor()
    prob = predictor.predict_signal(historical_ohlcv_df, sentiment_score=0.0)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from ml_core.lstm_engine import LSTMModel
from ml_core.meta_learner import MetaLearner
from ml_core.preprocessing import FeaturePreprocessor
from ml_core.xgb_engine import XGBEngine

logger = logging.getLogger(__name__)

# Default path — repo root / saved_models
_DEFAULT_MODEL_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models",
)


class TradePulsePredictor:
    """Inference wrapper that loads trained artifacts and generates signals.

    Parameters
    ----------
    model_dir : str, optional
        Directory containing the saved artifacts.  Defaults to
        ``<repo_root>/saved_models/``.
    """

    def __init__(self, model_dir: str = _DEFAULT_MODEL_DIR) -> None:
        self.model_dir = model_dir
        logger.info("Loading TradePulse artifacts from %s …", model_dir)

        self.preprocessor: FeaturePreprocessor = FeaturePreprocessor.load(
            os.path.join(model_dir, "preprocessor.joblib")
        )
        self.xgb_engine: XGBEngine = XGBEngine.load(
            os.path.join(model_dir, "xgb_engine.joblib")
        )
        self.lstm_model: LSTMModel = LSTMModel.load(
            os.path.join(model_dir, "lstm_model")
        )
        self.meta_learner: MetaLearner = MetaLearner.load(
            os.path.join(model_dir, "meta_learner.joblib")
        )
        logger.info("All TradePulse artifacts loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_signal(
        self,
        historical_ohlcv_df: pd.DataFrame,
        sentiment_score: float = 0.0,
    ) -> float:
        """Generate an ensemble prediction from live data.

        Parameters
        ----------
        historical_ohlcv_df : pd.DataFrame
            Recent OHLCV data for a single ticker, as returned by
            ``AngelOneClient.get_historical_data()``.  Must contain
            columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
            At least ``lstm_model.seq_len`` rows (default 30) are
            required.
        sentiment_score : float, optional
            Current sentiment score in [-1, 1].  Defaults to 0.0
            (neutral).

        Returns
        -------
        float
            Ensemble probability in [0.0, 1.0] where values above 0.5
            indicate a predicted UP move.
        """
        seq_len: int = self.lstm_model.seq_len

        # ----- Validate required OHLCV columns ----------------------------
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        for col in ohlcv_cols:
            if col not in historical_ohlcv_df.columns:
                raise ValueError(
                    f"Missing required column '{col}' in historical_ohlcv_df."
                )

        df = historical_ohlcv_df[ohlcv_cols].copy()

        # ----- Engineer the same 7 features used during training ----------
        # Training pipeline (yfinance_bulk_ingestion.py) sequences these
        # columns: open, high, low, close, volume, daily_return, volatility.
        # We must replicate that logic here to avoid a training-serving skew.
        df["daily_return"] = df["close"].pct_change()
        df["volatility"] = df["daily_return"].rolling(window=20).std()

        # Drop NaN rows introduced by pct_change() and rolling(window=20)
        df = df.dropna()

        # Ensure we still have enough rows after NaN removal
        if len(df) < seq_len:
            raise ValueError(
                f"Need at least {seq_len} rows of historical data, "
                f"got {len(df)}."
            )

        # Feature order must exactly match the training schema
        lstm_cols = ["open", "high", "low", "close", "volume",
                     "daily_return", "volatility"]

        # Take the most recent `seq_len` rows as a single sample
        seq_data = df[lstm_cols].iloc[-seq_len:].values.astype(np.float32)
        X_seq = seq_data[np.newaxis, :, :]  # (1, seq_len, 7)

        # ----- Build the tabular feature matrix for XGBoost ----------------
        # In production, raw technical indicators would be computed here.
        # For consistency with the training pipeline we generate a
        # deterministic feature vector derived from the available OHLCV data.
        n_raw_features = len(self.preprocessor.feature_names_in_)
        tabular_row = self._derive_tabular_features(df, n_raw_features)
        df_tabular = pd.DataFrame(
            [tabular_row],
            columns=self.preprocessor.feature_names_in_,
        )
        tab_pca = self.preprocessor.transform(df_tabular)

        # ----- Base-learner predictions ------------------------------------
        xgb_prob = self.xgb_engine.predict_proba(tab_pca)
        lstm_prob = self.lstm_model.predict_proba(X_seq)

        # ----- Meta-learner ensemble prediction ----------------------------
        final_prob = self.meta_learner.predict_proba(lstm_prob, xgb_prob)

        # Return a plain Python float
        return float(final_prob[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_tabular_features(
        df: pd.DataFrame,
        n_features: int,
    ) -> np.ndarray:
        """Derive a fixed-length feature vector from OHLCV + sentiment.

        .. note::

            **Placeholder implementation.**  In production this must be
            replaced with the actual technical-indicator computation
            (e.g. RSI, MACD, Bollinger Bands, etc.) that was used during
            training.  The current version generates a deterministic
            pseudo-random vector seeded from the raw data so that the
            inference pipeline can be exercised end-to-end.
        """
        base_values = df.values.astype(np.float64).flatten()
        rng = np.random.default_rng(seed=abs(hash(base_values.tobytes())) % (2**31))
        features = rng.standard_normal(n_features).astype(np.float32)
        return features
