"""
ml_core/meta_learner.py
=======================
Stacking meta-learner for TradePulse.

Role in the ensemble
--------------------
The meta-learner sits at the top of the two-tier stacking architecture:

    LSTM  ──────┐
                ├── [p_lstm, p_xgb] ──► MetaLearner ──► final_prob
    XGBoost ────┘

It receives the out-of-fold (OOF) probability predictions produced by the
base learners and learns a combination of those signals that maximises
binary cross-entropy on held-out data.

Why Logistic Regression?
------------------------
• **Interpretable** — the learnt coefficients directly show how much each
  base model contributes to the final prediction.
• **Regularised** — the ``C`` parameter controls L2 regularisation to
  prevent the meta-learner from over-fitting on the limited OOF dataset.
• **Calibrated** — outputs a well-calibrated probability in [0, 1].

Richer meta-features
--------------------
Beyond the raw probabilities ``[p_lstm, p_xgb]`` the meta-learner also
receives:

* **Confidence** signals — ``|p - 0.5| * 2`` maps [0, 1] to [0, 1] where
  0 = completely uncertain and 1 = maximally confident.  This tells the
  stacker *how sure* each base model is, not just *what it predicts*.
* **Disagreement** — ``|p_lstm - p_xgb|`` is high when the two models
  disagree, allowing the meta-learner to down-weight uncertain combinations.
* **Product interaction** — ``p_lstm * p_xgb`` is high only when *both*
  models are bullish, capturing the joint agreement signal.
* **Arithmetic mean** — ``(p_lstm + p_xgb) / 2`` as a simple baseline
  average that the LR can optionally recover as a special case.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Probability midpoint and scale factor for the confidence signal
_PROB_MIDPOINT: float = 0.5
_CONFIDENCE_SCALE: float = 2.0


class MetaLearner:
    """Logistic-regression stacker with enriched meta-features.

    Parameters
    ----------
    C : float
        Inverse of regularisation strength.  Smaller values enforce
        stronger regularisation.  Default 1.0.
    max_iter : int
        Maximum number of solver iterations.  Default 1000.
    random_state : int
        Seed for reproducibility.

    Attributes
    ----------
    model_ : LogisticRegression
        The fitted sklearn LogisticRegression estimator.
    scaler_ : StandardScaler
        Scaler applied to the meta-feature matrix before fitting.
        Helps the solver converge faster.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        if C <= 0:
            raise ValueError(f"C must be positive. Got {C}.")
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

        self.model_: Optional[LogisticRegression] = None
        self.scaler_: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        lstm_probs: np.ndarray,
        xgb_probs: np.ndarray,
        y: np.ndarray,
    ) -> "MetaLearner":
        """Train the meta-learner on base-model probability outputs.

        Parameters
        ----------
        lstm_probs : np.ndarray, shape (n_samples,)
            Out-of-fold probability predictions from ``LSTMModel``.
        xgb_probs : np.ndarray, shape (n_samples,)
            Out-of-fold probability predictions from ``XGBEngine``.
        y : np.ndarray, shape (n_samples,)
            True binary labels {0, 1}.

        Returns
        -------
        self
        """
        meta_X = self._build_meta_features(lstm_probs, xgb_probs)

        # Scale the probability inputs so the solver converges reliably
        self.scaler_ = StandardScaler()
        meta_X_scaled = self.scaler_.fit_transform(meta_X)

        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=self.random_state,
        )
        self.model_.fit(meta_X_scaled, y)

        coef = self.model_.coef_[0]
        feature_names = [
            "p_lstm", "p_xgb", "conf_lstm", "conf_xgb",
            "disagreement", "product", "mean",
        ]
        coef_str = "  ".join(
            f"{name}: {val:.4f}" for name, val in zip(feature_names, coef)
        )
        logger.info("MetaLearner fitted. Coefficients — %s", coef_str)
        return self

    def predict_proba(
        self,
        lstm_probs: np.ndarray,
        xgb_probs: np.ndarray,
    ) -> np.ndarray:
        """Return the final directional probability for each sample.

        Parameters
        ----------
        lstm_probs : np.ndarray, shape (n_samples,)
        xgb_probs  : np.ndarray, shape (n_samples,)

        Returns
        -------
        np.ndarray, shape (n_samples,)  — values in [0, 1]
            Probability that the asset will move UP.
        """
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("MetaLearner has not been fitted yet.")

        meta_X = self._build_meta_features(lstm_probs, xgb_probs)
        meta_X_scaled = self.scaler_.transform(meta_X)
        # Column 1 is the probability for class 1 (UP)
        return self.model_.predict_proba(meta_X_scaled)[:, 1]

    def predict(
        self,
        lstm_probs: np.ndarray,
        xgb_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return hard binary predictions (0 = DOWN, 1 = UP).

        Parameters
        ----------
        lstm_probs : np.ndarray, shape (n_samples,)
        xgb_probs  : np.ndarray, shape (n_samples,)
        threshold  : float
            Decision threshold.  Default 0.5.

        Returns
        -------
        np.ndarray, shape (n_samples,)  — dtype int, values in {0, 1}
        """
        probs = self.predict_proba(lstm_probs, xgb_probs)
        return (probs >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_meta_features(
        lstm_probs: np.ndarray,
        xgb_probs: np.ndarray,
    ) -> np.ndarray:
        """Build a rich (n_samples, 7) meta-feature matrix.

        Columns
        -------
        0  p_lstm         — raw LSTM probability
        1  p_xgb          — raw XGBoost probability
        2  conf_lstm       — LSTM confidence: |p_lstm - 0.5| * 2  ∈ [0, 1]
        3  conf_xgb        — XGBoost confidence: |p_xgb - 0.5| * 2  ∈ [0, 1]
        4  disagreement    — |p_lstm - p_xgb|  (high = models disagree)
        5  product         — p_lstm * p_xgb  (high only when both are bullish)
        6  mean            — (p_lstm + p_xgb) / 2  (simple ensemble average)
        """
        lstm_probs = np.asarray(lstm_probs, dtype=np.float64).reshape(-1)
        xgb_probs = np.asarray(xgb_probs, dtype=np.float64).reshape(-1)
        if len(lstm_probs) != len(xgb_probs):
            raise ValueError(
                "lstm_probs and xgb_probs must have the same length. "
                f"Got {len(lstm_probs)} vs {len(xgb_probs)}."
            )
        conf_lstm = np.abs(lstm_probs - _PROB_MIDPOINT) * _CONFIDENCE_SCALE
        conf_xgb = np.abs(xgb_probs - _PROB_MIDPOINT) * _CONFIDENCE_SCALE
        disagreement = np.abs(lstm_probs - xgb_probs)
        product = lstm_probs * xgb_probs
        mean = (lstm_probs + xgb_probs) / 2.0

        return np.column_stack(
            [lstm_probs, xgb_probs, conf_lstm, conf_xgb, disagreement, product, mean]
        )
