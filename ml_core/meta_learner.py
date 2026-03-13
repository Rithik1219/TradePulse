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
base learners and learns a linear combination of those signals that
maximises binary cross-entropy on held-out data.

Why Logistic Regression?
------------------------
• **Interpretable** — the learnt coefficients directly show how much each
  base model contributes to the final prediction.
• **Regularised** — the ``C`` parameter controls L2 regularisation to
  prevent the meta-learner from over-fitting on the limited OOF dataset.
• **Calibrated** — outputs a well-calibrated probability in [0, 1].
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MetaLearner:
    """Logistic-regression stacker that combines base-model probabilities.

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
        Scaler applied to the meta-feature matrix (probability columns)
        before fitting.  Helps the solver converge faster.
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
        logger.info(
            "MetaLearner fitted. Coefficients — LSTM: %.4f  XGBoost: %.4f",
            coef[0],
            coef[1],
        )
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
        """Stack the two probability vectors into a (n_samples, 2) matrix."""
        lstm_probs = np.asarray(lstm_probs, dtype=np.float64).reshape(-1)
        xgb_probs = np.asarray(xgb_probs, dtype=np.float64).reshape(-1)
        if len(lstm_probs) != len(xgb_probs):
            raise ValueError(
                "lstm_probs and xgb_probs must have the same length. "
                f"Got {len(lstm_probs)} vs {len(xgb_probs)}."
            )
        return np.column_stack([lstm_probs, xgb_probs])
