"""
ml_core/xgb_engine.py
=====================
XGBoost tabular classifier for TradePulse.

Responsibilities
----------------
• Accept the PCA-compressed feature matrix produced by
  ``FeaturePreprocessor.transform()``.
• Train a gradient-boosted tree ensemble calibrated for binary
  classification with built-in over-fitting controls:
    - ``max_depth=3``       : shallow trees reduce model complexity.
    - ``subsample=0.8``     : row sub-sampling (stochastic gradient boosting).
    - ``colsample_bytree``  : column sub-sampling per tree.
    - ``min_child_weight``  : minimum leaf-node sample weight.
    - ``early_stopping_rounds`` : halts training when eval metric stops
                                  improving on the validation set.
• Output a probability score in [0, 1] via ``predict_proba()``.

The ``XGBEngine`` class follows the same fit / predict_proba conventions
as ``LSTMModel`` so both base learners can be fed directly into the
``MetaLearner``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class XGBEngine:
    """XGBoost binary classifier wrapper with early-stopping support.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.  Shallow trees (default 3) generalise better
        on noisy financial features.
    n_estimators : int
        Maximum number of boosting rounds (trees).  Early stopping may
        reduce the actual number used.
    learning_rate : float
        Step-size shrinkage (η).  Lower values require more trees but
        generalise better.
    subsample : float
        Fraction of training rows sampled per tree.
    colsample_bytree : float
        Fraction of features sampled per tree.
    min_child_weight : int or float
        Minimum sum of instance weights required in a leaf.  Higher
        values prevent learning on rare noise patterns.
    reg_alpha : float
        L1 regularisation term on leaf weights.
    reg_lambda : float
        L2 regularisation term on leaf weights.
    early_stopping_rounds : int
        Stop training if the validation metric does not improve for this
        many consecutive rounds.
    eval_metric : str
        Evaluation metric used for early stopping (default ``"logloss"``).
    use_gpu : bool
        If ``True``, requests the ``"cuda"`` device tree-method.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        max_depth: int = 3,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 5.0,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 30,
        eval_metric: str = "logloss",
        use_gpu: bool = False,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.use_gpu = use_gpu
        self.random_state = random_state

        self.model_: Optional[XGBClassifier] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> XGBClassifier:
        kwargs: dict = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective="binary:logistic",
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            verbosity=0,
        )
        if self.use_gpu:
            kwargs["device"] = "cuda"
        return XGBClassifier(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBEngine":
        """Train the XGBoost model.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_pca_components)
            PCA-compressed training features.
        y_train : np.ndarray, shape (n_samples,)
            Binary labels {0, 1}.
        X_val : np.ndarray or None
            Validation features for early stopping.  When ``None``, a
            10 % hold-out is carved from *X_train* automatically.
        y_val : np.ndarray or None
            Validation labels (required when *X_val* is provided).

        Returns
        -------
        self
        """
        self.model_ = self._build_model()

        # If no explicit validation set provided, carve out 10 %
        if X_val is None or y_val is None:
            n_val = max(1, int(0.1 * len(X_train)))
            X_val = X_train[-n_val:]
            y_val = y_train[-n_val:]
            X_train = X_train[:-n_val]
            y_train = y_train[:-n_val]
            logger.info(
                "No validation set provided — using last %d samples as "
                "validation set for XGBoost early stopping.",
                n_val,
            )

        eval_set = [(X_val, y_val)]
        self.model_.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        best_round = getattr(self.model_, "best_iteration", None)
        logger.info(
            "XGBEngine fitted. Best iteration: %s",
            best_round if best_round is not None else "N/A",
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the UP class for each sample.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_pca_components)

        Returns
        -------
        np.ndarray, shape (n_samples,)  — values in [0, 1]
        """
        if self.model_ is None:
            raise RuntimeError("XGBEngine has not been fitted yet.")
        # XGBClassifier.predict_proba returns (n_samples, 2); we want
        # the probability for class 1 (UP).
        return self.model_.predict_proba(X)[:, 1]
