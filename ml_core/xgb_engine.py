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
    - ``max_depth``             : tree depth (default 4 for richer splits).
    - ``subsample``             : row sub-sampling (stochastic gradient boosting).
    - ``colsample_bytree``      : column sub-sampling per tree.
    - ``colsample_bylevel``     : column sub-sampling per tree level.
    - ``min_child_weight``      : minimum leaf-node sample weight.
    - ``gamma``                 : minimum loss reduction to make a split.
    - ``scale_pos_weight``      : class imbalance compensation.
    - ``early_stopping_rounds`` : halts training when eval metric stops
                                  improving on the validation set.
• Output a probability score in [0, 1] via ``predict_proba()``.
• Log the top-10 most important features after fitting.

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
        Maximum tree depth.  Default 4 provides richer feature interactions
        than the original 3 while remaining resistant to over-fitting when
        combined with the other regularisation parameters.
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
    colsample_bylevel : float
        Fraction of features sampled per tree level.  Provides an
        additional layer of randomisation on top of ``colsample_bytree``.
    min_child_weight : int or float
        Minimum sum of instance weights required in a leaf.  Higher
        values prevent learning on rare noise patterns.
    gamma : float
        Minimum loss-reduction required to make a split.  Acts as a
        threshold that pruning must beat, providing explicit complexity
        control.  Default 0.1.
    reg_alpha : float
        L1 regularisation term on leaf weights.
    reg_lambda : float
        L2 regularisation term on leaf weights.
    scale_pos_weight : float
        Ratio of negative to positive class samples.  Set to
        ``(n_neg / n_pos)`` when the dataset is imbalanced to give more
        weight to the minority class.  Default 1.0 (balanced).
    early_stopping_rounds : int
        Stop training if the validation metric does not improve for this
        many consecutive rounds.
    eval_metric : str
        Evaluation metric used for early stopping (default ``"logloss"``).
    n_jobs : int
        Number of parallel threads.  Default ``-1`` uses all available
        CPU cores.
    use_gpu : bool
        If ``True``, requests the ``"cuda"`` device tree-method.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        max_depth: int = 4,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        colsample_bylevel: float = 0.8,
        min_child_weight: float = 5.0,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        early_stopping_rounds: int = 30,
        eval_metric: str = "logloss",
        n_jobs: int = -1,
        use_gpu: bool = False,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
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
            colsample_bylevel=self.colsample_bylevel,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            objective="binary:logistic",
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            n_jobs=self.n_jobs,
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
        X_train : np.ndarray, shape (n_samples, n_features)
            PCA-compressed (or raw) training features.
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

        # If no explicit validation set provided, carve out the last 10 %.
        # For time-series data this is chronologically the most recent
        # observations, which is appropriate.  For randomly-ordered tabular
        # data you should provide an explicit *X_val* / *y_val* split.
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

        # Log top-10 feature importances (gain) when available
        importances = self.model_.feature_importances_
        if importances is not None and len(importances) > 0:
            top_k = min(10, len(importances))
            top_idx = np.argsort(importances)[::-1][:top_k]
            top_vals = importances[top_idx]
            logger.info(
                "Top-%d feature importances (gain): %s",
                top_k,
                ", ".join(f"feat[{i}]={v:.4f}" for i, v in zip(top_idx, top_vals)),
            )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the UP class for each sample.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)  — values in [0, 1]
        """
        if self.model_ is None:
            raise RuntimeError("XGBEngine has not been fitted yet.")
        # XGBClassifier.predict_proba returns (n_samples, 2); we want
        # the probability for class 1 (UP).
        return self.model_.predict_proba(X)[:, 1]
