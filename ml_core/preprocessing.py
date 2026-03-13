"""
ml_core/preprocessing.py
========================
Feature preprocessing pipeline for TradePulse.

Responsibilities
----------------
1. Accept a high-dimensional Pandas DataFrame containing raw technical
   indicators and sentiment scores (potentially thousands of columns).
2. Robustly scale every feature with ``RobustScaler`` so that outliers —
   common in financial time-series — do not distort the learned
   transformation.
3. Compress the scaled features with ``PCA`` retaining enough principal
   components to explain 95 % of the total variance.

The resulting ``FeaturePreprocessor`` class follows the standard
scikit-learn ``fit`` / ``transform`` / ``fit_transform`` contract so it
can be embedded inside ``sklearn.pipeline.Pipeline`` or used standalone.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Scaling + PCA dimensionality-reduction pipeline.

    Parameters
    ----------
    pca_variance_threshold : float, optional
        Fraction of total variance to retain via PCA.  Defaults to 0.95
        (i.e. 95 %).  Must be in the range (0, 1].
    random_state : int, optional
        Seed passed to ``PCA`` for reproducibility.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        The fitted scikit-learn pipeline (``RobustScaler`` → ``PCA``).
    n_components_ : int
        Number of principal components selected after fitting.
    feature_names_in_ : list[str]
        Column names of the DataFrame seen during ``fit``.
    """

    def __init__(
        self,
        pca_variance_threshold: float = 0.95,
        random_state: int = 42,
    ) -> None:
        if not (0 < pca_variance_threshold <= 1.0):
            raise ValueError(
                "pca_variance_threshold must be in the range (0, 1]. "
                f"Got {pca_variance_threshold}."
            )
        self.pca_variance_threshold = pca_variance_threshold
        self.random_state = random_state

        # Will be populated in fit()
        self.pipeline_: Optional[Pipeline] = None
        self.n_components_: Optional[int] = None
        self.feature_names_in_: list[str] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        """Construct the sklearn Pipeline (not yet fitted)."""
        scaler = RobustScaler()
        # n_components as a float instructs PCA to select the minimum
        # number of components that explain the requested variance ratio.
        pca = PCA(
            n_components=self.pca_variance_threshold,
            random_state=self.random_state,
        )
        return Pipeline(steps=[("scaler", scaler), ("pca", pca)])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "FeaturePreprocessor":
        """Fit the scaler and PCA on *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature matrix.  All columns must be numeric.
        y : ignored
            Present only for sklearn API compatibility.

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X)}.")
        if X.empty:
            raise ValueError("Input DataFrame is empty.")

        self.feature_names_in_ = list(X.columns)
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X.values.astype(np.float64))

        # Retrieve the number of components chosen by PCA
        self.n_components_ = self.pipeline_.named_steps["pca"].n_components_
        logger.info(
            "FeaturePreprocessor fitted: %d input features → %d PCA components "
            "(%.1f %% variance explained).",
            len(self.feature_names_in_),
            self.n_components_,
            self.pca_variance_threshold * 100,
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Scale and project *X* into PCA space.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature matrix with the same columns seen during ``fit``.

        Returns
        -------
        np.ndarray, shape (n_samples, n_components_)
            PCA-compressed feature matrix.
        """
        if self.pipeline_ is None:
            raise RuntimeError("FeaturePreprocessor has not been fitted yet.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X)}.")

        return self.pipeline_.transform(X.values.astype(np.float64))

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        """Fit on *X* and return the transformed array in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature matrix.
        y : ignored

        Returns
        -------
        np.ndarray, shape (n_samples, n_components_)
        """
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Per-component explained variance ratio (requires fitted state)."""
        if self.pipeline_ is None:
            raise RuntimeError("FeaturePreprocessor has not been fitted yet.")
        return self.pipeline_.named_steps["pca"].explained_variance_ratio_

    @property
    def cumulative_variance_(self) -> float:
        """Total cumulative variance explained by the selected components."""
        return float(np.sum(self.explained_variance_ratio_))
