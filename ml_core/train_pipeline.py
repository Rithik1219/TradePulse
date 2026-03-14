"""
ml_core/train_pipeline.py
=========================
End-to-end training script for TradePulse's Hybrid Ensemble.

Pipeline stages
---------------
1. **Mock data generation** — synthesises a DataFrame that simulates
   thousands of raw technical indicators and sentiment scores plus an
   OHLCV+sentiment time-series, so this script can be run and verified
   without any live market data.

2. **Preprocessing** — fits ``FeaturePreprocessor`` on the tabular
   features (RobustScaler → PCA at 95 % variance retention).

3. **Base-learner training with out-of-fold (OOF) predictions**:
   a. ``XGBEngine`` — trained on the PCA-compressed tabular features.
   b. ``LSTMModel`` — trained on the 3-D sequential OHLCV+sentiment
      tensor.
   Both models produce OOF probability vectors using time-series
   cross-validation (``TimeSeriesSplit``) to avoid look-ahead bias.

4. **Meta-learner training** — ``MetaLearner`` (Logistic Regression)
   is fitted on the OOF probability matrix.

5. **Final evaluation** — the full ensemble is evaluated on a hold-out
   test set and key metrics are reported.

Usage
-----
    python -m ml_core.train_pipeline          # from the repo root
    python ml_core/train_pipeline.py          # direct execution
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from ml_core.lstm_engine import LSTMModel
from ml_core.meta_learner import MetaLearner
from ml_core.preprocessing import FeaturePreprocessor
from ml_core.xgb_engine import XGBEngine

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
N_SAMPLES: int = 2000          # Total number of data points (days)
N_RAW_FEATURES: int = 500      # Simulated raw technical + sentiment columns
SEQ_LEN: int = 30              # Look-back window for the LSTM (days)
OHLCV_FEATURES: int = 6        # OHLCV (5) + sentiment score (1)
TEST_SIZE: int = 200           # Hold-out test-set size
N_CV_SPLITS: int = 5           # TimeSeriesSplit folds for OOF generation
PCA_VARIANCE: float = 0.95     # Variance threshold for PCA
SAVED_MODELS_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models",
)


# ---------------------------------------------------------------------------
# Mock data generator
# ---------------------------------------------------------------------------

def generate_mock_data(
    n_samples: int = N_SAMPLES,
    n_features: int = N_RAW_FEATURES,
    seq_len: int = SEQ_LEN,
    ohlcv_features: int = OHLCV_FEATURES,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate synthetic data mimicking the TradePulse feature set.

    Returns
    -------
    df_tabular : pd.DataFrame, shape (n_samples, n_features)
        High-dimensional tabular features (raw technical indicators +
        sentiment scores) before any preprocessing.
    X_seq : np.ndarray, shape (n_samples, seq_len, ohlcv_features)
        3-D sequential tensor for the LSTM: the last ``seq_len`` days of
        OHLCV data and a sentiment score for each sample.
    y : np.ndarray, shape (n_samples,)
        Binary labels: 1 = price went UP, 0 = price went DOWN.
    """
    rng = np.random.default_rng(random_state)

    # --- Tabular features --------------------------------------------------
    feature_names = [f"feature_{i:04d}" for i in range(n_features)]
    raw_data = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    # Introduce realistic sparsity / outliers
    outlier_mask = rng.random((n_samples, n_features)) < 0.01
    raw_data[outlier_mask] *= 10.0
    df_tabular = pd.DataFrame(raw_data, columns=feature_names)

    # --- Sequential tensor for LSTM ----------------------------------------
    # Simulate OHLCV-like data: cumulative price walk + volume + sentiment
    price = np.cumsum(rng.standard_normal((n_samples + seq_len, 1)), axis=0)
    # Build overlapping windows of shape (n_samples, seq_len, 1)
    price_windows = np.lib.stride_tricks.sliding_window_view(
        price[:, 0], seq_len
    )  # (n_samples, seq_len)
    # Repeat for O/H/L/C/V (4 synthetic price channels + 1 volume + 1 sentiment)
    price_windows = price_windows[:n_samples]
    volume = rng.lognormal(mean=10, sigma=1, size=(n_samples, seq_len, 1))
    sentiment = rng.uniform(-1, 1, size=(n_samples, seq_len, 1))
    price_channels = np.stack(
        [price_windows] * (ohlcv_features - 2), axis=-1
    )  # (n_samples, seq_len, ohlcv_features-2)
    X_seq = np.concatenate(
        [price_channels, volume, sentiment], axis=-1
    ).astype(np.float32)  # (n_samples, seq_len, ohlcv_features)

    # --- Labels (binary) ---------------------------------------------------
    # Use a simple future-return rule as a proxy for real labels
    future_returns = np.diff(price[:, 0])[:n_samples]
    y = (future_returns > 0).astype(np.float32)

    return df_tabular, X_seq, y


# ---------------------------------------------------------------------------
# OOF generation helpers
# ---------------------------------------------------------------------------

def generate_oof_predictions(
    df_tabular: pd.DataFrame,
    X_seq: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_CV_SPLITS,
) -> tuple[np.ndarray, np.ndarray, FeaturePreprocessor]:
    """Produce out-of-fold predictions from both base learners.

    Uses ``TimeSeriesSplit`` to respect temporal ordering and avoid
    look-ahead bias.

    Parameters
    ----------
    df_tabular : pd.DataFrame
        Raw tabular feature matrix.
    X_seq : np.ndarray
        Sequential tensor for the LSTM.
    y : np.ndarray
        Binary labels.
    n_splits : int
        Number of cross-validation folds.

    Returns
    -------
    oof_lstm  : np.ndarray, shape (n_samples,)
    oof_xgb   : np.ndarray, shape (n_samples,)
    preprocessor : FeaturePreprocessor
        The *last* fitted preprocessor (used for final test-set inference).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_lstm = np.zeros(len(y), dtype=np.float32)
    oof_xgb = np.zeros(len(y), dtype=np.float32)
    last_preprocessor: FeaturePreprocessor | None = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_tabular), 1):
        logger.info("OOF fold %d/%d …", fold, n_splits)

        # ---- Split --------------------------------------------------------
        tab_train = df_tabular.iloc[train_idx]
        tab_val = df_tabular.iloc[val_idx]
        seq_train = X_seq[train_idx]
        seq_val = X_seq[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        # ---- Preprocessing ------------------------------------------------
        preprocessor = FeaturePreprocessor(
            pca_variance_threshold=PCA_VARIANCE, random_state=RANDOM_SEED
        )
        tab_train_pca = preprocessor.fit_transform(tab_train)
        tab_val_pca = preprocessor.transform(tab_val)
        last_preprocessor = preprocessor

        # ---- XGBoost ------------------------------------------------------
        xgb = XGBEngine(random_state=RANDOM_SEED)
        # Pass the explicit validation fold so no additional rows are carved
        # out of the training split for early-stopping purposes.
        xgb.fit(tab_train_pca, y_train_fold, tab_val_pca, y_val_fold)
        oof_xgb[val_idx] = xgb.predict_proba(tab_val_pca)

        # ---- LSTM ---------------------------------------------------------
        lstm = LSTMModel(
            input_size=seq_train.shape[2],
            seq_len=seq_train.shape[1],
            hidden_size=64,
            num_layers=1,
            epochs=10,          # Low epochs for mock demo; increase in production
            patience=5,
            random_state=RANDOM_SEED,
        )
        lstm.fit(seq_train, y_train_fold, seq_val, y_val_fold)
        oof_lstm[val_idx] = lstm.predict_proba(seq_val)

    # The loop must execute at least once, so last_preprocessor is always set.
    assert last_preprocessor is not None, (
        "generate_oof_predictions requires n_splits >= 1."
    )
    return oof_lstm, oof_xgb, last_preprocessor


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline() -> None:
    """Execute the full end-to-end Hybrid Ensemble training pipeline."""

    logger.info("=" * 60)
    logger.info("TradePulse — Hybrid Ensemble Training Pipeline")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Generate mock data
    # -----------------------------------------------------------------------
    logger.info("Generating mock dataset …")
    df_tabular, X_seq, y = generate_mock_data()
    logger.info(
        "Dataset: %d samples | %d tabular features | "
        "seq shape: %s | label balance: %.2f %%",
        len(y),
        df_tabular.shape[1],
        X_seq.shape,
        100 * y.mean(),
    )

    # -----------------------------------------------------------------------
    # 2. Train/test split (time-aware — no shuffle)
    # -----------------------------------------------------------------------
    n_test = TEST_SIZE
    n_train = len(y) - n_test

    df_train, df_test = df_tabular.iloc[:n_train], df_tabular.iloc[n_train:]
    X_seq_train, X_seq_test = X_seq[:n_train], X_seq[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    logger.info("Train samples: %d | Test samples: %d", n_train, n_test)

    # -----------------------------------------------------------------------
    # 3. OOF predictions for meta-learner stacking
    # -----------------------------------------------------------------------
    logger.info("Generating out-of-fold predictions for stacking …")
    oof_lstm, oof_xgb, _ = generate_oof_predictions(
        df_train, X_seq_train, y_train
    )
    logger.info(
        "OOF AUC — LSTM: %.4f  XGBoost: %.4f",
        roc_auc_score(y_train, oof_lstm),
        roc_auc_score(y_train, oof_xgb),
    )

    # -----------------------------------------------------------------------
    # 4. Fit final base learners on full training data
    # -----------------------------------------------------------------------
    logger.info("Fitting final preprocessor on full training data …")
    preprocessor = FeaturePreprocessor(
        pca_variance_threshold=PCA_VARIANCE, random_state=RANDOM_SEED
    )
    tab_train_pca = preprocessor.fit_transform(df_train)
    tab_test_pca = preprocessor.transform(df_test)
    logger.info(
        "PCA: %d features → %d components (%.1f %% variance)",
        df_train.shape[1],
        preprocessor.n_components_,
        preprocessor.cumulative_variance_ * 100,
    )

    logger.info("Training final XGBEngine on full training split …")
    xgb_final = XGBEngine(random_state=RANDOM_SEED)
    xgb_final.fit(tab_train_pca, y_train)

    logger.info("Training final LSTMModel on full training split …")
    lstm_final = LSTMModel(
        input_size=X_seq_train.shape[2],
        seq_len=X_seq_train.shape[1],
        hidden_size=64,
        num_layers=1,
        epochs=10,
        patience=5,
        random_state=RANDOM_SEED,
    )
    lstm_final.fit(X_seq_train, y_train)

    # -----------------------------------------------------------------------
    # 5. Train meta-learner on OOF probabilities
    # -----------------------------------------------------------------------
    logger.info("Training MetaLearner on OOF probability vectors …")
    meta = MetaLearner(C=1.0, random_state=RANDOM_SEED)
    meta.fit(oof_lstm, oof_xgb, y_train)

    # -----------------------------------------------------------------------
    # 6. Evaluate on hold-out test set
    # -----------------------------------------------------------------------
    logger.info("Evaluating ensemble on hold-out test set …")
    test_lstm_probs = lstm_final.predict_proba(X_seq_test)
    test_xgb_probs = xgb_final.predict_proba(tab_test_pca)
    test_final_probs = meta.predict_proba(test_lstm_probs, test_xgb_probs)
    test_preds = meta.predict(test_lstm_probs, test_xgb_probs)

    auc = roc_auc_score(y_test, test_final_probs)
    ll = log_loss(y_test, test_final_probs)
    accuracy = float((test_preds == y_test).mean())

    logger.info("=" * 60)
    logger.info("FINAL TEST METRICS")
    logger.info("  ROC-AUC   : %.4f", auc)
    logger.info("  Log-Loss  : %.4f", ll)
    logger.info("  Accuracy  : %.4f", accuracy)
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 7. Save trained artifacts to disk
    # -----------------------------------------------------------------------
    logger.info("Saving trained artifacts to %s …", SAVED_MODELS_DIR)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    preprocessor.save(os.path.join(SAVED_MODELS_DIR, "preprocessor.joblib"))
    xgb_final.save(os.path.join(SAVED_MODELS_DIR, "xgb_engine.joblib"))
    lstm_final.save(os.path.join(SAVED_MODELS_DIR, "lstm_model"))
    meta.save(os.path.join(SAVED_MODELS_DIR, "meta_learner.joblib"))

    logger.info("All artifacts saved. Training pipeline complete.")


if __name__ == "__main__":
    run_training_pipeline()
