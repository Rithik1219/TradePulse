"""
ml_core/train_pipeline.py
=========================
End-to-end training script for TradePulse's Hybrid Ensemble.

Pipeline stages
---------------
1. **Load local data** — reads ``X_train.npy`` and ``y_train.npy`` from
   ``data_ingestion/training_data/`` (produced by
   ``data_ingestion/yfinance_bulk_ingestion.py``).

2. **Preprocessing** — derives a 2-D tabular representation from the
   3-D sequential tensor (by flattening each sample's time-step
   features), then fits ``FeaturePreprocessor`` on those tabular features
   (RobustScaler → PCA at 95 % variance retention).

3. **Base-learner training with out-of-fold (OOF) predictions**:
   a. ``XGBEngine`` — trained on the PCA-compressed tabular features.
   b. ``LSTMModel`` — trained on the 3-D sequential OHLCV tensor.
   Both models produce OOF probability vectors using time-series
   cross-validation (``TimeSeriesSplit``) to avoid look-ahead bias.

4. **Meta-learner training** — ``MetaLearner`` (Logistic Regression)
   is fitted on the OOF probability matrix.

5. **Final evaluation** — the full ensemble is evaluated on a hold-out
   test set and key metrics are reported.

6. **Save artifacts** — all trained models and the scaler/preprocessor
   are persisted to ``saved_models/`` for use by the inference engine.

Usage
-----
    python -m ml_core.train_pipeline          # from the repo root
    python ml_core/train_pipeline.py          # direct execution

Prerequisites
-------------
    Run ``python -m data_ingestion.yfinance_bulk_ingestion`` first to
    generate ``data_ingestion/training_data/X_train.npy`` and
    ``data_ingestion/training_data/y_train.npy``.
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

_REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RANDOM_SEED: int = 42
TEST_SIZE: int = 200           # Hold-out test-set size
N_CV_SPLITS: int = 5           # TimeSeriesSplit folds for OOF generation
PCA_VARIANCE: float = 0.95     # Variance threshold for PCA
SAVED_MODELS_DIR: str = os.path.join(_REPO_ROOT, "saved_models")
TRAINING_DATA_DIR: str = os.path.join(
    _REPO_ROOT, "data_ingestion", "training_data"
)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_training_data(
    data_dir: str = TRAINING_DATA_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-generated training tensors from disk.

    Expects ``X_train.npy`` and ``y_train.npy`` to have been produced by
    ``data_ingestion/yfinance_bulk_ingestion.py``.

    Parameters
    ----------
    data_dir : str
        Directory containing ``X_train.npy`` and ``y_train.npy``.

    Returns
    -------
    X_seq : np.ndarray, shape (n_samples, seq_len, n_features)
        3-D sequential tensor (OHLCV + engineered features per time step).
    y : np.ndarray, shape (n_samples,)
        Binary labels: 1 = next-day price UP, 0 = DOWN.

    Raises
    ------
    FileNotFoundError
        When either ``.npy`` file is missing from *data_dir*.
    """
    x_path = os.path.join(data_dir, "X_train.npy")
    y_path = os.path.join(data_dir, "y_train.npy")

    for path in (x_path, y_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Training data file not found: {path}\n"
                "Run 'python -m data_ingestion.yfinance_bulk_ingestion' first."
            )

    X_seq = np.load(x_path)
    y = np.load(y_path)

    logger.info("Loaded X_train from %s — shape: %s", x_path, X_seq.shape)
    logger.info("Loaded y_train from %s — shape: %s", y_path, y.shape)

    return X_seq.astype(np.float32), y.astype(np.float32)


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
    # 1. Load local training data
    # -----------------------------------------------------------------------
    logger.info("Loading training data from %s …", TRAINING_DATA_DIR)
    X_seq, y = load_training_data()

    # Derive a 2-D tabular representation for XGBoost/preprocessing by
    # flattening each sample's (seq_len × n_features) time-step matrix.
    n_samples, seq_len, n_seq_features = X_seq.shape
    n_tabular_features = seq_len * n_seq_features
    feature_names = [f"feature_{i:04d}" for i in range(n_tabular_features)]
    df_tabular = pd.DataFrame(
        X_seq.reshape(n_samples, n_tabular_features), columns=feature_names
    )

    logger.info(
        "Dataset: %d samples | %d tabular features (flattened) | "
        "seq shape: %s | label balance: %.2f %%",
        n_samples,
        n_tabular_features,
        X_seq.shape,
        100 * y.mean(),
    )

    # -----------------------------------------------------------------------
    # 2. Train/test split (time-aware — no shuffle)
    # -----------------------------------------------------------------------
    n_test = min(TEST_SIZE, n_samples // 10)
    n_train = n_samples - n_test

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
    logger.info("FINAL TEST METRICS (Meta-Learner on hold-out validation set)")
    logger.info("  ROC-AUC   : %.4f", auc)
    logger.info("  Log-Loss  : %.4f", ll)
    logger.info("  Accuracy  : %.4f", accuracy)
    logger.info("=" * 60)

    print("=" * 60)
    print("Meta-Learner Validation Results:")  # prominent console summary
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Log-Loss : {ll:.4f}")
    print(f"  Accuracy : {accuracy:.4f}")
    print("=" * 60)

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
