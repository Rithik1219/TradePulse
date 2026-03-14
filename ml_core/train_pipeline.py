"""
ml_core/train_pipeline.py
=========================
End-to-end training script for TradePulse's Hybrid Ensemble.

Pipeline stages
---------------
1. **Mock data generation** — synthesises a DataFrame that simulates
   realistic OHLCV + sentiment time-series data with embedded technical
   indicator signals, so this script can be run and verified without any
   live market data.

2. **Technical feature engineering** — computes RSI, MACD, Bollinger
   Bands, ATR, OBV, Stochastic, Williams %R, CCI, MFI and many more
   indicators via ``TechnicalFeatureEngineer``, dramatically expanding
   the feature set available to the tabular model.

3. **Preprocessing** — fits ``FeaturePreprocessor`` on the enriched
   tabular features (RobustScaler → PCA at 95 % variance retention).

4. **Base-learner training with out-of-fold (OOF) predictions**:
   a. ``XGBEngine`` — trained on the PCA-compressed tabular features
      (now enriched with technical indicators).
   b. ``LSTMModel`` — trained on the 3-D sequential OHLCV+indicator
      tensor using a bidirectional LSTM with self-attention pooling.
   Both models produce OOF probability vectors using time-series
   cross-validation (``TimeSeriesSplit``) to avoid look-ahead bias.

5. **Meta-learner training** — ``MetaLearner`` (Logistic Regression)
   is fitted on a richer OOF probability matrix that includes confidence
   signals, disagreement, and interaction features.

6. **Final evaluation** — the full ensemble is evaluated on a hold-out
   test set and key metrics are reported.

Usage
-----
    python -m ml_core.train_pipeline          # from the repo root
    python ml_core/train_pipeline.py          # direct execution
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from ml_core.feature_engineering import TechnicalFeatureEngineer
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
SEQ_LEN: int = 30              # Look-back window for the LSTM (days)
# OHLCV (5) + sentiment (1) + RSI (1) + MACD histogram (1) + BB %B (1)
# + ATR normalised (1) + volume pressure (1) = 11 features per time step
SEQ_FEATURES: int = 11
TEST_SIZE: int = 200           # Hold-out test-set size
N_CV_SPLITS: int = 5           # TimeSeriesSplit folds for OOF generation
PCA_VARIANCE: float = 0.95     # Variance threshold for PCA


# ---------------------------------------------------------------------------
# Mock data generator
# ---------------------------------------------------------------------------

def generate_mock_data(
    n_samples: int = N_SAMPLES,
    seq_len: int = SEQ_LEN,
    seq_features: int = SEQ_FEATURES,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate synthetic OHLCV + sentiment data and derive technical features.

    The mock generator creates a realistic price walk, then derives OHLCV
    columns from it.  ``TechnicalFeatureEngineer`` is used to compute
    technical indicators directly from the OHLCV data — the same code path
    that would run in production — so the training pipeline validates the
    full feature-engineering stack.

    Returns
    -------
    df_tabular : pd.DataFrame, shape (n_samples, n_technical_features)
        Enriched tabular features (OHLCV + all technical indicators +
        sentiment) ready for ``FeaturePreprocessor``.
    X_seq : np.ndarray, shape (n_samples, seq_len, seq_features)
        3-D sequential tensor for the LSTM containing OHLCV + selected
        per-step technical indicators.
    y : np.ndarray, shape (n_samples,)
        Binary labels: 1 = price went UP, 0 = price went DOWN.
    """
    rng = np.random.default_rng(random_state)

    # ---- Simulate a price series -----------------------------------------
    # Use a random walk with slight momentum / mean-reversion to give the
    # model learnable structure beyond pure noise.
    returns = rng.standard_normal(n_samples + seq_len + 50) * 0.01
    # Add a small momentum component (AR(1) with φ=0.05)
    for t in range(1, len(returns)):
        returns[t] += 0.05 * returns[t - 1]
    price = 100.0 * np.exp(np.cumsum(returns))

    close = price[seq_len: seq_len + n_samples]
    # Synthesise OHLC from close: realistic intraday spread
    daily_vol = np.abs(rng.standard_normal(n_samples)) * 0.008 + 0.002
    high = close * (1.0 + daily_vol)
    low = close * (1.0 - daily_vol)
    open_ = close * (1.0 + rng.standard_normal(n_samples) * 0.005)
    volume = rng.lognormal(mean=10.0, sigma=0.8, size=n_samples) * (
        1.0 + 0.3 * np.abs(returns[seq_len: seq_len + n_samples])
    )
    sentiment = rng.uniform(-1.0, 1.0, size=n_samples)

    # ---- Build OHLCV DataFrame and compute technical indicators ----------
    df_raw = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "sentiment": sentiment,
        }
    )

    engineer = TechnicalFeatureEngineer()
    df_tabular = engineer.transform(df_raw, extra_cols=["sentiment"])

    # ---- Sequential tensor for LSTM --------------------------------------
    # Build overlapping windows of shape (n_samples, seq_len, seq_features).
    # Features per time step: close, high, low, open, volume, sentiment,
    # rsi_14, macd_hist, bb_pct_b, atr_14_norm, volume_pressure
    seq_cols = (
        [engineer.close_col, engineer.high_col, engineer.low_col,
         engineer.open_col, engineer.volume_col, "sentiment"]
        + ["rsi_14", "macd_hist", "bb_pct_b", "atr_14_norm", "volume_pressure"]
    )
    # Pad df_tabular to include historical rows for window construction
    df_ext = pd.concat(
        [
            pd.DataFrame(
                np.zeros((seq_len, df_tabular.shape[1])),
                columns=df_tabular.columns,
            ),
            df_tabular,
        ],
        ignore_index=True,
    )
    seq_array = df_ext[seq_cols].values  # (n_samples + seq_len, seq_features)
    # Build sliding windows
    X_seq = np.stack(
        [seq_array[i: i + seq_len] for i in range(n_samples)], axis=0
    ).astype(np.float32)  # (n_samples, seq_len, seq_features)

    # ---- Labels (binary) --------------------------------------------------
    # 1 if close tomorrow is higher than close today
    y = (np.diff(close, append=close[-1]) > 0).astype(np.float32)
    # Fix the last label (no future data) — set to majority class
    y[-1] = float(y[:-1].mean() >= 0.5)

    logger.info(
        "Mock data: %d samples | tabular shape: %s | seq shape: %s | "
        "label balance: %.2f %%",
        n_samples,
        df_tabular.shape,
        X_seq.shape,
        100.0 * y.mean(),
    )
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
        Enriched tabular feature matrix (OHLCV + technical indicators).
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

        # ---- XGBoost — improved engine with more hyperparameters ----------
        xgb = XGBEngine(
            max_depth=4,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            min_child_weight=5.0,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
        )
        xgb.fit(tab_train_pca, y_train_fold, tab_val_pca, y_val_fold)
        oof_xgb[val_idx] = xgb.predict_proba(tab_val_pca)

        # ---- LSTM — bidirectional + attention, more features per step -----
        lstm = LSTMModel(
            input_size=seq_train.shape[2],
            seq_len=seq_train.shape[1],
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            n_attention_heads=4,
            learning_rate=1e-3,
            weight_decay=1e-4,
            epochs=15,          # Keep low for mock demo; increase in production
            patience=5,
            lr_scheduler_patience=3,
            label_smoothing=0.05,
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
    # 1. Generate mock data (OHLCV → technical features → sequential tensor)
    # -----------------------------------------------------------------------
    logger.info("Generating mock dataset with technical indicators …")
    df_tabular, X_seq, y = generate_mock_data()

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
    xgb_final = XGBEngine(
        max_depth=4,
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        min_child_weight=5.0,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
    )
    xgb_final.fit(tab_train_pca, y_train)

    logger.info("Training final LSTMModel on full training split …")
    lstm_final = LSTMModel(
        input_size=X_seq_train.shape[2],
        seq_len=X_seq_train.shape[1],
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        n_attention_heads=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=15,
        patience=5,
        lr_scheduler_patience=3,
        label_smoothing=0.05,
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
    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    run_training_pipeline()
