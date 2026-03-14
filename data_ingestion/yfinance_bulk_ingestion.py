"""
data_ingestion/yfinance_bulk_ingestion.py
=========================================
Standalone script to download 20+ years of historical OHLCV data for the
TradePulse portfolio via ``yfinance``, engineer features, create 30-day
sliding-window sequences for the LSTM, and save the training tensors to
disk.

Usage
-----
::

    python -m data_ingestion.yfinance_bulk_ingestion   # from repo root
    python data_ingestion/yfinance_bulk_ingestion.py   # direct execution
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Map Angel One trading symbols → Yahoo Finance tickers.
SYMBOL_MAP: Dict[str, str] = {
    "RELIANCE-EQ": "RELIANCE.NS",
    "TCS-EQ": "TCS.NS",
    "HDFC-EQ": "HDFCBANK.NS",
}

# Default look-back window (must match LSTMModel.seq_len).
DEFAULT_SEQ_LENGTH: int = 30

# Output directory for saved tensors (relative to the repository root).
_TRAINING_DATA_DIR: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "training_data",
)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_ticker_data(ticker: str) -> pd.DataFrame:
    """Download all available historical OHLCV data for *ticker*.

    Uses ``yfinance.download`` with ``period="max"`` to retrieve up to
    20+ years of daily data.  Missing values are dropped so downstream
    feature engineering receives a clean DataFrame.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. ``"RELIANCE.NS"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``Open``, ``High``, ``Low``, ``Close``, ``Volume``
        (standard ``yfinance`` column names).  Returns an empty
        ``DataFrame`` when the download fails or yields no rows.
    """
    logger.info("Downloading data for %s (period=max) …", ticker)
    try:
        df = yf.download(ticker, period="max", progress=False)
    except Exception:
        logger.exception("Failed to download data for %s.", ticker)
        return pd.DataFrame()

    if df.empty:
        logger.warning("No data returned for %s.", ticker)
        return df

    df = df.dropna()
    logger.info(
        "Downloaded %d rows for %s (range: %s → %s).",
        len(df),
        ticker,
        str(df.index.min().date()),
        str(df.index.max().date()),
    )
    return df


def download_all_symbols(
    symbol_map: Dict[str, str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data for every symbol in *symbol_map*.

    Parameters
    ----------
    symbol_map : dict, optional
        Angel One symbol → Yahoo Finance ticker mapping.  Falls back to
        :data:`SYMBOL_MAP` when *None*.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of **Yahoo Finance ticker** → cleaned OHLCV DataFrame.
        Tickers whose download returned no data are omitted.
    """
    if symbol_map is None:
        symbol_map = SYMBOL_MAP

    result: Dict[str, pd.DataFrame] = {}
    for angel_symbol, yf_ticker in symbol_map.items():
        logger.info(
            "Processing %s → %s …", angel_symbol, yf_ticker
        )
        df = download_ticker_data(yf_ticker)
        if not df.empty:
            result[yf_ticker] = df

    logger.info(
        "Bulk download complete — %d / %d symbols succeeded.",
        len(result),
        len(symbol_map),
    )
    return result


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic features used by the LSTM training pipeline.

    Features added:

    * **daily_return** — percentage change of Close (``Close.pct_change``).
    * **volatility** — 20-day rolling standard-deviation of daily returns.

    Rows containing ``NaN`` values introduced by the rolling window are
    dropped so the returned DataFrame is ready for sequencing.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame as returned by :func:`download_ticker_data`.

    Returns
    -------
    pd.DataFrame
        Original OHLCV columns plus ``daily_return`` and ``volatility``.
    """
    df = df.copy()

    # Flatten MultiIndex columns produced by yfinance when downloading a
    # single ticker (e.g. ``("Close", "RELIANCE.NS")`` → ``"Close"``).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["daily_return"] = df["Close"].pct_change()
    df["volatility"] = df["daily_return"].rolling(window=20).std()

    df = df.dropna()
    logger.info(
        "Feature engineering complete — %d rows, %d features.",
        len(df),
        len(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# Sequence creation
# ---------------------------------------------------------------------------

def create_sequences(
    df: pd.DataFrame,
    seq_length: int = DEFAULT_SEQ_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences and binary target labels.

    Slides a window of *seq_length* days over the feature matrix and
    produces:

    * **X** — shape ``(num_samples, seq_length, num_features)``
    * **y** — shape ``(num_samples,)`` with ``1`` if the next day's
      ``Close`` price is higher than the last day's ``Close`` in the
      window, else ``0``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame (from :func:`engineer_features`).
    seq_length : int
        Number of look-back days per sample (default 30).

    Returns
    -------
    X : np.ndarray, dtype float32
    y : np.ndarray, dtype float32
    """
    feature_cols = ["Open", "High", "Low", "Close", "Volume",
                    "daily_return", "volatility"]
    data = df[feature_cols].values.astype(np.float32)
    closes = df["Close"].values.astype(np.float32)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for i in range(len(data) - seq_length):
        X_list.append(data[i: i + seq_length])
        # Target: 1 if next day's close is higher than last day of window
        y_list.append(
            1.0 if closes[i + seq_length] > closes[i + seq_length - 1] else 0.0
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(
        "Created sequences — X shape: %s, y shape: %s, label balance: %.1f %% positive.",
        X.shape,
        y.shape,
        100.0 * y.mean() if len(y) else 0.0,
    )
    return X, y


# ---------------------------------------------------------------------------
# Save to disk
# ---------------------------------------------------------------------------

def save_training_data(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str = _TRAINING_DATA_DIR,
) -> None:
    """Persist training tensors as ``.npy`` files.

    Files written:

    * ``<output_dir>/X_train.npy``
    * ``<output_dir>/y_train.npy``

    Parameters
    ----------
    X : np.ndarray
        Feature tensor of shape ``(num_samples, seq_length, num_features)``.
    y : np.ndarray
        Label array of shape ``(num_samples,)``.
    output_dir : str
        Target directory (created if it does not exist).
    """
    os.makedirs(output_dir, exist_ok=True)

    x_path = os.path.join(output_dir, "X_train.npy")
    y_path = os.path.join(output_dir, "y_train.npy")

    np.save(x_path, X)
    np.save(y_path, y)

    logger.info("Saved X_train → %s  (%s)", x_path, X.shape)
    logger.info("Saved y_train → %s  (%s)", y_path, y.shape)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_ingestion(
    symbol_map: Dict[str, str] | None = None,
    seq_length: int = DEFAULT_SEQ_LENGTH,
    output_dir: str = _TRAINING_DATA_DIR,
) -> None:
    """Execute the full bulk-ingestion pipeline.

    1. Download OHLCV data for all symbols.
    2. Engineer features per ticker.
    3. Create sliding-window sequences.
    4. Concatenate across tickers and save to disk.

    Parameters
    ----------
    symbol_map : dict, optional
        Overrides :data:`SYMBOL_MAP` when provided.
    seq_length : int
        Look-back window length (default 30).
    output_dir : str
        Directory to save the ``.npy`` training tensors.
    """
    logger.info("=" * 60)
    logger.info("TradePulse — yfinance Bulk Data Ingestion")
    logger.info("=" * 60)

    all_data = download_all_symbols(symbol_map)
    if not all_data:
        logger.error("No data downloaded. Aborting.")
        return

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for ticker, raw_df in all_data.items():
        logger.info("Processing sequences for %s …", ticker)
        feat_df = engineer_features(raw_df)
        if len(feat_df) <= seq_length:
            logger.warning(
                "Skipping %s — not enough rows (%d) for seq_length=%d.",
                ticker,
                len(feat_df),
                seq_length,
            )
            continue
        X, y = create_sequences(feat_df, seq_length)
        X_parts.append(X)
        y_parts.append(y)

    if not X_parts:
        logger.error("No sequences created. Aborting.")
        return

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    logger.info(
        "Combined dataset — X: %s, y: %s, positive label ratio: %.1f %%.",
        X_all.shape,
        y_all.shape,
        100.0 * y_all.mean(),
    )

    save_training_data(X_all, y_all, output_dir)
    logger.info("Bulk ingestion pipeline complete.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )
    run_ingestion()
