"""
data_ingestion/tests.py
=======================
Tests for the yfinance bulk ingestion pipeline.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from data_ingestion.yfinance_bulk_ingestion import (
    create_sequences,
    engineer_features,
    save_training_data,
    run_ingestion,
    download_ticker_data,
    download_all_symbols,
    SYMBOL_MAP,
    DEFAULT_SEQ_LENGTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(rows: int = 100) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame similar to yfinance output."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start="2000-01-03", periods=rows, freq="B")
    close = 100.0 + np.cumsum(rng.standard_normal(rows))
    return pd.DataFrame(
        {
            "Open": close + rng.uniform(-1, 1, rows),
            "High": close + rng.uniform(0, 2, rows),
            "Low": close - rng.uniform(0, 2, rows),
            "Close": close,
            "Volume": rng.integers(1_000, 100_000, rows),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class EngineerFeaturesTests(unittest.TestCase):
    """Tests for :func:`engineer_features`."""

    def test_adds_daily_return_and_volatility(self):
        df = _make_ohlcv_df(100)
        result = engineer_features(df)
        self.assertIn("daily_return", result.columns)
        self.assertIn("volatility", result.columns)

    def test_drops_nan_rows(self):
        df = _make_ohlcv_df(100)
        result = engineer_features(df)
        self.assertFalse(result.isna().any().any())

    def test_preserves_ohlcv_columns(self):
        df = _make_ohlcv_df(50)
        result = engineer_features(df)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            self.assertIn(col, result.columns)

    def test_handles_multiindex_columns(self):
        df = _make_ohlcv_df(50)
        # Simulate yfinance MultiIndex: (metric, ticker)
        df.columns = pd.MultiIndex.from_tuples(
            [(c, "TEST.NS") for c in df.columns]
        )
        result = engineer_features(df)
        self.assertIn("daily_return", result.columns)
        self.assertIn("volatility", result.columns)


class CreateSequencesTests(unittest.TestCase):
    """Tests for :func:`create_sequences`."""

    def setUp(self):
        self.df = engineer_features(_make_ohlcv_df(200))

    def test_output_shapes(self):
        X, y = create_sequences(self.df, seq_length=30)
        num_features = 7  # OHLCV + daily_return + volatility
        expected_samples = len(self.df) - 30
        self.assertEqual(X.shape, (expected_samples, 30, num_features))
        self.assertEqual(y.shape, (expected_samples,))

    def test_labels_are_binary(self):
        _, y = create_sequences(self.df, seq_length=30)
        unique = set(np.unique(y))
        self.assertTrue(unique.issubset({0.0, 1.0}))

    def test_dtype_is_float32(self):
        X, y = create_sequences(self.df, seq_length=10)
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)

    def test_custom_seq_length(self):
        X, y = create_sequences(self.df, seq_length=5)
        self.assertEqual(X.shape[1], 5)


class SaveTrainingDataTests(unittest.TestCase):
    """Tests for :func:`save_training_data`."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_creates_npy_files(self):
        X = np.zeros((10, 30, 7), dtype=np.float32)
        y = np.zeros(10, dtype=np.float32)
        save_training_data(X, y, self.tmpdir)

        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "X_train.npy")))
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "y_train.npy")))

    def test_saved_arrays_match(self):
        rng = np.random.default_rng(99)
        X = rng.standard_normal((20, 30, 7)).astype(np.float32)
        y = rng.integers(0, 2, 20).astype(np.float32)
        save_training_data(X, y, self.tmpdir)

        X_loaded = np.load(os.path.join(self.tmpdir, "X_train.npy"))
        y_loaded = np.load(os.path.join(self.tmpdir, "y_train.npy"))
        np.testing.assert_array_equal(X, X_loaded)
        np.testing.assert_array_equal(y, y_loaded)

    def test_creates_output_directory(self):
        nested = os.path.join(self.tmpdir, "sub", "dir")
        X = np.zeros((5, 10, 7), dtype=np.float32)
        y = np.zeros(5, dtype=np.float32)
        save_training_data(X, y, nested)
        self.assertTrue(os.path.isdir(nested))


class DownloadTickerDataTests(unittest.TestCase):
    """Tests for :func:`download_ticker_data` (mocked network)."""

    @patch("data_ingestion.yfinance_bulk_ingestion.yf.download")
    def test_returns_dataframe_on_success(self, mock_dl):
        mock_dl.return_value = _make_ohlcv_df(50)
        result = download_ticker_data("TEST.NS")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    @patch("data_ingestion.yfinance_bulk_ingestion.yf.download")
    def test_returns_empty_on_failure(self, mock_dl):
        mock_dl.side_effect = Exception("network error")
        result = download_ticker_data("BAD.NS")
        self.assertTrue(result.empty)

    @patch("data_ingestion.yfinance_bulk_ingestion.yf.download")
    def test_drops_na_rows(self, mock_dl):
        df = _make_ohlcv_df(50)
        df.iloc[0, 0] = np.nan
        mock_dl.return_value = df
        result = download_ticker_data("TEST.NS")
        self.assertFalse(result.isna().any().any())


class DownloadAllSymbolsTests(unittest.TestCase):
    """Tests for :func:`download_all_symbols` (mocked network)."""

    @patch("data_ingestion.yfinance_bulk_ingestion.download_ticker_data")
    def test_returns_dict_of_dataframes(self, mock_dl):
        mock_dl.return_value = _make_ohlcv_df(50)
        result = download_all_symbols({"A-EQ": "A.NS", "B-EQ": "B.NS"})
        self.assertEqual(len(result), 2)
        self.assertIn("A.NS", result)

    @patch("data_ingestion.yfinance_bulk_ingestion.download_ticker_data")
    def test_skips_empty_downloads(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        result = download_all_symbols({"A-EQ": "A.NS"})
        self.assertEqual(len(result), 0)


class RunIngestionTests(unittest.TestCase):
    """Integration test for :func:`run_ingestion` (mocked network)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("data_ingestion.yfinance_bulk_ingestion.download_ticker_data")
    def test_end_to_end_saves_files(self, mock_dl):
        mock_dl.return_value = _make_ohlcv_df(200)
        run_ingestion(
            symbol_map={"TEST-EQ": "TEST.NS"},
            seq_length=10,
            output_dir=self.tmpdir,
        )
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "X_train.npy")))
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "y_train.npy")))

        X = np.load(os.path.join(self.tmpdir, "X_train.npy"))
        y = np.load(os.path.join(self.tmpdir, "y_train.npy"))
        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(X.shape[2], 7)
        self.assertEqual(len(y), X.shape[0])

    @patch("data_ingestion.yfinance_bulk_ingestion.download_ticker_data")
    def test_no_data_does_not_crash(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        # Should log an error but not raise.
        run_ingestion(
            symbol_map={"BAD-EQ": "BAD.NS"},
            output_dir=self.tmpdir,
        )
        self.assertFalse(
            os.path.isfile(os.path.join(self.tmpdir, "X_train.npy"))
        )


class SymbolMapTests(unittest.TestCase):
    """Verify the default symbol mapping constant."""

    def test_symbol_map_is_non_empty(self):
        self.assertGreater(len(SYMBOL_MAP), 0)

    def test_all_yahoo_tickers_end_with_ns(self):
        for yf_ticker in SYMBOL_MAP.values():
            self.assertTrue(
                yf_ticker.endswith(".NS"),
                f"{yf_ticker} does not end with .NS",
            )

    def test_default_seq_length_is_30(self):
        self.assertEqual(DEFAULT_SEQ_LENGTH, 30)


if __name__ == "__main__":
    unittest.main()
