"""
ml_core/tests.py
================
Tests for ml_core model persistence (save / load) and the TradePulsePredictor
inference engine.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml_core.lstm_engine import LSTMModel
from ml_core.meta_learner import MetaLearner
from ml_core.predictor import TradePulsePredictor
from ml_core.preprocessing import FeaturePreprocessor
from ml_core.xgb_engine import XGBEngine


class FeaturePreprocessorPersistenceTests(unittest.TestCase):
    """Tests for FeaturePreprocessor.save / .load."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(42)
        self.df = pd.DataFrame(
            rng.standard_normal((100, 20)),
            columns=[f"f{i}" for i in range(20)],
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_and_load_roundtrip(self):
        fp = FeaturePreprocessor(pca_variance_threshold=0.95, random_state=42)
        fp.fit(self.df)
        path = os.path.join(self.tmpdir, "preprocessor.joblib")
        fp.save(path)

        loaded = FeaturePreprocessor.load(path)
        original = fp.transform(self.df)
        restored = loaded.transform(self.df)
        np.testing.assert_array_almost_equal(original, restored)

    def test_save_unfitted_raises(self):
        fp = FeaturePreprocessor()
        path = os.path.join(self.tmpdir, "preprocessor.joblib")
        with self.assertRaises(RuntimeError):
            fp.save(path)


class XGBEnginePersistenceTests(unittest.TestCase):
    """Tests for XGBEngine.save / .load."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((100, 10)).astype(np.float32)
        self.y = (rng.random(100) > 0.5).astype(np.float32)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_and_load_roundtrip(self):
        engine = XGBEngine(n_estimators=10, random_state=42)
        engine.fit(self.X, self.y)
        path = os.path.join(self.tmpdir, "xgb.joblib")
        engine.save(path)

        loaded = XGBEngine.load(path)
        np.testing.assert_array_almost_equal(
            engine.predict_proba(self.X),
            loaded.predict_proba(self.X),
        )

    def test_save_unfitted_raises(self):
        engine = XGBEngine()
        path = os.path.join(self.tmpdir, "xgb.joblib")
        with self.assertRaises(RuntimeError):
            engine.save(path)


class LSTMModelPersistenceTests(unittest.TestCase):
    """Tests for LSTMModel.save / .load."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((50, 10, 6)).astype(np.float32)
        self.y = (rng.random(50) > 0.5).astype(np.float32)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_and_load_roundtrip(self):
        model = LSTMModel(
            input_size=6, seq_len=10, hidden_size=16,
            num_layers=1, epochs=2, patience=2, random_state=42,
            device="cpu",
        )
        model.fit(self.X, self.y)
        path = os.path.join(self.tmpdir, "lstm")
        model.save(path)

        loaded = LSTMModel.load(path, device="cpu")
        np.testing.assert_array_almost_equal(
            model.predict_proba(self.X),
            loaded.predict_proba(self.X),
            decimal=5,
        )

    def test_save_creates_pt_and_json_files(self):
        model = LSTMModel(
            input_size=6, seq_len=10, hidden_size=16,
            num_layers=1, epochs=2, patience=2, random_state=42,
            device="cpu",
        )
        model.fit(self.X, self.y)
        path = os.path.join(self.tmpdir, "lstm")
        model.save(path)

        self.assertTrue(os.path.isfile(path + ".pt"))
        self.assertTrue(os.path.isfile(path + ".json"))

    def test_save_unfitted_raises(self):
        model = LSTMModel(input_size=6, seq_len=10, device="cpu")
        path = os.path.join(self.tmpdir, "lstm")
        with self.assertRaises(RuntimeError):
            model.save(path)


class MetaLearnerPersistenceTests(unittest.TestCase):
    """Tests for MetaLearner.save / .load."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(42)
        self.lstm_probs = rng.random(100).astype(np.float64)
        self.xgb_probs = rng.random(100).astype(np.float64)
        self.y = (rng.random(100) > 0.5).astype(np.float32)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_and_load_roundtrip(self):
        meta = MetaLearner(C=1.0, random_state=42)
        meta.fit(self.lstm_probs, self.xgb_probs, self.y)
        path = os.path.join(self.tmpdir, "meta.joblib")
        meta.save(path)

        loaded = MetaLearner.load(path)
        np.testing.assert_array_almost_equal(
            meta.predict_proba(self.lstm_probs, self.xgb_probs),
            loaded.predict_proba(self.lstm_probs, self.xgb_probs),
        )

    def test_save_unfitted_raises(self):
        meta = MetaLearner()
        path = os.path.join(self.tmpdir, "meta.joblib")
        with self.assertRaises(RuntimeError):
            meta.save(path)


class TradePulsePredictorTests(unittest.TestCase):
    """Tests for the TradePulsePredictor inference engine."""

    @classmethod
    def setUpClass(cls):
        """Train a tiny model and save artifacts to a temp directory."""
        cls.tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(42)

        # Tabular features
        n_samples, n_features, seq_len = 200, 50, 10
        feature_names = [f"feature_{i:04d}" for i in range(n_features)]
        df_tab = pd.DataFrame(
            rng.standard_normal((n_samples, n_features)).astype(np.float32),
            columns=feature_names,
        )

        # Sequential tensor (OHLCV + daily_return + volatility = 7 channels)
        X_seq = rng.standard_normal((n_samples, seq_len, 7)).astype(np.float32)
        y = (rng.random(n_samples) > 0.5).astype(np.float32)

        # Fit & save preprocessor
        pp = FeaturePreprocessor(pca_variance_threshold=0.95, random_state=42)
        tab_pca = pp.fit_transform(df_tab)
        pp.save(os.path.join(cls.tmpdir, "preprocessor.joblib"))

        # Fit & save XGBoost
        xgb = XGBEngine(n_estimators=10, random_state=42)
        xgb.fit(tab_pca, y)
        xgb.save(os.path.join(cls.tmpdir, "xgb_engine.joblib"))

        # Fit & save LSTM
        lstm = LSTMModel(
            input_size=7, seq_len=seq_len, hidden_size=16,
            num_layers=1, epochs=2, patience=2, random_state=42,
            device="cpu",
        )
        lstm.fit(X_seq, y)
        lstm.save(os.path.join(cls.tmpdir, "lstm_model"))

        # Fit & save MetaLearner
        lstm_probs = lstm.predict_proba(X_seq)
        xgb_probs = xgb.predict_proba(tab_pca)
        meta = MetaLearner(C=1.0, random_state=42)
        meta.fit(lstm_probs, xgb_probs, y)
        meta.save(os.path.join(cls.tmpdir, "meta_learner.joblib"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_predictor_loads_successfully(self):
        predictor = TradePulsePredictor(model_dir=self.tmpdir)
        self.assertIsNotNone(predictor.preprocessor)
        self.assertIsNotNone(predictor.xgb_engine)
        self.assertIsNotNone(predictor.lstm_model)
        self.assertIsNotNone(predictor.meta_learner)

    def test_predict_signal_returns_float_in_range(self):
        predictor = TradePulsePredictor(model_dir=self.tmpdir)

        # Build a mock OHLCV DataFrame with enough rows.
        # After pct_change() + rolling(window=20), the first 20 rows become
        # NaN and are dropped; we need at least seq_len rows remaining.
        # seq_len=10 → supply 10 + 20 = 30 rows.
        rng = np.random.default_rng(99)
        hist_df = pd.DataFrame({
            "open": rng.uniform(100, 200, 30),
            "high": rng.uniform(100, 200, 30),
            "low": rng.uniform(100, 200, 30),
            "close": rng.uniform(100, 200, 30),
            "volume": rng.integers(1000, 50000, 30),
        })

        prob = predictor.predict_signal(hist_df, sentiment_score=0.0)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_predict_signal_raises_on_insufficient_rows(self):
        predictor = TradePulsePredictor(model_dir=self.tmpdir)
        hist_df = pd.DataFrame({
            "open": [100.0], "high": [110.0], "low": [90.0],
            "close": [105.0], "volume": [1000],
        })
        with self.assertRaises(ValueError):
            predictor.predict_signal(hist_df)

    def test_predict_signal_raises_on_missing_columns(self):
        predictor = TradePulsePredictor(model_dir=self.tmpdir)
        hist_df = pd.DataFrame({"open": [100.0] * 30})
        with self.assertRaises(ValueError):
            predictor.predict_signal(hist_df)

    def test_predictor_load_fails_on_missing_dir(self):
        with self.assertRaises(Exception):
            TradePulsePredictor(model_dir="/nonexistent/path")


if __name__ == "__main__":
    unittest.main()
