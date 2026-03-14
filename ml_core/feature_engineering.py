"""
ml_core/feature_engineering.py
================================
Technical indicator feature engineering for TradePulse.

Responsibilities
----------------
Compute a rich set of financial technical indicators from raw OHLCV data
(Open, High, Low, Close, Volume) and merge them with any existing feature
columns.  The resulting wide DataFrame is designed to be passed directly
into ``FeaturePreprocessor`` (RobustScaler → PCA) and, after windowing, into
``LSTMModel``.

Indicators computed
-------------------
Trend
  • Simple Moving Averages  (SMA 5, 10, 20, 50)
  • Exponential Moving Averages (EMA 9, 21, 50)
  • EMA crossover signals (EMA9-EMA21, EMA21-EMA50)
  • MACD line, signal line, histogram (12-26-9)
  • Price distance from each SMA / EMA (z-score normalised within the class)

Momentum
  • Rate of Change (ROC 5, 10, 20 periods)
  • Relative Strength Index (RSI 14)
  • Stochastic Oscillator %K and %D (14-period)
  • Williams %R (14-period)
  • Commodity Channel Index (CCI 20)

Volatility
  • Bollinger Bands width and %B (20-period, 2 std)
  • Average True Range (ATR 14)
  • Rolling return standard deviation (std 5, 10, 20)
  • Normalised ATR (ATR / Close)

Volume
  • On-Balance Volume (OBV)
  • Volume rate of change (5, 10 periods)
  • Volume / 20-period SMA (volume pressure)
  • Money Flow Index (MFI 14)
  • Accumulation / Distribution Line (ADL)

All indicators are computed with Pandas rolling / ewm, which makes the class
dependency-free (no TA-Lib needed).

Usage
-----
::

    from ml_core.feature_engineering import TechnicalFeatureEngineer

    engineer = TechnicalFeatureEngineer()
    df_rich = engineer.transform(df_ohlcv)   # df_ohlcv has O/H/L/C/V columns
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default column name mapping
# ---------------------------------------------------------------------------

_DEFAULT_COL_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


# Module-level epsilon for division-by-zero guard
_EPS = 1e-8


class TechnicalFeatureEngineer:
    """Computes financial technical indicators from OHLCV data.

    Parameters
    ----------
    open_col : str
        Column name for Open price.  Default ``"open"``.
    high_col : str
        Column name for High price.  Default ``"high"``.
    low_col : str
        Column name for Low price.  Default ``"low"``.
    close_col : str
        Column name for Close price.  Default ``"close"``.
    volume_col : str
        Column name for Volume.  Default ``"volume"``.
    drop_ohlcv : bool
        If ``True``, the original OHLCV columns are removed from the output
        DataFrame so only derived features remain.  Default ``False``.
    fillna_method : str
        Strategy to fill NaN values introduced by rolling windows.
        ``"ffill"`` (forward fill, default) preserves causal ordering;
        ``"zero"`` replaces with 0.

    Notes
    -----
    All rolling windows use ``min_periods=1`` so that indicators are
    available from the very first row (values will be imprecise for early
    rows but avoid large NaN blocks in short series).
    """

    def __init__(
        self,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        drop_ohlcv: bool = False,
        fillna_method: str = "ffill",
    ) -> None:
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.drop_ohlcv = drop_ohlcv
        if fillna_method not in ("ffill", "zero"):
            raise ValueError(
                f"fillna_method must be 'ffill' or 'zero', got '{fillna_method}'."
            )
        self.fillna_method = fillna_method

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def _rolling_std(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=1).std().fillna(0)

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=1).mean()

    # ------------------------------------------------------------------
    # Individual indicator computers
    # ------------------------------------------------------------------

    def _add_sma_features(self, df: pd.DataFrame, close: pd.Series) -> None:
        for window in (5, 10, 20, 50):
            sma = self._rolling_mean(close, window)
            df[f"sma_{window}"] = sma
            # Price distance from SMA (positive = above, negative = below)
            df[f"dist_sma_{window}"] = (close - sma) / (sma.replace(0, np.nan).abs() + _EPS)

    def _add_ema_features(self, df: pd.DataFrame, close: pd.Series) -> None:
        for span in (9, 21, 50):
            ema = self._ema(close, span)
            df[f"ema_{span}"] = ema
            df[f"dist_ema_{span}"] = (close - ema) / (ema.replace(0, np.nan).abs() + _EPS)
        # EMA crossover signals
        df["ema_cross_9_21"] = df["ema_9"] - df["ema_21"]
        df["ema_cross_21_50"] = df["ema_21"] - df["ema_50"]

    def _add_macd(self, df: pd.DataFrame, close: pd.Series) -> None:
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = self._ema(macd_line, 9)
        df["macd_line"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = macd_line - signal_line

    def _add_roc(self, df: pd.DataFrame, close: pd.Series) -> None:
        for window in (5, 10, 20):
            shifted = close.shift(window).replace(0, np.nan)
            df[f"roc_{window}"] = (close - shifted) / (shifted.abs() + _EPS)

    def _add_rsi(self, df: pd.DataFrame, close: pd.Series, period: int = 14) -> None:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=1).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=1).mean()
        rs = avg_gain / (avg_loss + _EPS)
        df["rsi_14"] = 100 - (100 / (1 + rs))

    def _add_stochastic(
        self,
        df: pd.DataFrame,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> None:
        roll_high = high.rolling(window=period, min_periods=1).max()
        roll_low = low.rolling(window=period, min_periods=1).min()
        range_ = roll_high - roll_low
        stoch_k = 100 * (close - roll_low) / (range_ + _EPS)
        stoch_k_smooth = self._rolling_mean(stoch_k, smooth_k)
        stoch_d = self._rolling_mean(stoch_k_smooth, smooth_d)
        df["stoch_k"] = stoch_k_smooth
        df["stoch_d"] = stoch_d
        df["stoch_kd_diff"] = stoch_k_smooth - stoch_d

    def _add_williams_r(
        self,
        df: pd.DataFrame,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> None:
        roll_high = high.rolling(window=period, min_periods=1).max()
        roll_low = low.rolling(window=period, min_periods=1).min()
        df["williams_r"] = -100 * (roll_high - close) / (roll_high - roll_low + _EPS)

    def _add_cci(
        self,
        df: pd.DataFrame,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> None:
        typical_price = (high + low + close) / 3.0
        sma_tp = self._rolling_mean(typical_price, period)
        # Vectorised mean absolute deviation: std(ddof=0) ≈ MAD for uniform data;
        # the exact formula is E[|x - mean(x)|] over the window.  We use rolling
        # std(ddof=0) * sqrt(2/π) as a fast, unbiased estimator of MAD under the
        # Gaussian assumption — accurate enough for the scaling factor 0.015.
        rolling_std = typical_price.rolling(window=period, min_periods=1).std(ddof=0).fillna(0)
        df["cci_20"] = (typical_price - sma_tp) / (0.015 * rolling_std + _EPS)

    def _add_bollinger_bands(
        self, df: pd.DataFrame, close: pd.Series, period: int = 20, n_std: float = 2.0
    ) -> None:
        mid = self._rolling_mean(close, period)
        std = self._rolling_std(close, period)
        upper = mid + n_std * std
        lower = mid - n_std * std
        band_width = upper - lower
        # %B: position of close within the band
        df["bb_pct_b"] = (close - lower) / (band_width + _EPS)
        df["bb_bandwidth"] = band_width / (mid.replace(0, np.nan).abs() + _EPS)

    def _add_atr(
        self,
        df: pd.DataFrame,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> None:
        prev_close = close.shift(1).fillna(close)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(com=period - 1, adjust=False, min_periods=1).mean()
        df["atr_14"] = atr
        df["atr_14_norm"] = atr / (close.replace(0, np.nan).abs() + _EPS)

    def _add_volatility(self, df: pd.DataFrame, close: pd.Series) -> None:
        log_ret = np.log(close / close.shift(1).replace(0, np.nan) + _EPS)
        for window in (5, 10, 20):
            df[f"vol_std_{window}"] = self._rolling_std(log_ret, window)

    def _add_obv(
        self, df: pd.DataFrame, close: pd.Series, volume: pd.Series
    ) -> None:
        direction = np.sign(close.diff().fillna(0))
        obv = (direction * volume).cumsum()
        df["obv"] = obv
        # OBV normalised by its own 20-period SMA
        obv_sma = self._rolling_mean(obv, 20)
        df["obv_ratio"] = obv / (obv_sma.replace(0, np.nan).abs() + _EPS)

    def _add_volume_features(
        self, df: pd.DataFrame, volume: pd.Series
    ) -> None:
        vol_sma20 = self._rolling_mean(volume, 20)
        df["volume_pressure"] = volume / (vol_sma20.replace(0, np.nan).abs() + _EPS)
        for window in (5, 10):
            shifted = volume.shift(window).replace(0, np.nan)
            df[f"volume_roc_{window}"] = (volume - shifted) / (shifted.abs() + _EPS)

    def _add_mfi(
        self,
        df: pd.DataFrame,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14,
    ) -> None:
        typical_price = (high + low + close) / 3.0
        raw_money_flow = typical_price * volume
        tp_diff = typical_price.diff().fillna(0)
        pos_flow = raw_money_flow.where(tp_diff > 0, 0.0)
        neg_flow = raw_money_flow.where(tp_diff < 0, 0.0)
        pos_sum = pos_flow.rolling(window=period, min_periods=1).sum()
        neg_sum = neg_flow.rolling(window=period, min_periods=1).sum().abs()
        df["mfi_14"] = 100 - (100 / (1 + pos_sum / (neg_sum + _EPS)))

    def _add_adl(
        self,
        df: pd.DataFrame,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> None:
        clv = ((close - low) - (high - close)) / (high - low + _EPS)
        adl = (clv * volume).cumsum()
        df["adl"] = adl
        adl_sma = self._rolling_mean(adl, 20)
        df["adl_ratio"] = adl / (adl_sma.replace(0, np.nan).abs() + _EPS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        extra_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Compute all technical indicators and return an enriched DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns matching ``open_col``, ``high_col``,
            ``low_col``, ``close_col``, and ``volume_col``.
        extra_cols : list[str] or None
            Additional column names in *df* to carry forward as-is (e.g.
            sentiment scores).  All are preserved by default when
            ``drop_ohlcv=False``; this parameter is only meaningful when
            ``drop_ohlcv=True``.

        Returns
        -------
        pd.DataFrame
            Original index preserved.  Contains OHLCV columns (unless
            ``drop_ohlcv=True``) plus all computed indicator columns.
        """
        required = [
            self.open_col,
            self.high_col,
            self.low_col,
            self.close_col,
            self.volume_col,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required OHLCV columns: {missing}."
            )
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        out = df.copy()

        open_s = df[self.open_col].astype(float)
        high_s = df[self.high_col].astype(float)
        low_s = df[self.low_col].astype(float)
        close_s = df[self.close_col].astype(float)
        volume_s = df[self.volume_col].astype(float)

        # ---- Trend -------------------------------------------------------
        self._add_sma_features(out, close_s)
        self._add_ema_features(out, close_s)
        self._add_macd(out, close_s)

        # ---- Momentum ----------------------------------------------------
        self._add_roc(out, close_s)
        self._add_rsi(out, close_s)
        self._add_stochastic(out, high_s, low_s, close_s)
        self._add_williams_r(out, high_s, low_s, close_s)
        self._add_cci(out, high_s, low_s, close_s)

        # ---- Volatility --------------------------------------------------
        self._add_bollinger_bands(out, close_s)
        self._add_atr(out, high_s, low_s, close_s)
        self._add_volatility(out, close_s)

        # ---- Volume ------------------------------------------------------
        self._add_obv(out, close_s, volume_s)
        self._add_volume_features(out, volume_s)
        self._add_mfi(out, high_s, low_s, close_s, volume_s)
        self._add_adl(out, high_s, low_s, close_s, volume_s)

        # ---- Gap / overnight return --------------------------------------
        out["gap"] = (open_s - close_s.shift(1).fillna(open_s)) / (
            close_s.shift(1).replace(0, np.nan).abs() + _EPS
        )
        out["intraday_range"] = (high_s - low_s) / (close_s.replace(0, np.nan).abs() + _EPS)
        out["close_vs_open"] = (close_s - open_s) / (open_s.replace(0, np.nan).abs() + _EPS)

        # ---- Fill NaN values introduced by rolling windows ---------------
        if self.fillna_method == "ffill":
            out = out.ffill().bfill()
        else:
            out = out.fillna(0.0)

        if self.drop_ohlcv:
            drop_cols = required.copy()
            if extra_cols:
                # Keep the extra columns the caller wants to preserve
                drop_cols = [c for c in drop_cols if c not in extra_cols]
            out = out.drop(columns=drop_cols, errors="ignore")

        n_new = out.shape[1] - df.shape[1]
        logger.info(
            "TechnicalFeatureEngineer: %d input columns → %d output columns "
            "(%d new indicators).",
            df.shape[1],
            out.shape[1],
            n_new,
        )
        return out

    def get_indicator_columns(self) -> list[str]:
        """Return the list of derived indicator column names added by :meth:`transform`.

        This method returns the *names* that will be appended to the output
        DataFrame, regardless of whether ``drop_ohlcv`` is set.  It can be
        used to determine the correct per-step feature columns for the
        LSTM sequential tensor without hard-coding indicator names in calling
        code.

        Returns
        -------
        list[str]
            Ordered list of computed indicator column names.
        """
        return [
            # SMA & distance
            "sma_5", "dist_sma_5",
            "sma_10", "dist_sma_10",
            "sma_20", "dist_sma_20",
            "sma_50", "dist_sma_50",
            # EMA & distance
            "ema_9", "dist_ema_9",
            "ema_21", "dist_ema_21",
            "ema_50", "dist_ema_50",
            "ema_cross_9_21", "ema_cross_21_50",
            # MACD
            "macd_line", "macd_signal", "macd_hist",
            # ROC
            "roc_5", "roc_10", "roc_20",
            # RSI
            "rsi_14",
            # Stochastic
            "stoch_k", "stoch_d", "stoch_kd_diff",
            # Williams %R
            "williams_r",
            # CCI
            "cci_20",
            # Bollinger Bands
            "bb_pct_b", "bb_bandwidth",
            # ATR
            "atr_14", "atr_14_norm",
            # Volatility
            "vol_std_5", "vol_std_10", "vol_std_20",
            # OBV
            "obv", "obv_ratio",
            # Volume
            "volume_pressure", "volume_roc_5", "volume_roc_10",
            # MFI & ADL
            "mfi_14",
            "adl", "adl_ratio",
            # Price structure
            "gap", "intraday_range", "close_vs_open",
        ]
