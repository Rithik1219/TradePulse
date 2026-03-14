"""
data_ingestion/angel_one_api.py
===============================
Angel One SmartAPI integration for TradePulse — Phase 2 / Phase 5.

Responsibilities
----------------
1. **Secure Authentication** — Load credentials from a ``.env`` file via
   ``python-dotenv``, generate the time-based OTP with ``pyotp``, and
   authenticate through the official Angel One ``SmartApi`` client.
2. **Data Extraction** — Retrieve the user's live *holdings* and
   *positions* from Angel One.
3. **Data Formatting** — Merge both datasets into a single, clean
   ``pandas.DataFrame`` ready for downstream ML pipelines.
4. **Historical OHLCV Data** — Fetch candlestick (OHLCV) data for a
   given symbol using the ``getCandleData`` endpoint, returning a
   ``pandas.DataFrame`` shaped for the LSTM/XGBoost training pipeline.

Environment Variables Required
------------------------------
``ANGEL_API_KEY``      — SmartAPI application key.
``ANGEL_CLIENT_ID``    — Angel One client / login ID.
``ANGEL_PIN``          — Account PIN / password.
``ANGEL_TOTP_SECRET``  — Base-32 TOTP secret for 2FA.

Usage
-----
::

    from data_ingestion import AngelOneClient

    client = AngelOneClient()
    client.login()
    df = client.get_portfolio()
    print(df)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import pyotp
import requests
from dotenv import load_dotenv
from SmartApi.smartConnect import SmartConnect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HOLDINGS_COLUMNS = {
    "tradingsymbol": "symbol",
    "quantity": "quantity",
    "averageprice": "avg_price",
}

_POSITIONS_COLUMNS = {
    "tradingsymbol": "symbol",
    "netqty": "quantity",
    "averageprice": "avg_price",
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AngelOneClient:
    """Wrapper around the Angel One SmartAPI for portfolio retrieval.

    Parameters
    ----------
    env_path : str or None, optional
        Explicit path to the ``.env`` file.  When *None* (the default)
        ``python-dotenv`` searches upward from the working directory.

    Attributes
    ----------
    smart_api : SmartConnect
        The underlying SmartAPI connection object (available after login).
    auth_token : str or None
        JWT auth token returned on successful login.
    """

    def __init__(self, env_path: Optional[str] = None) -> None:
        load_dotenv(dotenv_path=env_path)

        self._api_key: str = self._require_env("ANGEL_API_KEY")
        self._client_id: str = self._require_env("ANGEL_CLIENT_ID")
        self._pin: str = self._require_env("ANGEL_PIN")
        self._totp_secret: str = self._require_env("ANGEL_TOTP_SECRET")

        self.smart_api: Optional[SmartConnect] = None
        self.auth_token: Optional[str] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_env(name: str) -> str:
        """Return the value of an environment variable or raise."""
        value = os.getenv(name)
        if not value:
            raise EnvironmentError(
                f"Missing required environment variable: {name}. "
                "Please set it in your .env file."
            )
        return value

    @staticmethod
    def _generate_totp(secret: str) -> str:
        """Generate a time-based OTP from the TOTP secret."""
        totp = pyotp.TOTP(secret)
        return totp.now()

    @staticmethod
    def _normalise_records(
        raw: Any,
        column_map: Dict[str, str],
        source_label: str,
    ) -> List[Dict[str, Any]]:
        """Extract and rename fields from API response records.

        Parameters
        ----------
        raw : list[dict] or None
            Raw records returned by the SmartAPI.
        column_map : dict
            Mapping of *API field name* → *desired column name*.
        source_label : str
            Human-readable label (``"holdings"`` or ``"positions"``)
            added as the ``source`` column in each record.

        Returns
        -------
        list[dict]
        """
        if not raw:
            logger.info("No %s data returned by the API.", source_label)
            return []

        records: List[Dict[str, Any]] = []
        for item in raw:
            record = {
                target: item.get(api_key)
                for api_key, target in column_map.items()
            }
            record["source"] = source_label
            records.append(record)

        logger.info(
            "Extracted %d record(s) from %s.", len(records), source_label
        )
        return records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def login(self) -> None:
        """Authenticate with Angel One and store the session token.

        Raises
        ------
        ConnectionError
            If the API returns an error or the login payload is empty.
        """
        totp_value = self._generate_totp(self._totp_secret)

        self.smart_api = SmartConnect(api_key=self._api_key)

        data = self.smart_api.generateSession(
            clientCode=self._client_id,
            password=self._pin,
            totp=totp_value,
        )

        if not data or not data.get("status"):
            message = (data or {}).get("message", "Unknown authentication error")
            raise ConnectionError(f"Angel One login failed: {message}")

        self.auth_token = data["data"]["jwtToken"]
        logger.info(
            "Authenticated successfully. Client ID: %s", self._client_id
        )

    def get_portfolio(self) -> pd.DataFrame:
        """Fetch holdings and positions, returning a combined DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``symbol``, ``quantity``, ``avg_price``, ``source``.
            ``source`` is either ``"holdings"`` or ``"positions"``.

        Raises
        ------
        RuntimeError
            If called before a successful :py:meth:`login`.
        """
        if self.smart_api is None or self.auth_token is None:
            raise RuntimeError(
                "Not authenticated. Call login() before get_portfolio()."
            )

        # --- Holdings ---------------------------------------------------
        holdings_resp = self.smart_api.holding()
        holdings_data = (holdings_resp or {}).get("data", None)
        holdings_records = self._normalise_records(
            holdings_data, _HOLDINGS_COLUMNS, "holdings"
        )

        # --- Positions --------------------------------------------------
        positions_resp = self.smart_api.position()
        positions_data = (positions_resp or {}).get("data", None)
        positions_records = self._normalise_records(
            positions_data, _POSITIONS_COLUMNS, "positions"
        )

        # --- Combine ----------------------------------------------------
        all_records = holdings_records + positions_records
        if not all_records:
            logger.warning(
                "Portfolio is empty — no holdings or positions found."
            )
            return pd.DataFrame(
                columns=["symbol", "quantity", "avg_price", "source"]
            )

        df = pd.DataFrame(all_records)

        # Convert quantity — log a warning when values cannot be parsed.
        qty_numeric = pd.to_numeric(df["quantity"], errors="coerce")
        bad_qty = qty_numeric.isna() & df["quantity"].notna()
        if bad_qty.any():
            logger.warning(
                "Coerced %d non-numeric quantity value(s) to 0.", bad_qty.sum()
            )
        df["quantity"] = qty_numeric.fillna(0).astype(int)

        # Convert avg_price — same safeguard.
        price_numeric = pd.to_numeric(df["avg_price"], errors="coerce")
        bad_price = price_numeric.isna() & df["avg_price"].notna()
        if bad_price.any():
            logger.warning(
                "Coerced %d non-numeric avg_price value(s) to 0.0.",
                bad_price.sum(),
            )
        df["avg_price"] = price_numeric.fillna(0.0)

        logger.info("Portfolio DataFrame built — %d rows.", len(df))
        return df

    def get_historical_data(
        self,
        symbol: str,
        token: str,
        exchange: str = "NSE",
        interval: str = "ONE_DAY",
        from_date: str = "",
        to_date: str = "",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV candlestick data for a symbol.

        Uses the Angel One SmartAPI ``getCandleData`` endpoint to retrieve
        OHLCV bars for the requested time range.  The returned
        :class:`pandas.DataFrame` is shaped to match what the ML pipeline
        (``FeaturePreprocessor`` / ``LSTMModel``) expects:

        Columns: ``timestamp``, ``open``, ``high``, ``low``, ``close``,
        ``volume``.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g. ``"TCS"``).  Used only for logging.
        token : str
            Angel One instrument token for the symbol (e.g. ``"11536"``).
        exchange : str, optional
            Exchange code.  Defaults to ``"NSE"``.
        interval : str, optional
            Candle interval as accepted by the API.  Common values:
            ``"ONE_MINUTE"``, ``"FIVE_MINUTE"``, ``"ONE_DAY"``.
            Defaults to ``"ONE_DAY"``.
        from_date : str
            Start of the requested window in the format expected by the
            SmartAPI (``"YYYY-MM-DD HH:MM"`` for intraday, or
            ``"YYYY-MM-DD"`` for daily).
        to_date : str
            End of the requested window (same format as ``from_date``).

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp`` (datetime), ``open``, ``high``,
            ``low``, ``close`` (float64), ``volume`` (int64).
            An **empty** DataFrame with these columns is returned when the
            API reports no data for the requested range.

        Raises
        ------
        RuntimeError
            If called before a successful :py:meth:`login`.
        requests.exceptions.Timeout
            Re-raised when the underlying HTTP request times out so the
            caller can implement its own retry / back-off strategy.
        """
        if self.smart_api is None or self.auth_token is None:
            raise RuntimeError(
                "Not authenticated. Call login() before get_historical_data()."
            )

        _OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
        _EMPTY_DF = pd.DataFrame(columns=_OHLCV_COLUMNS)

        historic_param = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date,
        }

        logger.info(
            "Fetching historical data for %s (%s) from %s to %s [interval=%s].",
            symbol,
            token,
            from_date,
            to_date,
            interval,
        )

        try:
            response = self.smart_api.getCandleData(historic_param)
        except requests.exceptions.Timeout:
            logger.error(
                "Historical data request for %s timed out. "
                "Consider retrying with a shorter date range.",
                symbol,
            )
            raise
        except Exception:
            logger.exception(
                "Unexpected error fetching historical data for %s.", symbol
            )
            return _EMPTY_DF

        if not response or not response.get("status"):
            message = (response or {}).get("message", "Unknown error")
            logger.warning(
                "Historical data API returned an error for %s: %s", symbol, message
            )
            return _EMPTY_DF

        candles: List[Any] = (response.get("data") or [])
        if not candles:
            logger.info("No candle data returned for %s in the requested range.", symbol)
            return _EMPTY_DF

        df = pd.DataFrame(candles, columns=_OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Keep NaN for unparseable price values so downstream consumers can
        # detect and handle bad ticks rather than silently treating them as 0.
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = (
            pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        )

        logger.info(
            "Historical DataFrame for %s built — %d rows.", symbol, len(df)
        )
        return df


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    client = AngelOneClient()
    client.login()
    portfolio_df = client.get_portfolio()
    print(portfolio_df.to_string(index=False))
