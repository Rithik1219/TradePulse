from unittest.mock import MagicMock, patch
import json

import pandas as pd
from django.core.cache import cache
from django.test import TestCase, RequestFactory
from django.utils import timezone

from dashboard.models import PortfolioSnapshot
from dashboard.views import portfolio_view, news_predictions_view

class PortfolioViewTests(TestCase):
    """Tests for the portfolio_view."""

    def setUp(self):
        self.factory = RequestFactory()


    @patch("dashboard.views.AngelOneClient", create=True)
    def test_portfolio_view_renders_with_data(self, mock_import):
        """View returns portfolio data as JSON when API succeeds."""
        # We need to patch the import inside the function
        sample_df = pd.DataFrame([
            {"symbol": "TCS", "quantity": 10, "avg_price": 3500.0, "source": "holdings"},
            {"symbol": "INFY", "quantity": 5, "avg_price": 1500.0, "source": "positions"},
        ])

        with patch("dashboard.views.AngelOneClient", create=True) as MockClient:
            mock_instance = MagicMock()
            mock_instance.get_portfolio.return_value = sample_df
            MockClient.return_value = mock_instance

            # Patch the import inside the view
            import importlib
            with patch.dict("sys.modules", {
                "data_ingestion": MagicMock(),
                "data_ingestion.angel_one_api": MagicMock(AngelOneClient=MockClient),
            }):
                request = self.factory.get("/")
                response = portfolio_view(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")
        data = json.loads(response.content)
        symbols = [r["symbol"] for r in data["portfolio"]]
        self.assertIn("TCS", symbols)
        self.assertIn("INFY", symbols)
        tcs = next(r for r in data["portfolio"] if r["symbol"] == "TCS")
        self.assertEqual(tcs["avg_price"], 3500.0)
        self.assertEqual(tcs["source"], "holdings")

    def test_portfolio_view_handles_api_error(self):
        """View returns JSON error message when API fails."""
        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(
                AngelOneClient=MagicMock(side_effect=ConnectionError("API down"))
            ),
        }):
            request = self.factory.get("/")
            response = portfolio_view(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")
        data = json.loads(response.content)
        self.assertIn("Unable to retrieve portfolio data", data["error_message"])

    def test_portfolio_view_returns_json_response(self):
        """View returns a JSON response with the expected top-level keys."""
        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(
                AngelOneClient=MagicMock(side_effect=Exception("test"))
            ),
        }):
            request = self.factory.get("/")
            response = portfolio_view(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")
        data = json.loads(response.content)
        for key in ("portfolio", "error_message", "chart_labels", "chart_values",
                    "sector_labels", "sector_values", "historical_charts_json"):
            self.assertIn(key, data)


class NewsPredictionsViewTests(TestCase):
    """Tests for the /api/news-predictions/ endpoint."""

    def setUp(self):
        self.factory = RequestFactory()
        cache.clear()

    def tearDown(self):
        cache.clear()

    @patch("dashboard.views.run_local_news_prediction_pipeline")
    def test_news_predictions_response_shape(self, mock_pipeline):
        mock_pipeline.return_value = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "tickers": ["RELIANCE.NS"],
            "headlines": [],
            "sentiment": {
                "aggregate": {
                    "bullish": 0.2,
                    "bearish": 0.1,
                    "neutral": 0.7,
                    "headline_count": 0,
                },
                "per_headline": [],
            },
            "market_summary": "Neutral outlook.",
            "meta": {"engine": "local_finbert_plus_mistral"},
        }

        response = news_predictions_view(self.factory.get("/api/news-predictions/"))
        data = json.loads(response.content)

        self.assertEqual(response.status_code, 200)
        self.assertFalse(data["cache"]["hit"])
        self.assertIn("tickers", data)
        self.assertIn("sentiment", data)
        self.assertIn("market_summary", data)

    @patch("dashboard.views.run_local_news_prediction_pipeline")
    def test_news_predictions_uses_cache(self, mock_pipeline):
        mock_pipeline.return_value = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "tickers": ["TCS.NS"],
            "headlines": [],
            "sentiment": {
                "aggregate": {
                    "bullish": 0.5,
                    "bearish": 0.3,
                    "neutral": 0.2,
                    "headline_count": 0,
                },
                "per_headline": [],
            },
            "market_summary": "Summary.",
            "meta": {"engine": "local_finbert_plus_mistral"},
        }

        first = news_predictions_view(self.factory.get("/api/news-predictions/"))
        second = news_predictions_view(self.factory.get("/api/news-predictions/"))
        first_data = json.loads(first.content)
        second_data = json.loads(second.content)

        self.assertFalse(first_data["cache"]["hit"])
        self.assertTrue(second_data["cache"]["hit"])
        self.assertEqual(mock_pipeline.call_count, 1)


class PortfolioSnapshotModelTests(TestCase):
    """Tests for the PortfolioSnapshot model."""

    def test_str_representation(self):
        """__str__ returns symbol and date."""
        snapshot = PortfolioSnapshot.objects.create(
            symbol="RELIANCE",
            quantity=3,
            avg_price=2800.0,
            source="holdings",
        )
        expected = f"RELIANCE – {snapshot.timestamp.date()}"
        self.assertEqual(str(snapshot), expected)

    def test_snapshot_fields_stored_correctly(self):
        """All fields are persisted as expected."""
        snapshot = PortfolioSnapshot.objects.create(
            symbol="TCS",
            quantity=10,
            avg_price=3500.0,
            source="holdings",
        )
        fetched = PortfolioSnapshot.objects.get(pk=snapshot.pk)
        self.assertEqual(fetched.symbol, "TCS")
        self.assertEqual(fetched.quantity, 10)
        self.assertAlmostEqual(fetched.avg_price, 3500.0)
        self.assertEqual(fetched.source, "holdings")
        self.assertIsNotNone(fetched.timestamp)


class PortfolioSnapshotDedupTests(TestCase):
    """Tests for the daily deduplication logic in portfolio_view."""

    def _make_mock_client(self, df):
        MockClient = MagicMock()
        mock_instance = MagicMock()
        mock_instance.get_portfolio.return_value = df
        MockClient.return_value = mock_instance
        return MockClient

    def test_snapshot_saved_on_first_visit(self):
        """Snapshots are created when none exist for today."""
        sample_df = pd.DataFrame([
            {"symbol": "TCS", "quantity": 10, "avg_price": 3500.0, "source": "holdings"},
        ])
        MockClient = self._make_mock_client(sample_df)

        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(AngelOneClient=MockClient),
        }):
            request = RequestFactory().get("/")
            portfolio_view(request)

        self.assertEqual(PortfolioSnapshot.objects.filter(symbol="TCS").count(), 1)

    def test_snapshot_not_duplicated_on_second_visit(self):
        """A second page load on the same day does not create duplicate snapshots."""
        sample_df = pd.DataFrame([
            {"symbol": "INFY", "quantity": 5, "avg_price": 1500.0, "source": "positions"},
        ])
        MockClient = self._make_mock_client(sample_df)

        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(AngelOneClient=MockClient),
        }):
            rf = RequestFactory()
            portfolio_view(rf.get("/"))
            portfolio_view(rf.get("/"))

        self.assertEqual(PortfolioSnapshot.objects.filter(symbol="INFY").count(), 1)

    def test_snapshot_created_for_new_day(self):
        """A snapshot is created for a symbol if the existing record is from a previous day."""
        yesterday = timezone.now() - timezone.timedelta(days=1)
        PortfolioSnapshot.objects.create(
            symbol="WIPRO",
            quantity=20,
            avg_price=400.0,
            source="holdings",
        )
        # Backdate the existing snapshot to yesterday
        PortfolioSnapshot.objects.filter(symbol="WIPRO").update(timestamp=yesterday)

        sample_df = pd.DataFrame([
            {"symbol": "WIPRO", "quantity": 22, "avg_price": 410.0, "source": "holdings"},
        ])
        MockClient = self._make_mock_client(sample_df)

        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(AngelOneClient=MockClient),
        }):
            portfolio_view(RequestFactory().get("/"))

        self.assertEqual(PortfolioSnapshot.objects.filter(symbol="WIPRO").count(), 2)


class PortfolioChartDataTests(TestCase):
    """Tests for chart_labels and chart_values passed to portfolio template."""

    def test_chart_data_included_in_context(self):
        """chart_labels and chart_values are present and valid JSON strings in the response."""
        PortfolioSnapshot.objects.create(
            symbol="TCS", quantity=10, avg_price=3500.0, source="holdings"
        )
        PortfolioSnapshot.objects.create(
            symbol="INFY", quantity=5, avg_price=1500.0, source="positions"
        )

        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(
                AngelOneClient=MagicMock(side_effect=Exception("no api"))
            ),
        }):
            request = RequestFactory().get("/")
            response = portfolio_view(request)

        self.assertEqual(response["Content-Type"], "application/json")
        data = json.loads(response.content)
        self.assertIn("chart_labels", data)
        self.assertIn("chart_values", data)
        # chart_labels is a JSON-encoded string; parse it and verify today's date is present
        labels = json.loads(data["chart_labels"])
        today_str = timezone.now().date().strftime("%Y-%m-%d")
        self.assertIn(today_str, labels)

    def test_chart_values_reflect_total_portfolio_value(self):
        """chart_values contains sum of quantity*avg_price for each snapshot day."""
        PortfolioSnapshot.objects.create(
            symbol="TCS", quantity=10, avg_price=3500.0, source="holdings"
        )
        PortfolioSnapshot.objects.create(
            symbol="INFY", quantity=5, avg_price=1500.0, source="positions"
        )
        # Expected total: 10*3500 + 5*1500 = 35000 + 7500 = 42500

        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(
                AngelOneClient=MagicMock(side_effect=Exception("no api"))
            ),
        }):
            request = RequestFactory().get("/")
            response = portfolio_view(request)

        data = json.loads(response.content)
        parsed_values = json.loads(data["chart_values"])
        self.assertEqual(parsed_values, [42500.0])

    def test_chart_data_empty_when_no_snapshots(self):
        """chart_labels and chart_values are empty JSON arrays when no snapshots exist."""
        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(
                AngelOneClient=MagicMock(side_effect=Exception("no api"))
            ),
        }):
            request = RequestFactory().get("/")
            response = portfolio_view(request)

        data = json.loads(response.content)
        self.assertEqual(json.loads(data["chart_labels"]), [])
        self.assertEqual(json.loads(data["chart_values"]), [])


class AiSignalTests(TestCase):
    """Tests for the ai_signal placeholder added to each portfolio row."""

    def _make_mock_client(self, df):
        MockClient = MagicMock()
        mock_instance = MagicMock()
        mock_instance.get_portfolio.return_value = df
        MockClient.return_value = mock_instance
        return MockClient

    def test_ai_signal_present_in_portfolio_records(self):
        """Each portfolio record in the JSON response includes an 'ai_signal' key."""
        sample_df = pd.DataFrame([
            {"symbol": "TCS", "quantity": 10, "avg_price": 3500.0, "source": "holdings"},
            {"symbol": "INFY", "quantity": 5, "avg_price": 1500.0, "source": "positions"},
        ])
        MockClient = self._make_mock_client(sample_df)

        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(AngelOneClient=MockClient),
        }):
            request = RequestFactory().get("/")
            response = portfolio_view(request)

        self.assertEqual(response["Content-Type"], "application/json")
        data = json.loads(response.content)
        for record in data["portfolio"]:
            self.assertIn("ai_signal", record)
            self.assertEqual(record["ai_signal"], "Analyzing...")

    def test_ai_signal_column_present_in_table_header(self):
        """The JSON response portfolio records include an 'ai_signal' field."""
        sample_df = pd.DataFrame([
            {"symbol": "TCS", "quantity": 10, "avg_price": 3500.0, "source": "holdings"},
        ])
        MockClient = self._make_mock_client(sample_df)

        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(AngelOneClient=MockClient),
        }):
            request = RequestFactory().get("/")
            response = portfolio_view(request)

        data = json.loads(response.content)
        self.assertTrue(len(data["portfolio"]) > 0)
        self.assertIn("ai_signal", data["portfolio"][0])

    def test_ai_signal_badge_rendered_for_each_row(self):
        """Each portfolio record in the JSON response has an 'ai_signal' value."""
        sample_df = pd.DataFrame([
            {"symbol": "TCS", "quantity": 10, "avg_price": 3500.0, "source": "holdings"},
            {"symbol": "INFY", "quantity": 5, "avg_price": 1500.0, "source": "positions"},
            {"symbol": "WIPRO", "quantity": 8, "avg_price": 420.0, "source": "holdings"},
        ])
        MockClient = self._make_mock_client(sample_df)

        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(AngelOneClient=MockClient),
        }):
            request = RequestFactory().get("/")
            response = portfolio_view(request)

        data = json.loads(response.content)
        analyzing_count = sum(
            1 for r in data["portfolio"] if r.get("ai_signal") == "Analyzing..."
        )
        self.assertEqual(analyzing_count, 3)


class GetHistoricalDataTests(TestCase):
    """Tests for AngelOneClient.get_historical_data."""

    def _make_client(self):
        """Return a partially-mocked AngelOneClient that is already 'logged in'."""
        smart_api_mock = MagicMock()
        smart_connect_mock = MagicMock(return_value=smart_api_mock)
        with patch.dict("sys.modules", {
            "SmartApi": MagicMock(),
            "SmartApi.smartConnect": MagicMock(SmartConnect=smart_connect_mock),
        }), patch.dict("os.environ", {
            "ANGEL_API_KEY": "key",
            "ANGEL_CLIENT_ID": "client",
            "ANGEL_PIN": "pin",
            "ANGEL_TOTP_SECRET": "JBSWY3DPEHPK3PXP",
        }):
            from importlib import import_module, reload
            import data_ingestion.angel_one_api as api_module
            reload(api_module)
            client = api_module.AngelOneClient()
        client.smart_api = MagicMock()
        client.auth_token = "fake_token"
        return client

    def test_returns_ohlcv_dataframe_on_success(self):
        """get_historical_data returns a DataFrame with OHLCV columns on success."""
        client = self._make_client()
        candle_data = [
            ["2024-01-01T09:15:00+05:30", 3500.0, 3550.0, 3490.0, 3520.0, 100000],
            ["2024-01-02T09:15:00+05:30", 3520.0, 3600.0, 3510.0, 3580.0, 120000],
        ]
        client.smart_api.getCandleData.return_value = {
            "status": True,
            "message": "SUCCESS",
            "data": candle_data,
        }

        df = client.get_historical_data(
            symbol="TCS",
            token="11536",
            from_date="2024-01-01 09:15",
            to_date="2024-01-02 15:30",
        )

        self.assertListEqual(list(df.columns), ["timestamp", "open", "high", "low", "close", "volume"])
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df.iloc[0]["open"], 3500.0)
        self.assertAlmostEqual(df.iloc[1]["close"], 3580.0)
        self.assertEqual(df.iloc[0]["volume"], 100000)

    def test_raises_runtime_error_when_not_logged_in(self):
        """get_historical_data raises RuntimeError if called before login."""
        with patch.dict("sys.modules", {
            "SmartApi": MagicMock(),
            "SmartApi.smartConnect": MagicMock(),
        }), patch.dict("os.environ", {
            "ANGEL_API_KEY": "key",
            "ANGEL_CLIENT_ID": "client",
            "ANGEL_PIN": "pin",
            "ANGEL_TOTP_SECRET": "JBSWY3DPEHPK3PXP",
        }):
            from importlib import reload
            import data_ingestion.angel_one_api as api_module
            reload(api_module)
            client = api_module.AngelOneClient()

        with self.assertRaises(RuntimeError):
            client.get_historical_data(
                symbol="TCS",
                token="11536",
                from_date="2024-01-01",
                to_date="2024-01-02",
            )

    def test_returns_empty_dataframe_on_api_error(self):
        """get_historical_data returns an empty DataFrame when the API reports failure."""
        client = self._make_client()
        client.smart_api.getCandleData.return_value = {
            "status": False,
            "message": "Invalid token",
            "data": None,
        }

        df = client.get_historical_data(
            symbol="TCS",
            token="BADTOKEN",
            from_date="2024-01-01",
            to_date="2024-01-02",
        )

        self.assertTrue(df.empty)
        self.assertListEqual(list(df.columns), ["timestamp", "open", "high", "low", "close", "volume"])

    def test_returns_empty_dataframe_on_no_candle_data(self):
        """get_historical_data returns an empty DataFrame when data list is empty."""
        client = self._make_client()
        client.smart_api.getCandleData.return_value = {
            "status": True,
            "message": "SUCCESS",
            "data": [],
        }

        df = client.get_historical_data(
            symbol="TCS",
            token="11536",
            from_date="2024-01-01",
            to_date="2024-01-01",
        )

        self.assertTrue(df.empty)

    def test_raises_timeout_on_requests_timeout(self):
        """get_historical_data re-raises requests.exceptions.Timeout."""
        import requests
        client = self._make_client()
        client.smart_api.getCandleData.side_effect = requests.exceptions.Timeout("timed out")

        with self.assertRaises(requests.exceptions.Timeout):
            client.get_historical_data(
                symbol="TCS",
                token="11536",
                from_date="2024-01-01",
                to_date="2024-01-31",
            )

    def test_returns_empty_dataframe_on_unexpected_exception(self):
        """get_historical_data returns empty DataFrame for non-timeout exceptions."""
        client = self._make_client()
        client.smart_api.getCandleData.side_effect = ValueError("unexpected")

        df = client.get_historical_data(
            symbol="TCS",
            token="11536",
            from_date="2024-01-01",
            to_date="2024-01-31",
        )

        self.assertTrue(df.empty)

    def test_default_exchange_and_interval(self):
        """get_historical_data passes NSE and ONE_DAY defaults to the API."""
        client = self._make_client()
        client.smart_api.getCandleData.return_value = {
            "status": True,
            "message": "SUCCESS",
            "data": [],
        }

        client.get_historical_data(
            symbol="TCS",
            token="11536",
            from_date="2024-01-01",
            to_date="2024-01-31",
        )

        call_args = client.smart_api.getCandleData.call_args[0][0]
        self.assertEqual(call_args["exchange"], "NSE")
        self.assertEqual(call_args["interval"], "ONE_DAY")
