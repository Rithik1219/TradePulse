from unittest.mock import MagicMock, patch

import pandas as pd
from django.test import TestCase, RequestFactory
from django.utils import timezone

from dashboard.models import PortfolioSnapshot
from dashboard.views import portfolio_view


class PortfolioViewTests(TestCase):
    """Tests for the portfolio_view."""

    def setUp(self):
        self.factory = RequestFactory()

    @patch("dashboard.views.AngelOneClient", create=True)
    def test_portfolio_view_renders_with_data(self, mock_import):
        """View renders portfolio data when API succeeds."""
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
        content = response.content.decode()
        self.assertIn("TCS", content)
        self.assertIn("INFY", content)
        self.assertIn("3500", content)
        self.assertIn("holdings", content)

    def test_portfolio_view_handles_api_error(self):
        """View renders error message when API fails."""
        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(
                AngelOneClient=MagicMock(side_effect=ConnectionError("API down"))
            ),
        }):
            request = self.factory.get("/")
            response = portfolio_view(request)

        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn("Unable to retrieve portfolio data", content)

    def test_portfolio_view_uses_correct_template(self):
        """View uses the portfolio.html template."""
        with patch.dict("sys.modules", {
            "data_ingestion": MagicMock(),
            "data_ingestion.angel_one_api": MagicMock(
                AngelOneClient=MagicMock(side_effect=Exception("test"))
            ),
        }):
            request = self.factory.get("/")
            response = portfolio_view(request)

        self.assertEqual(response.status_code, 200)
        content = response.content.decode()
        self.assertIn("TradePulse", content)
        self.assertIn("Live Portfolio", content)


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

