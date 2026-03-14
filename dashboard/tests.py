from unittest.mock import MagicMock, patch

import pandas as pd
from django.test import TestCase, RequestFactory

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

