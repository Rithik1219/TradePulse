import logging

from django.shortcuts import render
from django.utils import timezone

from dashboard.models import PortfolioSnapshot

logger = logging.getLogger(__name__)


def portfolio_view(request):
    """Fetch live portfolio data, persist daily snapshots, and render the table.

    Instantiates :class:`~data_ingestion.angel_one_api.AngelOneClient`,
    logs in, and converts the resulting :class:`pandas.DataFrame` into a
    list of dictionaries so it can be iterated in the template.

    For each row in the DataFrame a :class:`~dashboard.models.PortfolioSnapshot`
    record is saved to the database only if no record for that ``symbol``
    already exists for today, preventing duplicates on repeated page loads.
    A single query fetches all already-saved symbols for today and new
    records are written in bulk, avoiding N+1 query overhead.

    If the Angel One API is unreachable or any other exception occurs the
    page still renders — with an empty table and a user-friendly error
    message.
    """
    portfolio = []
    error_message = None

    try:
        from data_ingestion.angel_one_api import AngelOneClient

        client = AngelOneClient()
        client.login()
        df = client.get_portfolio()
        portfolio = df.to_dict(orient="records")

        today = timezone.now().date()
        existing_symbols = set(
            PortfolioSnapshot.objects.filter(
                timestamp__date=today,
            ).values_list("symbol", flat=True)
        )

        new_snapshots = [
            PortfolioSnapshot(
                symbol=record.get("symbol", ""),
                quantity=record.get("quantity", 0),
                avg_price=record.get("avg_price", 0.0),
                source=record.get("source", ""),
            )
            for record in portfolio
            if record.get("symbol", "") not in existing_symbols
        ]
        if new_snapshots:
            PortfolioSnapshot.objects.bulk_create(new_snapshots)
    except Exception:
        logger.exception("Failed to fetch portfolio from Angel One API")
        error_message = (
            "Unable to retrieve portfolio data. "
            "Please check your API credentials and try again later."
        )

    return render(
        request,
        "dashboard/portfolio.html",
        {"portfolio": portfolio, "error_message": error_message},
    )

