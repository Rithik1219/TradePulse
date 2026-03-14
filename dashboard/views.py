import logging

from django.shortcuts import render

logger = logging.getLogger(__name__)


def portfolio_view(request):
    """Fetch live portfolio data and render the dashboard table.

    Instantiates :class:`~data_ingestion.angel_one_api.AngelOneClient`,
    logs in, and converts the resulting :class:`pandas.DataFrame` into a
    list of dictionaries so it can be iterated in the template.

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

