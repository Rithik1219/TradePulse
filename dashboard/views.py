import json
import logging
import time

from datetime import timedelta

from django.db.models import FloatField, Sum
from django.db.models.expressions import ExpressionWrapper, F
from django.db.models.functions import TruncDate
from django.shortcuts import render
from django.utils import timezone

from dashboard.models import PortfolioSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MARKET_OPEN_TIME = "09:15"
_MARKET_CLOSE_TIME = "15:30"
_ANGEL_ONE_REQUEST_DELAY = 0.5  # seconds — stay under 3 req/s rate limit

# ---------------------------------------------------------------------------
# ML predictor — loaded once at module level so the heavy artifacts stay in
# memory for the lifetime of the server process.
# ---------------------------------------------------------------------------

_predictor = None

try:
    from ml_core.predictor import TradePulsePredictor
    _predictor = TradePulsePredictor()
except Exception:
    logger.warning(
        "TradePulsePredictor could not be loaded. "
        "AI signals will fall back to 'Analyzing…'.",
        exc_info=True,
    )


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

        for record in portfolio:
            ai_signal = "Analyzing..."
            if _predictor is not None:
                try:
                    symbol = record.get("symbol", "")
                    token = record.get("symboltoken", "")
                    now = timezone.now()
                    from_date = (now - timedelta(days=30)).strftime(
                        f"%Y-%m-%d {_MARKET_OPEN_TIME}"
                    )
                    to_date = now.strftime(
                        f"%Y-%m-%d {_MARKET_CLOSE_TIME}"
                    )
                    hist_df = client.get_historical_data(
                        symbol=symbol,
                        token=token,
                        from_date=from_date,
                        to_date=to_date,
                    )
                    time.sleep(_ANGEL_ONE_REQUEST_DELAY)
                    if not hist_df.empty:
                        prob = _predictor.predict_signal(
                            hist_df, sentiment_score=0.0
                        )
                        ai_signal = f"{prob:.2f}"
                except Exception:
                    logger.warning(
                        "AI prediction failed for %s; using fallback.",
                        record.get("symbol", "?"),
                        exc_info=True,
                    )
            record["ai_signal"] = ai_signal

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

    snapshot_qs = (
        PortfolioSnapshot.objects.annotate(date=TruncDate("timestamp"))
        .values("date")
        .annotate(
            total_value=Sum(
                ExpressionWrapper(
                    F("quantity") * F("avg_price"),
                    output_field=FloatField(),
                )
            )
        )
        .order_by("date")
    )

    chart_labels = json.dumps(
        [entry["date"].strftime("%Y-%m-%d") for entry in snapshot_qs]
    )
    chart_values = json.dumps(
        [round(entry["total_value"] or 0, 2) for entry in snapshot_qs]
    )

    return render(
        request,
        "dashboard/portfolio.html",
        {
            "portfolio": portfolio,
            "error_message": error_message,
            "chart_labels": chart_labels,
            "chart_values": chart_values,
        },
    )

