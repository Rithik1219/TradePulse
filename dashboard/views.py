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
_HISTORICAL_DAYS = 60  # days of OHLCV history to fetch per symbol

# ---------------------------------------------------------------------------
# Sector mapping — maps NSE trading symbols to their market sector.
# Unmapped symbols fall back to "Other".
# ---------------------------------------------------------------------------

_SECTOR_MAP: dict = {
    "RELIANCE-EQ": "Energy",
    "ONGC-EQ": "Energy",
    "COALINDIA-EQ": "Energy",
    "BPCL-EQ": "Energy",
    "IOC-EQ": "Energy",
    "TCS-EQ": "IT",
    "INFY-EQ": "IT",
    "WIPRO-EQ": "IT",
    "TECHM-EQ": "IT",
    "HCLTECH-EQ": "IT",
    "LTIM-EQ": "IT",
    "MPHASIS-EQ": "IT",
    "HDFCBANK-EQ": "Banking",
    "ICICIBANK-EQ": "Banking",
    "SBIN-EQ": "Banking",
    "KOTAKBANK-EQ": "Banking",
    "AXISBANK-EQ": "Banking",
    "INDUSINDBK-EQ": "Banking",
    "FEDERALBNK-EQ": "Banking",
    "BAJFINANCE-EQ": "Finance",
    "BAJAJFINSV-EQ": "Finance",
    "HDFCLIFE-EQ": "Finance",
    "SBILIFE-EQ": "Finance",
    "BHARTIARTL-EQ": "Telecom",
    "IDEA-EQ": "Telecom",
    "SUNPHARMA-EQ": "Pharma",
    "DRREDDY-EQ": "Pharma",
    "CIPLA-EQ": "Pharma",
    "DIVISLAB-EQ": "Pharma",
    "APOLLOHOSP-EQ": "Pharma",
    "HINDUNILVR-EQ": "FMCG",
    "NESTLEIND-EQ": "FMCG",
    "BRITANNIA-EQ": "FMCG",
    "DABUR-EQ": "FMCG",
    "MARICO-EQ": "FMCG",
    "TATAMOTORS-EQ": "Automobile",
    "MARUTI-EQ": "Automobile",
    "EICHERMOT-EQ": "Automobile",
    "HEROMOTOCO-EQ": "Automobile",
    "BAJAJ-AUTO-EQ": "Automobile",
    "LT-EQ": "Infrastructure",
    "ADANIPORTS-EQ": "Infrastructure",
    "ULTRACEMCO-EQ": "Infrastructure",
    "GRASIM-EQ": "Infrastructure",
    "POWERGRID-EQ": "Utilities",
    "NTPC-EQ": "Utilities",
    "TATAPOWER-EQ": "Utilities",
    "TITAN-EQ": "Consumer Goods",
    "ASIANPAINT-EQ": "Consumer Goods",
    "TATASTEEL-EQ": "Metals",
    "HINDALCO-EQ": "Metals",
    "JSWSTEEL-EQ": "Metals",
}

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
    historical_charts: dict = {}

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
                    from_date = (now - timedelta(days=_HISTORICAL_DAYS)).strftime(
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
                        # Store historical close prices for front-end drill-down.
                        historical_charts[symbol] = {
                            "dates": [
                                str(ts.date()) for ts in hist_df["timestamp"]
                            ],
                            "closes": [
                                round(float(c), 2)
                                for c in hist_df["close"].fillna(0)
                            ],
                        }
                except Exception:
                    logger.warning(
                        "AI prediction failed for %s; using fallback.",
                        record.get("symbol", "?"),
                        exc_info=True,
                    )
            record["ai_signal"] = ai_signal
            record["ai_signal_float"] = (
                float(ai_signal) if ai_signal != "Analyzing..." else None
            )
            record["market_value"] = round(
                float(record.get("quantity", 0)) * float(record.get("avg_price", 0.0)), 2
            )

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

    # -----------------------------------------------------------------------
    # Portfolio Net-Worth history chart (from persisted snapshots)
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Sector Allocation doughnut chart data
    # -----------------------------------------------------------------------
    sector_totals: dict = {}
    for record in portfolio:
        sym = record.get("symbol", "")
        sector = _SECTOR_MAP.get(sym, "Other")
        sector_totals[sector] = sector_totals.get(sector, 0.0) + record.get("market_value", 0.0)

    sector_labels = json.dumps(list(sector_totals.keys()))
    sector_values = json.dumps([round(v, 2) for v in sector_totals.values()])

    # -----------------------------------------------------------------------
    # Historical per-stock close-price data for the JS drill-down modal
    # -----------------------------------------------------------------------
    historical_charts_json = json.dumps(historical_charts)

    return render(
        request,
        "dashboard/portfolio.html",
        {
            "portfolio": portfolio,
            "error_message": error_message,
            "chart_labels": chart_labels,
            "chart_values": chart_values,
            "sector_labels": sector_labels,
            "sector_values": sector_values,
            "historical_charts_json": historical_charts_json,
        },
    )

