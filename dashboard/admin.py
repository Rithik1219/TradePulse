from django.contrib import admin

from dashboard.models import PortfolioSnapshot


@admin.register(PortfolioSnapshot)
class PortfolioSnapshotAdmin(admin.ModelAdmin):
    list_display = ("symbol", "quantity", "avg_price", "source", "timestamp")
    list_filter = ("symbol", "timestamp")
