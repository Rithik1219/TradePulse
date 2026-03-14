from django.db import models


class PortfolioSnapshot(models.Model):
    """Daily snapshot of a single portfolio holding."""

    symbol = models.CharField(max_length=50, db_index=True)
    quantity = models.IntegerField()
    avg_price = models.FloatField()
    source = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.symbol} – {self.timestamp.date()}"
