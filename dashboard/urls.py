from django.urls import path

from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.portfolio_view, name="portfolio"),
    path("api/news-predictions/", views.news_predictions_view, name="news_predictions"),
]
