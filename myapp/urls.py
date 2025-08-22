# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'), # Route for the main page
    path('predict/', views.predict, name='predict'), # Route for the prediction endpoint
]
