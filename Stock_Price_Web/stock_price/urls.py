from . import views
from django.urls import path

urlpatterns = [
    
    path('', views.home, name= 'stock_price_home'),
    path('stocks/', views.stocks, name= 'stocks'),
    path('about/', views.about, name= 'about me'),
    path('predictions/', views.predictValues, name= 'predictions'),
]