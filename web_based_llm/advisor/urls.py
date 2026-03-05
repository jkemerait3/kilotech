from django.urls import path
from . import views

urlpatterns = [
    path('', views.advisor_home, name='advisor_home'), 
]