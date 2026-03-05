from django.contrib import admin
from django.urls import path, include # Import include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('advisor.urls')), # This connects your app to the root URL
]