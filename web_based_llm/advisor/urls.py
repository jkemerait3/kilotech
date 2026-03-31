from django.urls import path
from . import views

urlpatterns = [
    path('', views.advisor_home, name='advisor_home'), 
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('history/', views.conversation_history_view, name='conversation_history'),
    path('history/user/<int:user_id>/', views.conversation_history_view, name='conversation_history_for_user'),
]