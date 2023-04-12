from django.urls import path
from .views import *

urlpatterns =[
    path('reg/',RegView.as_view(),name="reg"),
    path('logout/',LogOut.as_view(),name="logout"),
]