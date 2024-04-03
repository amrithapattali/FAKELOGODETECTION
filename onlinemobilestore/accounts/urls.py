from django.urls import path
from .views import *

urlpatterns =[
    path('log/',LogView.as_view(),name="log"),
    path('reg/',RegView.as_view(),name="reg"),
    path('logout/',LogOut.as_view(),name="logout"),
     path('home', home, name='home')
    
]