from django.urls import path
from .views import *

urlpatterns =[
    path('storehome/',StoreHome.as_view(),name="sh"),
    path('additem/',AddProduct.as_view(),name="aproduct"),
    path('pic/',PicView.as_view(),name="pic"),
    path('profile/',ProfileView.as_view(),name="pro"),
    path('myproducts/',MyProduct.as_view(),name="myproduct"),
    path('updateprofile/<int:pk>/',UpdateProfile.as_view(),name="uppro"),
    path('updateproduct/<int:pk>/',AddUpdate.as_view(),name="upproduct"),
    path('deleteproduct/<int:pk>/',DelProduct.as_view(),name="delproduct"),
    path('deleteaccount/<int:pk>/',AcDelete.as_view(),name="delst"),
    path('sale/',MySale.as_view(),name="s"),
    path('editpass/',ChangePassView.as_view(),name="cps"),
    path('editpic/<int:pk>',UpdatePic.as_view(),name="editp"),
    path('reviews/',ReviewView.as_view(),name="rev"),
]