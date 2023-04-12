from django.urls import path
from .views import *

urlpatterns =[
     path('userhome/',UserHome.as_view(),name="uh"),
     path('userprofile/',UserProfile.as_view(),name="upro"),
     path('addcart/<int:cid>/',addcart,name="acart"),
     path('updatepro/<int:pk>/',UpdateProfile.as_view(),name="ueditpro"),
     path('cart/',CartView.as_view(),name="cart"),
     path('viewpayment/<int:pid>/',PaymentView.as_view(),name="viewpay"),
     path('payment/<int:pid>/',PaymentConfirm,name="pay"),
     path('buy/<int:pid>/',BuyView.as_view(),name="buy"),
     path('delcart/<int:cid>/',delcart.as_view(),name="delcart"),
     path('delbuy/<int:cid>/',delbuy.as_view(),name="delbuy"),
     path('cp/',ChangePasswordView.as_view(),name="cp"),
     path('order/',MyOrder.as_view(),name="mypur"),
     path('review/<int:pid>/',addcomment,name="comment"),
     path('deleteaccount/<int:pk>/',AcDeleteuser.as_view(),name="delus"),
     path('uppic/<int:pk>',UserPic.as_view(),name="upic"),
]
