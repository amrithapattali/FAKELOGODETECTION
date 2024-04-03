from django.db import models
from accounts.models import *

# Create your models here.


class Product(models.Model):
    productname=models.CharField(max_length=200)
    image=models.FileField(upload_to="product_image")    
    model=models.CharField(max_length=100)    
    color=models.CharField(max_length=100) 
    size=models.IntegerField()
    # battery=models.CharField(max_length=50,null=True)
    # warranty=models.CharField(max_length=50)
    datetime=models.DateTimeField(auto_now_add=True)
    description=models.TextField(max_length=500,null=True)
    orginalprice=models.IntegerField(null=True)
    price=models.IntegerField(null=True)
    user=models.ForeignKey(CustUser,on_delete=models.CASCADE,related_name="p_user")
    
    def __str__(self):
        return self.productname

class Profile(models.Model):
    image=models.ImageField(upload_to="profile_pic",null=True)
    user=models.OneToOneField(CustUser,on_delete=models.CASCADE,related_name="p_pic",null=True)
    
    def __str__(self):
        return self.user.username
    
    
   
# Create your models here.
