from django.db import models
from accounts.models import CustUser
from store.models import Product
from datetime import datetime

# Create your models here.

class Review(models.Model):
    review=models.CharField(max_length=200)
    datetime=models.DateTimeField(auto_now_add=True)  
    product=models.ForeignKey(Product,on_delete=models.CASCADE,related_name="r_user")  
    user=models.ForeignKey(CustUser,on_delete=models.CASCADE,related_name="c_user") 

    def __str__(self):
        return self.review
    

class Payment(models.Model):
    bank=models.CharField(max_length=100)
    acholdername=models.CharField(max_length=100)
    accno=models.IntegerField()
    ifsc=models.CharField(max_length=100,null=True)
    user=models.ForeignKey(CustUser,on_delete=models.CASCADE,related_name="u_payment")
    product=models.ForeignKey(Product,on_delete=models.CASCADE,related_name="p_payment",null=True)
    quantity=models.PositiveBigIntegerField(null=True)
    status=models.CharField(max_length=100,null=True)
    
    def __str__(self):
        return self.acholdername




class Cart(models.Model): 
    class Meta:
        unique_together=['user','product']
    status=models.CharField(max_length=100)
    product=models.ForeignKey(Product,on_delete=models.CASCADE,related_name="c_product")   
    user=models.ForeignKey(CustUser,on_delete=models.CASCADE,related_name="cart_user")
    
    def __str__(self):
        return self.product.productname
   