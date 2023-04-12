from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.


class CustUser(AbstractUser):
    options=(
        ("Male","  Male"),
        ("Female","Female"),
        ("Others","Others")
    )
    gender=models.CharField(max_length=100,choices=options,default="Male")
    address=models.TextField()
    place=models.CharField(max_length=200,null=True)
    phone=models.IntegerField(null=True)
    typeop=(
        ("Store","Store"),
        ("Customer","Customer")
    )
    usertype=models.CharField(max_length=100,choices=typeop,default="Customer")
# Create your models here.
