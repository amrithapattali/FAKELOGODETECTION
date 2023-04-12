from django import forms
from .models import *



class ProductForm(forms.ModelForm):
    class Meta:
        model=Product
        exclude=["user","datetime"]
        widgets={
            "productname":forms.TextInput(attrs={"class":"form-control","placeholder":"Productname"}),
            "model":forms.TextInput(attrs={"class":"form-control","placeholder":"Model"}),
            "price":forms.NumberInput(attrs={"class":"form-control","placeholder":"Price"}),
            "orginalprice":forms.NumberInput(attrs={"class":"form-control","placeholder":"Company Price"}),
            "description":forms.Textarea(attrs={"class":"form-control","placeholder":"Details","rows":3}),
            "color":forms.TextInput(attrs={"class":"form-control","placeholder":"Color"}),
            "ram":forms.TextInput(attrs={"class":"form-control","placeholder":"RAM"}),
            "rom":forms.TextInput(attrs={"class":"form-control","placeholder":"ROM"}),
            "battery":forms.TextInput(attrs={"class":"form-control","placeholder":"Battery"}),
            "warranty":forms.TextInput(attrs={"class":"form-control","placeholder":"Warranty"})
       
        }


class PicForm(forms.ModelForm):
    class Meta:
        model=Profile
        fields=["image"]        