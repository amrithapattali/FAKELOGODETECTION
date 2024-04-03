from django import forms
from .models import *
from django.contrib.auth.forms import UserCreationForm



class RegForm(UserCreationForm):
    class Meta:
        model=CustUser
        fields=["first_name","last_name","gender","address","place","phone","email","usertype","username","password1","password2"]
        widgets={
            "first_name":forms.TextInput(attrs={"class":"form-control","placeholder":"Firstname"}),
            "last_name":forms.TextInput(attrs={"class":"form-control","placeholder":"Lastname"}),
            "gender":forms.RadioSelect(),
            "address":forms.Textarea(attrs={"class":"form-control","placeholder":"Address","rows":3}),
            "place":forms.TextInput(attrs={"class":"form-control","placeholder":"Place"}),
            "phone":forms.NumberInput(attrs={"class":"form-control","placeholder":"Phone"}),
            "email":forms.EmailInput(attrs={"class":"form-control","placeholder":"Email"}),
            "username":forms.TextInput(attrs={"class":"form-control","placeholder":"Username"}),
            "password1":forms.PasswordInput(attrs={"class":"form-control","placeholder":"Password"}),
            "password2":forms.PasswordInput(attrs={"class":"form-control","placeholder":"Confirm Password"})
        }
   


class LogForm(forms.Form):
    username=forms.CharField(max_length=100,widget=forms.TextInput(attrs={"class":"form-control","placeholder":"Username"}))        
    password=forms.CharField(max_length=100,widget=forms.PasswordInput(attrs={"class":"form-control","placeholder":"Password","name":"password"}))    
    

class ImageUploadForm(forms.Form):
    keyword = forms.CharField(max_length=100)
    image = forms.ImageField()
       
# class ComplaintForm(forms.ModelForm):
#     class Meta:
#         model = Complaint
#         fields = ['complaint_text']
#         widgets = {
#             'complaint_text': forms.Textarea(attrs={'rows': 5}),
#         }    