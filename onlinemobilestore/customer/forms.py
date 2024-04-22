from django import forms
from .models import *



class PaymentForm(forms.ModelForm):
    class Meta:
        model=Payment
        exclude=["product","user"]
        widgets={
            "bank":forms.TextInput(attrs={"class":"form-control","placeholder":"Bank"}),
            "acholdername":forms.TextInput(attrs={"class":"form-control","placeholder":"AccountHolder"}),
            "accno":forms.NumberInput(attrs={"class":"form-control","placeholder":"Account Number"}),
            "ifsc":forms.TextInput(attrs={"class":"form-control","placeholder":"IFSC Code"}),
            "quantity":forms.NumberInput(attrs={"class":"form-control","placeholder":"Quantity"}),

        }

class ChangePasswordForm(forms.Form):
    current_password=forms.CharField(max_length=50,label="current password",widget=forms.PasswordInput(attrs={"placeholder":"Password","class":"form-control"}))
    new_password=forms.CharField(max_length=50,label="new password",widget=forms.PasswordInput(attrs={"placeholder":"Password","class":"form-control"}))
    confirm_password=forms.CharField(max_length=50,label="confirm password",widget=forms.PasswordInput(attrs={"placeholder":"Password","class":"form-control"}))

        
class CommentForm(forms.ModelForm):
    class Meta:
        model=Review
        fields=["review"]
        widgets={
            "review":forms.Textarea(attrs={"class":"form-control","placeholder":"Review","rows":3})
        }
        
