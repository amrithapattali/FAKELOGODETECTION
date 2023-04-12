from django.shortcuts import render,redirect
from django.views.generic import View,TemplateView,UpdateView,CreateView,FormView,DeleteView
from accounts.models import CustUser
from accounts.forms import RegForm
from django.urls import reverse_lazy
from store.views import Product
from .models import *
from .forms import *
from store.forms import *
from django.contrib.auth import authenticate,login,logout
from django.utils.decorators import method_decorator
from django.contrib import messages
# Create your views here.

def signin_required(fn):
    def wrapper(request,*args,**kwargs):
        if request.user.is_authenticated:
            return fn(request,*args,**kwargs)
    return wrapper

@method_decorator(signin_required,name="dispatch")
class UserHome(TemplateView):
    template_name="userhome.html"
    def get_context_data(self, **kwargs):
        context= super().get_context_data(**kwargs)
        context["data"]=Product.objects.all().order_by("-datetime")
        # context["cdata"]=Cart.objects.filter(user=self.request.user)
        return context 

@method_decorator(signin_required,name="dispatch")
class UserProfile(TemplateView):
    template_name="userprofile.html"

@method_decorator(signin_required,name="dispatch")
class UpdateProfile(UpdateView):
    template_name="updateuserprofile.html"  
    form_class=RegForm
    model=CustUser
    success_url=reverse_lazy("uh")
         


def addcart(request,*args,**kwargs):
    id=kwargs.get("cid")  
    product=Product.objects.get(id=id)
    user=request.user
    if  Cart.objects.filter(product=product,user=request.user,status="Carted"):
        messages.success(request,"Already Added!!")
        return redirect("uh")
    else:
       Cart.objects.create(product=product,user=user,status="Carted")
       return redirect("uh")

@method_decorator(signin_required,name="dispatch")
class CartView(TemplateView):
    template_name="cart.html"
    def get_context_data(self, **kwargs):
        context= super().get_context_data(**kwargs)
        context["data"]=Cart.objects.filter(user=self.request.user ,status = "Carted")
        return context
    


@method_decorator(signin_required,name="dispatch")
class BuyView(TemplateView):
    template_name="buy.html"
    def get_context_data(self, **kwargs):  
        context = super().get_context_data(**kwargs)
        id=kwargs.get("pid")
        context["data"]=Product.objects.filter(id=id)
        return context  

@method_decorator(signin_required,name="dispatch")
class PaymentView(TemplateView):
    template_name="payment.html"
    def get_context_data(self, **kwargs):  
        context = super().get_context_data(**kwargs)
        id=kwargs.get("pid")
        context["data"]=Product.objects.filter(id=id)
        context["form"]=PaymentForm()
        return context  


def PaymentConfirm(request,*args,**kwargs):
    id=kwargs.get("pid")
    product=Product.objects.get(id=id)
    user=request.user
    bank=request.POST.get('bank')
    acholdername=request.POST.get('acholdername')
    accno=request.POST.get('accno')
    ifsc=request.POST.get('ifsc')
    quantity=request.POST.get('quantity')
    Payment.objects.create(bank=bank,acholdername=acholdername,accno=accno,ifsc=ifsc,quantity=quantity,product=product,user=user,status="Purchased")
    # messages.success(request,"Order has been placed")
    return redirect('uh')



        

    
    
        
@method_decorator(signin_required,name="dispatch")
class  delcart(View):
    def get(self,request,*args,**kwargs):    
        id=kwargs.get("cid")
        cart=Cart.objects.get(id=id)
        cart.delete()
        return redirect('cart')    

@method_decorator(signin_required,name="dispatch")
class  delbuy(View):
    def get(self,request,*args,**kwargs):    
        id=kwargs.get("cid")
        cart=Payment.objects.get(id=id)
        cart.delete()
        return redirect('uh')  

@method_decorator(signin_required,name="dispatch")
class ChangePasswordView(FormView):
    template_name="changepassword.html"
    form_class=ChangePasswordForm
    def post(self,request,*args,**kwargs):
        form_data=ChangePasswordForm(data=request.POST)
        if form_data.is_valid():
            current=form_data.cleaned_data.get("current_password")
            new=form_data.cleaned_data.get("new_password")
            confirm=form_data.cleaned_data.get("confirm_password")
            user=authenticate(request,username=request.user.username,password=current)
            if user:
                if new==confirm:
                    user.set_password(new)
                    user.save()
                    # messages.success(request,"password changed")
                    logout(request)
                    return redirect("h")
                else:
                    # messages.error(request,"password mismatches!")
                    return redirect("cp")
            else:
                # messages.error(request,"passsword incorrect!")
                return redirect("cp")
        else:
            return render(request,"changepassword.html",{"form":form_data})
        

@method_decorator(signin_required,name="dispatch")
class MyOrder(TemplateView):
    template_name="myorder.html"   
    def get_context_data(self, **kwargs):
        context= super().get_context_data(**kwargs)     
             
        context["data"]=Payment.objects.filter(user=self.request.user,status="Purchased")
        context["cdata"]=CommentForm()
        context["rdata"]=Review.objects.all().order_by("-datetime")
        return context


def addcomment(request,*args,**kwargs):
      if request.method=="POST":
            id=kwargs.get("pid")
            products=Product.objects.get(id=id)
            user=request.user
            review=request.POST.get("review")
            Review.objects.create(review=review,user=user,product=products)
            return redirect("mypur")
      

@method_decorator(signin_required,name="dispatch")
class AcDeleteuser(DeleteView):
    template_name="useracdel.html"      
    model=CustUser
    success_url=reverse_lazy("uh")


@method_decorator(signin_required,name="dispatch")
class UserPic(UpdateView):
    template_name="updatepic.html"
    model=Profile
    form_class=PicForm
    success_url=reverse_lazy("upro")    