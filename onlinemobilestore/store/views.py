from django.shortcuts import render,redirect
from django.views.generic import View,TemplateView,UpdateView,CreateView,DeleteView,FormView
from accounts.models import CustUser
from accounts.forms  import *
from django.urls import reverse_lazy
from .forms import *
from .models import *
from customer.models import *
from customer.forms import *
from django.contrib.auth import authenticate,logout,login
from django.utils.decorators import method_decorator

# Create your views here.

def signin_required(fn):
    def wrapper(request,*args,**kwargs):
        if request.user.is_authenticated:
            return fn(request,*args,**kwargs)
    return wrapper

@method_decorator(signin_required,name="dispatch")
class StoreHome(TemplateView):
    template_name="storehome.html"   
    def get_context_data(self, **kwargs):
        context= super().get_context_data(**kwargs)
        context["data"]=Product.objects.all().order_by("-datetime")
        return context

@method_decorator(signin_required,name="dispatch")
class ProfileView(TemplateView):
    template_name="profile.html"  

@method_decorator(signin_required,name="dispatch")
class PicView(CreateView):
    template_name="pic.html"    
    model=Profile
    form_class=PicForm
    success_url=reverse_lazy("pro")
    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)

@method_decorator(signin_required,name="dispatch")
class UpdateProfile(UpdateView):
    form_class=RegForm
    model=CustUser
    template_name="updateprofile.html"
    success_url=reverse_lazy("pro")
    pk_url_kwarg="pk"
    # def form_invalid(self, form):
    #     print("update")
    #     return super().form_invalid(form)
@method_decorator(signin_required,name="dispatch")    
class AddProduct(CreateView): 
    form_class=ProductForm
    model=Product
    template_name="product.html"
    success_url=reverse_lazy("sh")
    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)
    
@method_decorator(signin_required,name="dispatch")
class MyProduct(TemplateView):
    template_name="myproducts.html"    
    def get_context_data(self, **kwargs):
        context= super().get_context_data(**kwargs)
        context["data"]=Product.objects.filter(user=self.request.user)
        return context

@method_decorator(signin_required,name="dispatch")    
class AddUpdate(UpdateView):
    template_name="updateproduct.html"   
    form_class=ProductForm
    model=Product
    success_url=reverse_lazy("myproduct")

@method_decorator(signin_required,name="dispatch")
class DelProduct(DeleteView):
        template_name="delproduct.html"
        model=Product
        success_url=reverse_lazy("myproduct")

@method_decorator(signin_required,name="dispatch")        
class MySale(TemplateView):
    template_name="mysales.html"   
    def get_context_data(self, **kwargs):
        context= super().get_context_data(**kwargs)     
        context["data"]=Payment.objects.all()
        context["cdata"]=Product.objects.filter(user=self.request.user)
        return context
    

@method_decorator(signin_required,name="dispatch")
class ChangePassView(FormView):
    template_name="storeeditpass.html"
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
            return render(request,"storeeditpass.html",{"form":form_data})    
        
@method_decorator(signin_required,name="dispatch")
class UpdatePic(UpdateView):
    template_name="upic.html"
    model=Profile
    form_class=PicForm
    success_url=reverse_lazy("pro")
 

# @method_decorator(signin_required,name="dispatch")
class ReviewView(TemplateView):
    template_name="review.html"
    def get_context_data(self, **kwargs):
        context =super().get_context_data(**kwargs)
        context["data"]=Product.objects.filter(user=self.request.user)
        context["cdata"]=Review.objects.all().order_by("-datetime")
        return context 
    
@method_decorator(signin_required,name="dispatch")
class AcDelete(DeleteView):
    template_name="storeacdel.html"      
    model=CustUser
    success_url=reverse_lazy("sh")
    
