from django.shortcuts import render,redirect
from django.views.generic import View,TemplateView,FormView,CreateView,UpdateView,DeleteView
from .models import *
from .forms import *
from django.urls import reverse_lazy
from django.contrib.auth import authenticate,login,logout
from django.http import HttpResponse
from store.models import Product

# Create your views here.

class LogView(FormView):
    template_name="login.html"
    form_class=LogForm
    def post(self,request,*args,**kwargs):
        formlog=LogForm(data=request.POST)
        if formlog.is_valid():
            un=formlog.cleaned_data.get("username")
            ps=formlog.cleaned_data.get("password")
            user=authenticate(request,username=un,password=ps)
            if user:
                login(request, user)
                if user.is_staff:
                    return redirect('/admin/')  # Redirect to the admin dashboard
                elif user.usertype == "Store":
                    return redirect("sh")
                else:
                    return redirect("uh")
            else:
                 return render(request,"login.html",{"form":formlog})  
        else:
            return render(request,"login.html",{"form":formlog})  




class RegView(CreateView):
    template_name = "reg.html"
    model= CustUser
    form_class = RegForm
    success_url=reverse_lazy("h")


class LogOut(View):
    def get(self,request,*args,**kwargs):
        logout(request)
        return redirect("h")   
    
def home(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            keyword = form.cleaned_data['keyword']
            test_image = form.cleaned_data['image']
            # Save the uploaded image to a temporary location
            with open('test_image.jpg', 'wb') as f:
                for chunk in test_image.chunks():
                    f.write(chunk)
            # Perform image similarity calculation
            similarity_score = calculate_image_similarity('test_image.jpg', keyword)
            return render(request, 'result.html', {'similarity_score': similarity_score})
    else:
        form = ImageUploadForm()
    return render(request, 'userhome.html', {'form': form}) 
    
def submit_complaint(request):
    if request.method == 'POST':
        form = ComplaintForm(request.POST)
        if form.is_valid():
            # Save the complaint
            complaint = form.save(commit=False)
            complaint.user = request.user
            complaint.save()
            # Delete the product associated with the complaint
            product_to_delete = Product.objects.filter(user=request.user).first()
            if product_to_delete:
                product_to_delete.delete()

            return redirect('complaint_success')
            
         
    else:
        form = ComplaintForm()
    return render(request, 'complaint.html', {'form': form})

def complaint_success(request):
    return render(request, 'complaint_success.html')  # Render a success page
    

