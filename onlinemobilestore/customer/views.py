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
from django.shortcuts import get_object_or_404
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
    
from django.shortcuts import render
from django.http import HttpResponse
from .models import Product
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tensorflow.keras.applications import VGG16
import easyocr
import requests
from bs4 import BeautifulSoup


def download_image(url, keyword, index):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        if response.status_code == 200:
            filename = f"{keyword}_{index}.jpg"
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while downloading image: {str(e)}")
    return None


def search_and_download_images(keyword, max_images=10):
    try:
        search_query = f"https://www.google.com/search?q={keyword}&tbm=isch"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(search_query, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        img_tags = soup.find_all('img')
        img_urls = [img['src'] for img in img_tags if img.get('src') and img['src'].startswith('http')]

        downloaded_images = []
        for i, img_url in enumerate(img_urls[:max_images]):
            image_path = download_image(img_url, keyword, i + 1)
            if image_path:
                downloaded_images.append(image_path)
        return downloaded_images
    except Exception as e:
        print(f"Error occurred while searching and downloading: {str(e)}")
        return None


def load_and_preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array


def extract_features(image_path):
    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    img = load_and_preprocess_image(image_path)
    features = model.predict(img)
    return features.flatten()


def cosine_similarity(features1, features2):
    dot_product = np.dot(features1, features2)
    norm_features1 = np.linalg.norm(features1)
    norm_features2 = np.linalg.norm(features2)
    similarity = dot_product / (norm_features1 * norm_features2)
    return similarity


def pearson_correlation(features1, features2):
    correlation_coefficient, _ = pearsonr(features1, features2)
    return correlation_coefficient


def extract_text(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    text = ' '.join([result[1] for result in results]) if results else ''
    return text


def calculate_image_similarity(image1_path, image2_path):
    text_similarity = 1 if extract_text(image1_path) == extract_text(image2_path) else -1 if extract_text(
        image1_path) and extract_text(image2_path) else 0
    features1 = extract_features(image1_path)
    features2 = extract_features(image2_path)
    cosine_sim = cosine_similarity(features1, features2)
    pearson_corr = pearson_correlation(features1, features2)
    combined_similarity = (0.4 * cosine_sim) + (0.3 * (pearson_corr + 1) / 2)
    if text_similarity == 1:
        combined_similarity += 1
    elif text_similarity == -1:
        combined_similarity -= 1
    return combined_similarity


def delete_downloaded_images(image_paths):
    for image_path in image_paths:
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_path}")
        else:
            print(f"Image not found: {image_path}")


def image_similarity_view(request,**kwargs):
    
        # product_id = request.POST.get('product_id')  
        id=kwargs.get('pk')
        print(id)
        product = get_object_or_404(Product,id=id)
        keyword = f"{product.model}"  
        test_image_path = product.image.path

        test_text = extract_text(test_image_path)

        real_image_paths = search_and_download_images(keyword)

        if real_image_paths:
            similarity_scores = []

            for real_image_path in real_image_paths:
                real_text = extract_text(real_image_path)
                similarity_score = calculate_image_similarity(test_image_path, real_image_path)
                similarity_scores.append((similarity_score, real_image_path))

            similarity_scores.sort(reverse=True)

            if similarity_scores:
                max_similarity_score, _ = similarity_scores[0]

                if max_similarity_score > 0.75:
                    prediction = "The logo is Real...!!!"
                else:
                    prediction = "The logo is fake..!!!"
            else:
                prediction = "No similar logo found."

            plt.figure(figsize=(8, 6))
            plt.imshow(plt.imread(test_image_path))
            plt.title(prediction, fontsize=16, color='Black', loc='center', pad=20)
            plt.axis('off')
            plt.show()
        else:
            print("No images found for the given keyword.")

        delete_downloaded_images(real_image_paths)

    #     return HttpResponse("Image similarity checked successfully!")
    # else:
    #     return HttpResponse("Invalid request method")
  