import os
import json
import logging
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from scipy.spatial.distance import euclidean
from django.conf import settings
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from .models import CustomUser
from .serializers import SignupSerializer
from .utils import extract_features, find_best_match
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.base import ContentFile
from django.contrib.auth import logout
from django.shortcuts import redirect
from .form import ContactMessageForm
# Setup logging


# ✅ Define trained model paths correctly
MODEL_PATHS = {
    "bellpepper": os.path.join(settings.BASE_DIR, "models/models/npy_models/bellpeppertrained_fixed.npy"),
    "tomato": os.path.join(settings.BASE_DIR, "models/models/npy_models/tomatotrained_fixed.npy"),
    "apple": os.path.join(settings.BASE_DIR, "models/models/npy_models/appletrained_fixed.npy"),
    "cherry": os.path.join(settings.BASE_DIR, "models/models/npy_models/cherrytrained_fixed.npy"),
    "grape": os.path.join(settings.BASE_DIR, "models/models/npy_models/grapetrained_fixed.npy"),
    "potato": os.path.join(settings.BASE_DIR, "models/models/npy_models/potatotrained_fixed.npy"),
    "wheat": os.path.join(settings.BASE_DIR, "models/models/npy_models/wheattrained_fixed.npy"),
    "strawberry": os.path.join(settings.BASE_DIR, "models/models/npy_models/strawberrytrained_fixed.npy"),
    "rice": os.path.join(settings.BASE_DIR, "models/models/npy_models/ricetrained_fixed.npy"),
    "peach": os.path.join(settings.BASE_DIR, "models/models/npy_models/peachtrained_fixed.npy"),
    "corn": os.path.join(settings.BASE_DIR, "models/models/npy_models/corntrained_fixed.npy"),
    
    # Add paths for all models...
}

# ✅ Ensure media/temp directory exists
MEDIA_TEMP_DIR = os.path.join(settings.MEDIA_ROOT, 'temp')
os.makedirs(MEDIA_TEMP_DIR, exist_ok=True)

def check_authentication(request):
    return JsonResponse({"authenticated": request.user.is_authenticated})

def home(request):
    return render(request, "home.html")
def intro(request):
    return render(request, "intro.html")

def home2(request):
    return render(request, "home2.html")
def code_2(request):
    return render(request, "code_2.html")

def signup_page(request):
    return render(request, "signUp.html")

def login_page(request):
    return render(request, "login.html")

def contact_page(request):
    return render(request, "contact.html")

def contact_view(request):
    if request.method == 'POST':
        form = ContactMessageForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, 'intro.html')  # or redirect to a success page
    else:
        form = ContactMessageForm()
    return render(request, 'contact.html', {'form': form})

def load_model(plant_type):
    """Load the correct `.npy` model for the selected plant."""
    npy_model_path = MODEL_PATHS.get(plant_type)

    if not npy_model_path or not os.path.exists(npy_model_path):
        logger.error(f"Model file not found for plant: {plant_type}")
        return None  # ✅ Corrected return value

    try:
        # ✅ Step 1: Load the .npy file
        loaded_data = np.load(npy_model_path, allow_pickle=True)

        # ✅ Step 2: Ensure loaded data is a dictionary
        if isinstance(loaded_data, dict):
            logger.debug(f"Successfully loaded model for {plant_type}")
            return loaded_data  # Correct format
        elif isinstance(loaded_data, np.ndarray):
            extracted_data = loaded_data.item()
            if isinstance(extracted_data, dict):
                logger.debug(f"Successfully extracted dictionary from {plant_type} model")
                return extracted_data
            else:
                logger.error(f"Invalid model format in {npy_model_path}")
                return None
        else:
            logger.error(f"Unexpected data type in {npy_model_path}: {type(loaded_data)}")
            return None

    except Exception as e:
        logger.exception(f"Error loading model for {plant_type}: {e}")
        return None  # ✅ Corrected return value


def logout_view(request):
    # Handle GET logout
    logout(request)
    messages.success(request, "Logout successful!")
    return redirect('intro')  # Replace 'home' with your homepage route name




def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

@login_required(login_url="/users/login/")
def home2(request):
    if request.method == 'POST' and 'leaf_image' in request.FILES:
        plant_type = request.POST.get("plant")

        if not plant_type:
            messages.error(request, "Please select a plant before scanning.")
            return render(request, "home2.html", {"error": "Please select a plant before scanning."})

        image = request.FILES['leaf_image']
        image_path = default_storage.save(f'temp/{image.name}', image)
        full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)

        # ✅ Load the correct model
        model = load_model(plant_type)
        if model is None:
            return render(request, 'home2.html', {'error': "Model not found!", 'plant_type': plant_type})

        if "feature_matrix" in model and "labels" in model:
            extracted_features = extract_features(full_image_path)
            if extracted_features is None:
                return render(request, 'home2.html', {'error': "Feature extraction failed!", 'plant_type': plant_type})

            distances = [euclidean(extracted_features, feature_vector) for feature_vector in model['feature_matrix']]
            best_match_index = np.argmin(distances)
            prediction = f"Classified as: {model['labels'][best_match_index]}"

        else:
            return render(request, 'home2.html', {'error': "Invalid model format!"})

        return render(request, 'home2.html', {'plant_type': plant_type, 'result': prediction})

    return render(request, 'home2.html')

@csrf_exempt
def user_signup(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            username, email, password = data.get("username"), data.get("email"), data.get("password")

            if CustomUser.objects.filter(email=email).exists():
                return JsonResponse({"error": "Email already registered. Please log in."}, status=400)

            if CustomUser.objects.filter(username=username).exists():
                return JsonResponse({"error": "Username already exists. Please log in."}, status=400)

            CustomUser.objects.create_user(username=username, email=email, password=password)
            return JsonResponse({"message": "Signup successful. Redirecting to login."}, status=201)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data"}, status=400)
    return JsonResponse({"error": "Invalid request"}, status=400)

@csrf_exempt
def user_login(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            email, password = data.get("email"), data.get("password")

            if not email or not password:
                return JsonResponse({"error": "Email and password required"}, status=400)

            try:
                user = CustomUser.objects.get(email=email)
            except CustomUser.DoesNotExist:
                return JsonResponse({"error": "Invalid credentials. Try again."}, status=400)

            user = authenticate(request, username=user.username, password=password)
            if user is not None:
                login(request, user)
                return JsonResponse({"message": "Login successful"}, status=200)
            else:
                return JsonResponse({"error": "Invalid credentials. Try again."}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
    return JsonResponse({"error": "Invalid request"}, status=400)
logger = logging.getLogger(__name__)


@csrf_exempt  
def classify_image(request):
    """Classify an uploaded leaf image."""
    if request.method == "POST":
        logger.debug("Received Image Classification Request")

        # ✅ Step 1: Get Image and Plant Type
        image_file = request.FILES.get("leaf_image")
        plant_type = request.POST.get("plant")

        if not image_file or not plant_type:
            return JsonResponse({"error": "Image and plant type are required"}, status=400)

        # ✅ Step 2: Load Model
        model_data = load_model(plant_type)
        if model_data is None:
            logger.error(f"Model not found for {plant_type}")
            return JsonResponse({"error": f"Model not found for {plant_type}"}, status=400)

        dataset_features = model_data.get("feature_matrix")
        labels = model_data.get("labels")

        if dataset_features is None or labels is None:
            logger.error("Invalid model format: Missing feature_matrix or labels")
            return JsonResponse({"error": "Invalid model format"}, status=400)

        # ✅ Step 3: Save Uploaded Image Temporarily
        temp_path = os.path.join(settings.MEDIA_ROOT, "temp", image_file.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)  # Ensure directory exists

        with open(temp_path, "wb+") as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        # ✅ Step 4: Extract Features from Uploaded Image
        extracted_features = extract_features(temp_path)
        if extracted_features is None:
            logger.error("Feature extraction failed")
            return JsonResponse({"error": "Feature extraction failed"}, status=400)

        logger.debug(f"Extracted Features: {extracted_features}")

        # ✅ Step 5: Find Best Match from Dataset
        predicted_index = find_best_match(extracted_features, dataset_features)

        if predicted_index is not None and 0 <= predicted_index < len(labels):
            predicted_label = labels[predicted_index]
        else:
            predicted_label = "Unknown"

        logger.debug(f"Predicted Label: {predicted_label}")

        return JsonResponse({"prediction": str(predicted_label)})

    return JsonResponse({"error": "Invalid request"}, status=400)
