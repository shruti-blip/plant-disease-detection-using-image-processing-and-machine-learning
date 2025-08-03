import sys
sys.path.append("C:/Users/DELL/cropguard/venv/Lib/site-packages")

import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torchvision.models as models

# Load ResNet50 model (pretrained on ImageNet)
model = models.resnet50(pretrained=True)
model.fc = nn.Identity()  # Remove classification layer to get features only
model.eval()

# Move model to CPU or GPU based on availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_features(image_path):
    """Extracts a deep feature vector from an image using ResNet50."""
    try:
        # Load the image and convert it to RGB format
        image = Image.open(image_path).convert("RGB")

        # Apply image transformations required for ResNet50
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Convert image to tensor and add batch dimension
        image_tensor = transform(image).unsqueeze(0)  

        # Move image tensor to the same device as the model (CPU/GPU)
        image_tensor = image_tensor.to(device)

        # Extract features without gradient computation (for efficiency)
        with torch.no_grad():
            features = model(image_tensor)  

        # Convert tensor to NumPy array and return
        return features.cpu().numpy().flatten()  

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def find_best_match(extracted_features, dataset_features):
    """Finds the closest matching label from the dataset using cosine similarity."""
    try:
        # Convert extracted features to a 2D array (1, 2048)
        extracted_features = np.array(extracted_features).reshape(1, -1)  
        dataset_features = np.array(dataset_features)  

        # Ensure dataset features are also 2D (N, 2048)
        if len(dataset_features.shape) == 1:
            dataset_features = dataset_features.reshape(1, -1)

        # Check if feature dimensions match before computing similarity
        if dataset_features.shape[1] != extracted_features.shape[1]:
            print(f"Error: Feature dimension mismatch! Extracted: {extracted_features.shape}, Dataset: {dataset_features.shape}")
            return None

        # Compute cosine similarity between input features and dataset features
        similarities = cosine_similarity(extracted_features, dataset_features)[0]
        best_match_index = np.argmax(similarities)  # Get index of highest similarity
        return best_match_index  

    except Exception as e:
        print(f"Error in finding best match: {e}")
        return None
