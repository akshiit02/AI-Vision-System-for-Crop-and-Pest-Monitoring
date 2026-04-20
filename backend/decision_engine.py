import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
import os

# ---------------------------------------------------
# DEVICE
# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------
# BASE PATH
# ---------------------------------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------
CROP_MODEL_PATH = os.path.join(BASE_PATH, "../models/crop_disease/best_model.pth")
PEST_MODEL_PATH = os.path.join(BASE_PATH, "../models/pest/best_model.pth")
SOIL_MODEL_PATH = os.path.join(BASE_PATH, "../models/soil/soil_model.pkl")
SCALER_PATH = os.path.join(BASE_PATH, "../models/soil/scaler.pkl")

# ---------------------------------------------------
# DATA PATHS (for class names)
# ---------------------------------------------------
CROP_DATA_PATH = os.path.join(BASE_PATH, "../data/crop_disease/PlantVillage_Color")
PEST_DATA_PATH = os.path.join(BASE_PATH, "../data/pest/farm_insects")

# ---------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------
crop_classes = sorted(os.listdir(CROP_DATA_PATH))
pest_classes = sorted(os.listdir(PEST_DATA_PATH))

# ---------------------------------------------------
# LOAD CROP MODEL
# ---------------------------------------------------
def load_crop_model():

    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(crop_classes)
    )

    model.load_state_dict(
        torch.load(CROP_MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------
# LOAD PEST MODEL
# ---------------------------------------------------
def load_pest_model():

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(pest_classes)
    )

    model.load_state_dict(
        torch.load(PEST_MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------
# LOAD SOIL MODEL
# ---------------------------------------------------
soil_model = joblib.load(SOIL_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------------------------------------------
# LOAD DEEP LEARNING MODELS
# ---------------------------------------------------
crop_model = load_crop_model()
pest_model = load_pest_model()

# ---------------------------------------------------
# IMAGE TRANSFORM
# ---------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# ---------------------------------------------------
# CROP DISEASE PREDICTION
# ---------------------------------------------------
def predict_crop_disease(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = crop_model(image)

        _, predicted = torch.max(outputs,1)

    return crop_classes[predicted.item()]


# ---------------------------------------------------
# PEST PREDICTION
# ---------------------------------------------------
def predict_pest(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = pest_model(image)

        _, predicted = torch.max(outputs,1)

    return pest_classes[predicted.item()]


# ---------------------------------------------------
# SOIL PREDICTION
# ---------------------------------------------------
def predict_soil(values):

    values = scaler.transform([values])

    prediction = soil_model.predict(values)

    return prediction[0]


# ---------------------------------------------------
# FINAL DECISION ENGINE
# ---------------------------------------------------
def final_decision(soil_values, leaf_image, pest_image):

    soil_result = predict_soil(soil_values)

    disease_result = predict_crop_disease(leaf_image)

    pest_result = predict_pest(pest_image)

    result = {

        "Recommended Crop": soil_result,
        "Disease Detected": disease_result,
        "Pest Detected": pest_result

    }

    if "healthy" not in disease_result.lower():

        advice = "Disease detected. Apply treatment."

    elif pest_result.lower() not in ["none","no_pest"]:

        advice = "Pest risk detected. Apply pest control."

    else:

        advice = "Crop condition looks healthy."

    result["Advice"] = advice

    return result