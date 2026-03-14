import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
import os

# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================================================
# PATHS
# ======================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

CROP_MODEL_PATH = os.path.join(BASE_PATH, "../models/crop_disease/best_model.pth")
PEST_MODEL_PATH = os.path.join(BASE_PATH, "../models/pest/best_model.pth")
SOIL_MODEL_PATH = os.path.join(BASE_PATH, "../models/soil/soil_model.pkl")
SCALER_PATH = os.path.join(BASE_PATH, "../models/soil/scaler.pkl")

CROP_DATA_PATH = os.path.join(BASE_PATH, "../data/crop_disease/PlantVillage_Color")
PEST_DATA_PATH = os.path.join(BASE_PATH, "../data/pest/farm_insects")

# ======================================================
# LOAD CLASS NAMES
# ======================================================
crop_classes = sorted(os.listdir(CROP_DATA_PATH))
pest_classes = sorted(os.listdir(PEST_DATA_PATH))

# ======================================================
# LOAD CROP MODEL (MobileNetV2)
# ======================================================
def load_crop_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(crop_classes))
    model.load_state_dict(torch.load(CROP_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# ======================================================
# LOAD PEST MODEL (EfficientNet)
# ======================================================
def load_pest_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(pest_classes))
    model.load_state_dict(torch.load(PEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# ======================================================
# LOAD SOIL MODEL
# ======================================================
soil_model = joblib.load(SOIL_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load deep models
crop_model = load_crop_model()
pest_model = load_pest_model()

# ======================================================
# IMAGE TRANSFORM
# ======================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ======================================================
# PREDICTION FUNCTIONS
# ======================================================
def predict_crop_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = crop_model(image)
        _, predicted = torch.max(outputs, 1)

    return crop_classes[predicted.item()]

def predict_pest(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = pest_model(image)
        _, predicted = torch.max(outputs, 1)

    return pest_classes[predicted.item()]

def predict_soil(values):
    values = scaler.transform([values])
    prediction = soil_model.predict(values)
    return prediction[0]

# ======================================================
# FINAL DECISION LOGIC
# ======================================================
def final_decision(soil_values, leaf_image, pest_image):

    soil_result = predict_soil(soil_values)
    disease_result = predict_crop_disease(leaf_image)
    pest_result = predict_pest(pest_image)

    result = {
        "Recommended Crop": soil_result,
        "Disease Detected": disease_result,
        "Pest Detected": pest_result
    }

    # Advisory Logic
    if "healthy" not in disease_result.lower():
        advice = "Disease detected. Apply appropriate treatment."

    elif pest_result.lower() not in ["none", "no_pest"]:
        advice = "Pest risk detected. Consider pest control measures."

    else:
        advice = "Crop condition looks healthy."

    result["Advice"] = advice

    return result

# ======================================================
# TEST RUN
# ======================================================
if __name__ == "__main__":

    # Example soil input
    soil_input = [
    float(input("Enter Nitrogen (N): ")),
    float(input("Enter Phosphorus (P): ")),
    float(input("Enter Potassium (K): ")),
    float(input("Enter Temperature: ")),
    float(input("Enter Humidity: ")),
    float(input("Enter pH: ")),
    float(input("Enter Rainfall: "))
]


    # Replace with real test images
    leaf_image_path = "../test_leaf.jpg"
    pest_image_path = "../test_pest.jpg"

    if os.path.exists(leaf_image_path) and os.path.exists(pest_image_path):

        output = final_decision(soil_input, leaf_image_path, pest_image_path)

        print("\n===== AI CROP MONITORING RESULT =====")
        for key, value in output.items():
            print(f"{key}: {value}")

    else:
        print("\nAdd test_leaf.jpg and test_pest.jpg in project root to test.")
