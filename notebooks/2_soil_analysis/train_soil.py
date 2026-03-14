import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ======================================================
# PATHS
# ======================================================
DATA_PATH = "../../data/soil/Crop_recommendation.csv"
SAVE_DIR = "../../models/soil"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# ======================================================
# SPLIT FEATURES & LABEL
# ======================================================
X = df.drop("label", axis=1)
y = df["label"]

# ======================================================
# TRAIN TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================================
# FEATURE SCALING
# ======================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# MODEL
# ======================================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# ======================================================
# EVALUATION
# ======================================================
preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
print("\nAccuracy:", accuracy * 100)

print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# ======================================================
# SAVE MODEL + SCALER
# ======================================================
joblib.dump(model, os.path.join(SAVE_DIR, "soil_model.pkl"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))

print("\nSoil model saved successfully.")
