# AgroVision AI — Crop, Soil & Pest Monitoring System

An AI-driven agricultural advisory system that detects crop diseases, identifies pests, and recommends suitable crops from soil parameters — combined into a single recommendation and served through a working API and web interface.

## Overview

The system integrates three machine learning models:

- **Crop disease detection** — CNN trained on the PlantVillage dataset
- **Pest detection** — computer vision model trained on an insect image dataset
- **Soil-based crop recommendation** — ML model trained on soil nutrient data (N, P, K, temperature, humidity, pH)

A decision engine combines the outputs of all three into one final recommendation.

## My contribution

This started as a joint project with [Arjun Khimta](https://github.com/ArjunKhimta), who built and trained the three ML models and the original decision engine. I built the deployment layer on top: a FastAPI backend (`backend/api.py`) with an `/analyze` endpoint that accepts leaf and pest images, plus a frontend (`frontend/`) that lets a user upload images and run the analysis from the browser — turning the original research notebooks into something actually usable end to end.

## Project structure

```
├── main.py
├── backend/
│   ├── api.py              # FastAPI app, /analyze endpoint
│   └── decision_engine.py  # combines model outputs
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── models/                 # trained model weights
└── notebooks/               # training notebooks for each model
```

## Built with

Python, PyTorch, scikit-learn, FastAPI, NumPy, Pandas

## Running it

```bash
git clone https://github.com/akshiit02/AI-Vision-System-for-Crop-and-Pest-Monitoring.git
cd AI-Vision-System-for-Crop-and-Pest-Monitoring
pip install fastapi uvicorn python-multipart torch torchvision pillow joblib scikit-learn numpy pandas
uvicorn backend.api:app --reload
```
Then open `frontend/index.html` in a browser.

Datasets aren't included here due to size — download them yourself and place them as described in the notebooks:
- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [Dangerous Insects Dataset](https://www.kaggle.com/datasets/tarundalal/dangerous-insects-dataset)
- [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

## What I'd add next

- Deploy the backend so the demo doesn't require local setup
- Add basic auth + usage logging on the API
- Real-time camera-based monitoring instead of single-image upload

## Authors

Arjun Khimta, Akshit Sharma
