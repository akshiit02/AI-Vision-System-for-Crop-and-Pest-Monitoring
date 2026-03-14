AgroVision AI – Vision System for Crop and Pest Monitoring

Overview

AgroVision AI is an artificial intelligence–based agricultural advisory system that helps farmers identify crop diseases, detect harmful pests, and recommend suitable crops based on soil parameters.

The system integrates multiple machine learning and deep learning models to provide intelligent insights for improving crop productivity and preventing crop damage.

The project consists of three main AI modules:
	•	Crop Disease Detection using deep learning
	•	Pest Detection using computer vision
	•	Soil-based Crop Recommendation using machine learning

These modules are combined using a decision engine that provides the final agricultural recommendation.



Features
	•	Crop disease detection from plant leaf images
	•	Pest detection from insect images
	•	Crop recommendation based on soil nutrient values
	•	Integrated decision engine to combine model outputs
	•	Modular architecture for easy model replacement or retraining

Project Structure:

AgroVision-AI
│
├── main.py
├── app/
│   └── decision_engine.py
│
├── models/
│   ├── crop_disease/
│   ├── pest/
│   └── soil/
│
├── notebooks/
│   ├── 1_crop_disease/
│   ├── 2_soil_analysis/
│   └── 3_pest_detection/
│
├── reports/
│
├── structure.txt
└── .gitignore
Explanation:
	•	main.py – Main application entry point
	•	decision_engine.py – Combines outputs of multiple models
	•	models/ – Contains trained model weights
	•	notebooks/ – Training scripts for the ML models
	•	reports/ – Experimental results and analysis


Dataset

Due to size limitations, the datasets are not included in this repository.
Download them from the following sources and place them inside the data/ folder.

Crop Disease Dataset

PlantVillage Dataset
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
Place inside:
data/crop_disease/
Pest Detection Dataset

Dangerous Insects Dataset
https://www.kaggle.com/datasets/tarundalal/dangerous-insects-dataset

Place inside:
data/pest/
Crop Recommendation Dataset

Crop Recommendation Dataset
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Place inside:
data/soil/
Installation

Clone the repository:
git clone https://github.com/ArjunKhimta/AgroVision-AI-AI-Vision-System-for-Crop-and-Pest-Monitoring.git
cd AgroVision-AI-AI-Vision-System-for-Crop-and-Pest-Monitoring
pip install -r requirements.txt

Running the Project

Run the main application:
python main.py
The system will:
	1.	Detect crop diseases from leaf images
	2.	Identify harmful pests
	3.	Recommend suitable crops based on soil conditions
	4.	Combine results using the decision engine

Models Used

Crop Disease Detection
	•	Deep Learning CNN model
	•	Trained on the PlantVillage dataset

Pest Detection
	•	Computer Vision classification model
	•	Trained on insect image dataset

Crop Recommendation
	•	Machine learning model trained on soil nutrient dataset

Soil parameters used include:
	•	Nitrogen
	•	Phosphorus
	•	Potassium
	•	Temperature
	•	Humidity
	•	pH



Technologies Used
	•	Python
	•	PyTorch
	•	Scikit-learn
	•	NumPy
	•	Pandas


Future Improvements
	•	Real-time camera based crop monitoring
	•	Mobile application for farmers
	•	Weather data integration
	•	Disease severity prediction


Authors:

Arjun Khimta, Akshit Sharma

This project was developed as part of an AI/ML system for crop health monitoring and agricultural decision support.


License

This project is intended for educational and research purposes.