# Medical-NoShow-Prediction-with-SHAP

This repository contains a machine learning model to predict no-show appointments using the Kaggle Medical Appointment dataset and explains feature importance with SHAP.

## Requirements
- Python 3.x
- scikit-learn
- shap
- pandas
- matplotlib

## Usage
1. Clone the repository: `git clone <repository_url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the script: `python script.py`

## Dataset
The dataset is sourced from Kaggle (Medical Appointment No-Shows). A sample is included due to size limitations.

## Results
- Model: RandomForestClassifier
- SHAP analysis visualizes feature importance (e.g., DayDiff, Age).
