import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import re
import sys

# Load Models
rf_model = joblib.load("random_forest_model.pkl")
dnn_model = tf.keras.models.load_model("dnn_model.h5")
lr_model = joblib.load("logistic_regression_model.pkl")  # Load Logistic Regression
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def is_mental_health_related(user_input):
    """
    Checks if the input is related to mental health using keyword matching.
    """
    mental_health_keywords = [
        "anxiety", "depression", "stress", "mental health", "therapy", "panic", 
        "sad", "overthinking", "mood swings", "insomnia", "fatigue", "hopeless",
        "self-harm", "suicide", "psychologist", "counseling", "trauma"
    ]
    
    user_input = user_input.lower()

    for keyword in mental_health_keywords:
        if keyword in user_input:
            return True
    return False

def extract_features(user_input):
    """
    Extracts structured features from a natural language input.
    """
    user_input = user_input.lower()

    gender_map = {"male": 1, "female": 0, "other": 2}
    
    # Extract age
    age_match = re.search(r"\b(\d{1,2})\s*years?\b", user_input)
    age = int(age_match.group(1)) if age_match else 30  # Default age 30

    # Extract gender
    gender = 2  # Default "other"
    for key in gender_map:
        if key in user_input:
            gender = gender_map[key]
            break

    # Extract binary responses
    family_history = 1 if "family history" in user_input and "yes" in user_input else 0
    benefits = 1 if "benefits" in user_input and "yes" in user_input else 0
    care_options = 2 if "care options" in user_input and "yes" in user_input else 0
    anonymity = 1 if "anonymity" in user_input and "yes" in user_input else 0
    leave = 1 if "leave" in user_input and "yes" in user_input else 0
    work_interfere = 2 if "work interfere" in user_input and "yes" in user_input else 0

    return {
        "Age": age,
        "Gender": gender,
        "family_history": family_history,
        "benefits": benefits,
        "care_options": care_options,
        "anonymity": anonymity,
        "leave": leave,
        "work_interfere": work_interfere
    }

def get_natural_language_output(predictions):
    """
    Converts model predictions into a natural language response.
    """
    responses = {
        "Likely Treatment Needed": "Based on your symptoms, seeking professional mental health support may be beneficial.",
        "Unlikely Treatment Needed": "Your symptoms do not strongly indicate a need for mental health treatment, but self-care is important.",
    }

    response_text = "\n🔍 **Mental Health Assessment:**\n"
    for model, result in predictions.items():
        if result in responses:
            response_text += f"🧠 {model}: {responses[result]}\n"
    
    return response_text

def predict(user_input):
    """
    Takes natural language input and predicts mental health condition using all models.
    If input is unrelated to mental health, it returns a rejection message.
    """

    if not is_mental_health_related(user_input):
        return "\n⚠️ Your input does not seem related to mental health. Please describe symptoms like anxiety, stress, or depression."

    symptoms = extract_features(user_input)
    input_df = pd.DataFrame([symptoms])
    input_scaled = scaler.transform(input_df)

    # Predict using all models
    rf_pred = rf_model.predict(input_df)[0]
    dnn_pred = np.argmax(dnn_model.predict(input_scaled), axis=1)[0]
    lr_pred = lr_model.predict(input_scaled)[0]

    # Convert numerical predictions (0/1) to readable labels
    label_map = {0: "Unlikely Treatment Needed", 1: "Likely Treatment Needed"}

    rf_pred_label = label_map.get(rf_pred, "Unknown")
    dnn_pred_label = label_map.get(dnn_pred, "Unknown")
    lr_pred_label = label_map.get(lr_pred, "Unknown")

    predictions = {
        "Random Forest Prediction": rf_pred_label,
        "DNN Prediction": dnn_pred_label,
        "Logistic Regression Prediction": lr_pred_label
    }

    return get_natural_language_output(predictions)

if __name__ == "__main__":
    user_text = input("📝 Describe your symptoms: ")
    result = predict(user_text)
    print(result)
