import streamlit as st
import numpy as np
import joblib

# Load the dictionary
saved = joblib.load("liver_model.pkl")
model = saved["model"]
scaler = saved["scaler"]
threshold = saved["threshold"]

st.title("🧪 Liver Disease Prediction App")

# Input fields
age = st.number_input("Age", 1, 120, 30)
gender = st.selectbox("Gender", ["Female", "Male"])
gender = 1 if gender == "Male" else 0
tb = st.number_input("Total Bilirubin", 0.0, 75.0, 1.0)
db = st.number_input("Direct Bilirubin", 0.0, 20.0, 0.5)
alp = st.number_input("Alkaline Phosphotase", 50, 2000, 200)
alt = st.number_input("Alamine Aminotransferase (SGPT)", 0, 5000, 30)
ast = st.number_input("Aspartate Aminotransferase (SGOT)", 0, 5000, 30)
tp = st.number_input("Total Proteins", 0.0, 10.0, 6.5)
alb = st.number_input("Albumin", 0.0, 6.0, 3.0)
agr = st.number_input("Albumin and Globulin Ratio", 0.0, 3.0, 1.0)

if st.button("Predict"):
    features = np.array([[age, gender, tb, db, alp, alt, ast, tp, alb, agr]])
    features_scaled = scaler.transform(features)

    # Predict probabilities
    proba = model.predict_proba(features_scaled)[0, 1]
    pred = 1 if proba >= threshold else 0

    if pred == 1:
        st.error(f"⚠️ Positive (Liver Disease)\nConfidence: {proba:.4f}")
    else:
        st.success(f"✅ Negative (No Liver Disease)\nConfidence: {1-proba:.4f}")

