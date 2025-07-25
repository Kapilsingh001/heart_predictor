import streamlit as st
import numpy as np
import os
import pickle

model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
model = pickle.load(open(model_path, 'rb'))


st.title("💓 Heart Disease Prediction App")
st.write("Fill the details below to check your risk of heart disease.")

# ----------- Input fields ------------ #

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                  format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x])

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                   format_func=lambda x: "No (≤ 120 mg/dl)" if x == 0 else "Yes (> 120 mg/dl)")

restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                       format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])

thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)

exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                     format_func=lambda x: "No" if x == 0 else "Yes")

oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

slope = st.selectbox("Slope of the ST Segment", options=[0, 1, 2],
                     format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])

ca = st.selectbox("Number of Major Vessels (0–3)", options=[0, 1, 2, 3])

thal = st.selectbox("Thalassemia", options=[1, 2, 3],
                    format_func=lambda x: ["Normal", "Fixed defect", "Reversible defect"][x - 1])

# ------------- Prediction Section ------------- #

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

if st.button("Predict"):
    # Get prediction and probability
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of class 1 (disease)

    st.markdown(f"### 🧮 Probability of Heart Disease: **{probability * 100:.2f}%**")

    if prediction == 1:
        st.error("❗ High risk of heart disease.")
    else:
        st.success("✅ Low risk of heart disease.")
        
# -----------------Disclaimer-------------#        
st.markdown("⚠️ **Disclaimer:** This app is a machine learning demonstration and not a medical diagnostic tool. Always consult a healthcare professional.")
        