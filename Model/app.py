import streamlit as st
import numpy as np
import os
import sys

# To import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Model.modeling import load_model  # import your load_model function

# Title
st.title("🔥 Calories Burnt Prediction App 💪")

# --- Load Model Bundle (Model + Scaler + Encoder) ---
@st.cache_resource
def load_all(model_name):
    return load_model(model_name)

# Model selection
model_name = st.selectbox("Choose Model to Load:", [
    "Linear Regression",
    "Ridge",
    "Random Forest Regressor",
    "SVR",
    "XGB Regressor"
])

model, scaler, encoder = load_all(model_name)

# === User Inputs ===
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=25)
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=1, max_value=200, value=70)
duration = st.number_input("Duration (minutes)", min_value=1, max_value=300, value=30)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=90)
body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)

# === Encoding Gender ===
# You used OneHotEncoder(drop='first'), so only 'Gender_male' is added
gender_male = 1 if gender == "Male" else 0

# === Prepare input ===
# Step 1: Separate numeric features (used in scaling)
numeric_input = np.array([[age, height, weight, duration, heart_rate, body_temp]])

# Step 2: Scale only numeric data
input_scaled = scaler.transform(numeric_input)

# Step 3: Add gender_male as last column (not scaled)
final_input = np.hstack((input_scaled, [[gender_male]]))

# === Predict ===
if st.button("Predict Calories Burnt"):
    prediction = model.predict(final_input)
    st.success(f"🔥 Estimated Calories Burnt: {prediction[0]:.2f} cal")
