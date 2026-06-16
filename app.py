import streamlit as st
import pickle
import numpy as np

# Load both models
linear_model = pickle.load(open("linear_models.pkl", "rb"))
rf_model = pickle.load(open("random_forest_ml.pkl", "rb"))

st.title("Concrete Compressive Strength Predictor")

cement = st.number_input("Cement (kg/m³)", 0.0, 600.0, 300.0)
slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 400.0, 0.0)
flyash = st.number_input("Fly Ash (kg/m³)", 0.0, 300.0, 0.0)
water = st.number_input("Water (kg/m³)", 100.0, 300.0, 180.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 50.0, 5.0)
coarse = st.number_input("Coarse Aggregate (kg/m³)", 700.0, 1200.0, 1000.0)
fine = st.number_input("Fine Aggregate (kg/m³)", 500.0, 1000.0, 750.0)
age = st.number_input("Age (days)", 1, 365, 28)

if st.button("Predict Strength"):

    data = np.array([[
        cement,
        slag,
        flyash,
        water,
        superplasticizer,
        coarse,
        fine,
        age
    ]])

    # Predictions
    linear_prediction = linear_model.predict(data)
    rf_prediction = rf_model.predict(data)

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Linear Regression",
            value=f"{linear_prediction[0]:.2f} MPa"
        )

    with col2:
        st.metric(
            label="Random Forest",
            value=f"{rf_prediction[0]:.2f} MPa"
        )