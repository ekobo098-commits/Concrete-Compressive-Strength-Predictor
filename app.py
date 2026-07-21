import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

# --- App Title & Header ---
st.title("🏗️ Concrete Compressive Strength Predictor")
st.write(
    "Predict concrete compressive strength based on mix proportions using a trained **XGBoost Regressor**. "
    "Inputs are automatically appended to a running log for comparison and download."
)
st.write("---")

# --- 1. Load the Saved Model Safely ---
MODEL_PATH = r"D:\AiDL\concrete_data\xgboost_concrete_model.pkl"

@st.cache_resource
def load_concrete_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_concrete_model(MODEL_PATH)

# Fallback mechanism if the model file is missing
if model is None:
    st.error(f"❌ Could not find **'{MODEL_PATH}'** in the current directory.")
    st.info("Please make sure you run your training script first to generate the model file.")
    st.stop()

# --- 2. Initialize Session State for Data Logging ---
if 'history_df' not in st.session_state:
    # Explicitly matching column names from your notebook features + target
    st.session_state.history_df = pd.DataFrame(columns=[
        "cement", "blast_furnace_slag", "fly_ash", "water", 
        "superplasticizer", "coarse_aggregate", "fine_aggregate", "age",
        "predicted_strength_mpa"
    ])

# --- 3. Sidebar Input Elements ---
st.sidebar.header("🔧 Mix Proportion Inputs")
st.sidebar.write("Adjust parameters to calculate compressive strength:")

# Feature Inputs matched to your dataset bounds/defaults
cement = st.sidebar.number_input("Cement (kg/m³)", min_value=0.0, max_value=700.0, value=540.0, step=10.0)
blast_furnace_slag = st.sidebar.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, max_value=500.0, value=0.0, step=10.0)
fly_ash = st.sidebar.number_input("Fly Ash (kg/m³)", min_value=0.0, max_value=500.0, value=0.0, step=10.0)
water = st.sidebar.number_input("Water (kg/m³)", min_value=0.0, max_value=400.0, value=162.0, step=5.0)
superplasticizer = st.sidebar.number_input("Superplasticizer (kg/m³)", min_value=0.0, max_value=50.0, value=2.5, step=0.5)
coarse_aggregate = st.sidebar.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, max_value=1500.0, value=1040.0, step=10.0)
fine_aggregate = st.sidebar.number_input("Fine Aggregate (kg/m³)", min_value=0.0, max_value=1500.0, value=676.0, step=10.0)
age = st.sidebar.number_input("Age (days)", min_value=1, max_value=365, value=28, step=1)

# --- 4. Prediction Logic Loop ---
if st.sidebar.button("🔮 Predict Strength", type="primary"):
    # Constructing a clean DataFrame with identical names to ensure XGBoost maps columns correctly
    input_data = pd.DataFrame([{
        "cement": cement,
        "blast_furnace_slag": blast_furnace_slag,
        "fly_ash": fly_ash,
        "water": water,
        "superplasticizer": superplasticizer,
        "coarse_aggregate": coarse_aggregate,
        "fine_aggregate": fine_aggregate,
        "age": int(age)
    }])
    
    # Calculate Prediction
    prediction = model.predict(input_data)[0]
    formatted_prediction = round(float(prediction), 2)
    
    # Add predicted result value into the record payload
    input_data["predicted_strength_mpa"] = formatted_prediction
    
    # Append the record into our dynamic logging table
    st.session_state.history_df = pd.concat([st.session_state.history_df, input_data], ignore_index=True)
    
    # Output metrics to UI dashboard
    st.markdown(f"### 🎉 Result: **{formatted_prediction} MPa**")

# --- 5. Data History Dashboard Panel ---
st.subheader("📋 Session Prediction Logs")

if not st.session_state.history_df.empty:
    # Render scannable table tracking performance
    st.dataframe(st.session_state.history_df, use_container_width=True)
    
    action_col1, action_col2 = st.columns([1, 5])
    
    with action_col1:
        # Create standard file byte download pipeline
        csv_buffer = st.session_state.history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer,
            file_name="concrete_strength_predictions.csv",
            mime="text/csv"
        )
        
    with action_col2:
        # User dynamic log wipe switch
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history_df = pd.DataFrame(columns=[
                "cement", "blast_furnace_slag", "fly_ash", "water", 
                "superplasticizer", "coarse_aggregate", "fine_aggregate", "age",
                "predicted_strength_mpa"
            ])
            st.hybrid_fallback_rerun() if hasattr(st, "hybrid_fallback_rerun") else st.rerun()
else:
    st.info("Adjust inputs in the left sidebar and click **Predict Strength** to log calculations here.")