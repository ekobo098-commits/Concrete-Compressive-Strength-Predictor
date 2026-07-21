# ==========================================================
# Concrete Compressive Strength Prediction using XGBoost
# Train Model and Save as .pkl File
# ==========================================================

# Install if required
# !pip install xgboost joblib

# =========================
# Import Libraries
# =========================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    explained_variance_score
)

from xgboost import XGBRegressor

# =========================
# Load Dataset
# =========================
df = pd.read_csv("2_concrete_data.csv")

# =========================
# Data Cleaning
# =========================

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Display dataset information
print("\nDataset Information")
print(df.info())

# Check missing values
print("\nMissing Values")
print(df.isnull().sum())

# Fill missing values (if any)
df.fillna(df.median(numeric_only=True), inplace=True)

# Remove duplicate rows
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

df.drop_duplicates(inplace=True)

print(f"\nDataset Shape after Cleaning: {df.shape}")

# =========================
# Feature Selection
# =========================
X = df.drop("concrete_compressive_strength", axis=1)
y = df["concrete_compressive_strength"]

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =========================
# Build XGBoost Model
# =========================
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# =========================
# Train Model
# =========================
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# =========================
# Make Predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# Evaluate Model
# =========================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
evs = explained_variance_score(y_test, y_pred)

print("\n==============================")
print("Model Evaluation")
print("==============================")
print(f"R² Score                 : {r2:.4f}")
print(f"Mean Absolute Error      : {mae:.4f}")
print(f"Mean Squared Error       : {mse:.4f}")
print(f"Root Mean Squared Error  : {rmse:.4f}")
print(f"MAPE                     : {mape:.2f}%")
print(f"Explained Variance Score : {evs:.4f}")

# Optional Regression Accuracy
regression_accuracy = 100 - mape
print(f"Regression Accuracy      : {regression_accuracy:.2f}%")

# =========================
# Save Model
# =========================
model_filename = "xgboost_concrete_model.pkl"

joblib.dump(model, model_filename)

print(f"\nModel successfully saved as '{model_filename}'")

# =========================
# Load Saved Model
# =========================
loaded_model = joblib.load(model_filename)

print("Saved model loaded successfully!")

# =========================
# Test Prediction
# =========================
sample_data = pd.DataFrame([{
    "cement": 540.0,
    "blast_furnace_slag": 0.0,
    "fly_ash": 0.0,
    "water": 162.0,
    "superplasticizer": 2.5,
    "coarse_aggregate": 1040.0,
    "fine_aggregate": 676.0,
    "age": 28
}])

prediction = loaded_model.predict(sample_data)

print("\n==============================")
print("Sample Prediction")
print("==============================")
print(f"Predicted Concrete Compressive Strength: {prediction[0]:.2f} MPa")