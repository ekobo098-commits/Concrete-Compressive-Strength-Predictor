# 🏗️ Concrete Compressive Strength Predictor

Machine Learning web application built with **Streamlit** and **Scikit-learn** that predicts the compressive strength of concrete based on its material composition.

The application compares predictions from **Linear Regression** and **Random Forest Regression** models, allowing users to observe the performance of different machine learning algorithms on the same input data.

---

## 📌 Features

* Predict concrete compressive strength (MPa)
* Compare results from two machine learning models:

  * Linear Regression
  * Random Forest Regression
* Simple and interactive Streamlit interface
* Real-world civil engineering application
* Easy to run locally and deploy on Streamlit Community Cloud

---

## 📊 Dataset

This project uses the **Concrete Compressive Strength Dataset** from Kaggle/UCI.

### Input Features

| Feature            | Unit  |
| ------------------ | ----- |
| Cement             | kg/m³ |
| Blast Furnace Slag | kg/m³ |
| Fly Ash            | kg/m³ |
| Water              | kg/m³ |
| Superplasticizer   | kg/m³ |
| Coarse Aggregate   | kg/m³ |
| Fine Aggregate     | kg/m³ |
| Age                | Days  |

### Target

* Concrete Compressive Strength (MPa)

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* NumPy
* Pandas
* Pickle

---

## 📁 Project Structure

```
Concrete-Strength-Predictor/
│
├── app.py
├── train_linear.py
├── train_random_forest.py
├── linear_model.pkl
├── random_forest_model.pkl
├── concrete.csv
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/concrete-strength-predictor.git
```

Navigate to the project folder:

```bash
cd concrete-strength-predictor
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

## 💻 How to Use

1. Enter the concrete mix proportions.
2. Provide the curing age (days).
3. Click **Predict Strength**.
4. View predictions from:

   * Linear Regression Model
   * Random Forest Regression Model

---

## 📈 Machine Learning Models

### Linear Regression

A simple regression model that assumes a linear relationship between input variables and compressive strength.

**Advantages**

* Fast
* Easy to interpret
* Good baseline model

### Random Forest Regression

An ensemble learning method that combines multiple decision trees for improved prediction accuracy.

**Advantages**

* Handles nonlinear relationships
* More robust to outliers
* Generally produces higher accuracy

---

## 🎯 Future Improvements

* Add XGBoost and Gradient Boosting models
* Upload CSV files for batch predictions
* Display model evaluation metrics (R², MAE, RMSE)
* Visualize feature importance
* Plot prediction comparisons
* Deploy using Streamlit Community Cloud


## 👨‍💻 Author: Sarthak Salve

Created as a Civil Engineering + Machine Learning project using Streamlit and Scikit-learn.

Feel free to fork this repository, improve the models, and contribute.
