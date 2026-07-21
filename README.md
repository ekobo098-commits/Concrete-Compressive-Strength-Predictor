# 🏗️ Concrete Compressive Strength Predictor

### Machine Learning Framework for Predicting Concrete Compressive Strength Using XGBoost

An interactive **Streamlit** application that predicts the compressive strength of concrete based on mix proportions using a trained **XGBoost Regressor**. The application enables engineers, students, and construction professionals to estimate concrete strength before physical testing while maintaining a session prediction history for comparison and analysis.

---

## 📖 Overview

Concrete compressive strength is one of the most important quality indicators in construction and civil engineering. Laboratory testing requires curing specimens for several days before compressive strength can be measured.

This project demonstrates how Machine Learning can estimate concrete compressive strength directly from material composition and curing age, providing rapid decision support during concrete mix design.

The developed application combines an XGBoost prediction model with an interactive Streamlit dashboard that records prediction history and allows users to export prediction results for engineering analysis.

---

## 🎯 Project Objectives

- Predict concrete compressive strength (MPa)
- Analyze the influence of concrete mix proportions
- Provide an interactive engineering dashboard
- Maintain session prediction history
- Export prediction records as CSV
- Demonstrate practical Machine Learning applications in Civil Engineering

---

## 🧠 Machine Learning Model

**Model Used**

- XGBoost Regressor

### Why XGBoost?

XGBoost was selected because it provides excellent predictive performance for structured engineering datasets while effectively modeling nonlinear relationships between concrete mix components and compressive strength.

**Advantages**

- High prediction accuracy
- Handles nonlinear relationships
- Robust against overfitting
- Efficient training
- Excellent performance on tabular engineering datasets

---

## 📊 Dataset

This project uses the **Concrete Compressive Strength Dataset**.

### Input Features

| Feature | Unit |
|----------|------|
| Cement | kg/m³ |
| Blast Furnace Slag | kg/m³ |
| Fly Ash | kg/m³ |
| Water | kg/m³ |
| Superplasticizer | kg/m³ |
| Coarse Aggregate | kg/m³ |
| Fine Aggregate | kg/m³ |
| Age | Days |

### Target Variable

Concrete Compressive Strength (MPa)

---

## ⚙️ Machine Learning Workflow

```text
Concrete Dataset

        │

        ▼

Data Cleaning

        │

        ▼

Feature Engineering

        │

        ▼

Model Training

        │

        ▼

XGBoost Regression

        │

        ▼

Prediction

        │

        ▼

Interactive Dashboard

        │

        ▼

Prediction History

        │

        ▼

CSV Export
```

---

## 💻 Software Features

- Predict concrete compressive strength
- Interactive Streamlit dashboard
- Session prediction history
- Prediction logging
- CSV export
- Clear prediction history
- Simple engineering interface

---

# 🖥 Dashboard
<img width="1920" height="862" alt="concrete_strength_streamlit" src="https://github.com/user-attachments/assets/edda4fe1-73a1-4435-a981-5665b16b0458" />


---

## 📈 Model Evaluation

The XGBoost model was evaluated using standard regression metrics including:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Additional evaluation included:

- Actual vs Predicted Plot
- Feature Importance Analysis
- Correlation Heatmap
- Training vs Validation Performance

---

## 📁 Project Structure

```text
Concrete-Strength-Predictor/

│

├── app.py

├── model.py

├── xgboost_concrete_model.pkl

├── 2_concrete_data.csv

├── README.md

├── LICENSE

├── .gitignore

├── concrete(1).ipynb

├── concrete_actual.png

├── concrete_heatmap.png

├── concrete_top20.png

├── concrete_trainingvsvalidation.png
```

---

## 🛠 Technologies Used

Programming Language

- Python

Machine Learning

- XGBoost

Data Processing

- Pandas
- NumPy

Visualization

- Matplotlib

Web Application

- Streamlit

Model Storage

- Pickle

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/concrete-strength-predictor.git
```

Navigate into the project folder

```bash
cd concrete-strength-predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🏗 Practical Applications

This framework can support decision-making in several construction activities including:

- Concrete mix design
- Civil engineering education
- Quality estimation
- Construction material analysis
- Research and laboratory studies
- Preliminary engineering evaluation

---

## 🔮 Future Improvements

Potential future enhancements include:

- Batch prediction using CSV uploads
- Explainable AI (SHAP)
- Mix optimization recommendations
- PDF report generation
- Multi-model comparison
- Cloud deployment
- Integration with laboratory information systems

---

## 📄 License

This project is released under the MIT License.

---

## 👨‍💻 Author

**Sarthak Salve**

Industrial Artificial Intelligence • Civil Engineering • Machine Learning • Predictive Analytics
