# ❤️ Heart Disease Risk Predictor

This web app uses a trained machine learning model to predict the risk of heart disease based on patient health parameters.

Powered by logistic regression and built with [Streamlit](https://streamlit.io), the app is interactive, intuitive, and provides real-time risk estimates.

Author: Priyanshu  
Date: July 2025  
Submission for: Week‑7 Assignment

---

## 🚀 Try the App

**[🔗 Click here to use the app](https://heart-disease-predictor-app-csi7.streamlit.app/)**  

---

## How It Works

### Input:
The user fills out a simple form with values such as:
- Age, Cholesterol, Blood Pressure, Chest pain type, Heart rate, etc.

### Prediction:
The trained logistic regression model predicts:
- Probability of heart disease (0–100%)

### Visualisations:
- A **risk gauge** shows your predicted risk
- A **radar chart** visualizes key health indicators

---

## Files Included

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app |
| `best_model.pkl` | Trained logistic regression model |
| `scaler.pkl` | StandardScaler for input preprocessing |
| `feature_columns.pkl` | Feature column names for input alignment |
| `requirements.txt` | Required Python libraries |
| `heart_disease_uci.csv` | Dataset used for model training |

---

## 🧠 Model Details

- **Model**: Logistic Regression
- **Trained on**: Public heart disease dataset (UCI-style)
- **Preprocessing**:
  - One-hot encoding of categorical features
  - Feature scaling using StandardScaler

---

## 🛠️ Setup Locally

```bash
git clone https://github.com/priyanshhu04/heart-disease-predictor.git
cd heart-disease-predictor-app
pip install -r requirements.txt
streamlit run app.py
