import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("""
This app uses a trained machine learning model to predict your risk of heart disease.
Fill in your health details on the left and click **Predict** to see the result.
""")

st.sidebar.header("🩺 Patient Health Information")
data = {
        'age': st.sidebar.slider(
            'Age- in years', 25, 85, 50,
            help="Enter your current age in years."
        ),
        'sex': st.sidebar.selectbox(
            'Sex (1 = male, 0 = female)', [1, 0],
            help="Biological sex as recorded in medical history."
        ),
        'cp': st.sidebar.selectbox(
            'Chest Pain Type',
            ['typical angina: during activity', 'atypical angina: unusual pattern', 'non-anginal: not heart-related', 'asymptomatic: no pain'],
            help="What kind of pain or discomfort you experience in your chest."
        ),
        'trestbps': st.sidebar.slider(
            'Resting Blood Pressure (mm Hg)', 90, 200, 130,
            help="Resting blood pressure. Normal range is 120/80 mm Hg."
        ),
        'chol': st.sidebar.slider(
            'Cholesterol (mg/dl)', 100, 600, 250,
            help="Cholesterol level in the blood. Below 200 is generally desirable."
        ),
        'fbs': st.sidebar.selectbox(
            'Fasting Blood Sugar > 120 mg/dl (1 = yes, 0 = no)', [1, 0],
            help="Whether your fasting blood sugar is higher than 120 mg/dL."
        ),
        'restecg': st.sidebar.selectbox(
            'Resting ECG',
            ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'],
            help="Results from ECG test. Indicates any electrical activity issues of the heart."
        ),
        'thalach': st.sidebar.slider(
            'Max Heart Rate Achieved', 70, 210, 150,
            help="Highest heart rate recorded during exercise or stress test."
        ),
        'exang': st.sidebar.selectbox(
            'Exercise Induced Angina (1 = yes, 0 = no)', [0, 1],
            help="Did you experience chest pain (angina) during exercise?"
        ),
        'oldpeak': st.sidebar.slider(
            'ST depression induced by exercise', 0.0, 6.2, 1.0,
            help="Difference in ECG measurements before and after exercise. High values may indicate ischemia."
        ),
        'slope': st.sidebar.selectbox(
            'Slope of the peak ST segment', [0, 1, 2],
            help="Describes the shape of the ST segment during peak exercise on ECG."
        ),
        'ca': st.sidebar.selectbox(
            'Number of major vessels (0–3)', [0, 1, 2, 3],
            help="Number of major blood vessels showing blockage. Higher number = more severe."
        ),
        'thal': st.sidebar.selectbox(
            'Thalassemia',
            ['normal', 'fixed defect', 'reversible defect'],
            help="Indicates how well blood carries oxygen. Defects may affect heart function."
        )
}

user_input = pd.DataFrame([data])

user_input["sex"] = user_input["sex"].map({"Male": 1, "Female": 0})
user_input["fbs"] = user_input["fbs"].map({"Yes": "1", "No": "0"})
user_input["exang"] = user_input["exang"].map({"Yes": "1", "No": "0"})

input_encoded = pd.get_dummies(user_input)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

input_scaled = scaler.transform(input_encoded)

if st.button("🔍 Predict Heart Disease Risk"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1] * 100

    st.subheader("🧠 Prediction Result")
    st.write("**Probability of Heart Disease:** {:.2f}%".format(proba))

    if proba < 40:
        st.success("✅ Low risk")
    elif proba < 70:
        st.warning("⚠️ Moderate risk")
    else:
        st.error("🚨 High risk of heart disease")

    st.subheader("🧭 Risk Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Heart Disease Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "crimson"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 Patient Health Overview")
radar_mapping = [
    ("age", "Age"),
    ("trestbps", "Resting Blood Pressure"),
    ("chol", "Cholesterol Level"),
    ("thalach", "Max Heart Rate Achieved"),
    ("oldpeak", "ST Depression (Oldpeak)")
]
radar_labels = [label for _, label in radar_mapping]
radar_values = [user_input[col].values[0] for col, _ in radar_mapping]
radar_min = min(radar_values)
radar_max = max(radar_values)
radar_norm = [(val - radar_min) / (radar_max - radar_min) for val in radar_values]
radar_df = pd.DataFrame({"Metric": radar_labels, "Value": radar_norm})

fig_radar = px.line_polar(radar_df, r='Value', theta='Metric', line_close=True,
                           title="Normalized Health Indicators", range_r=[0, 1])
fig_radar.update_traces(
    line_color='blue',
    fill='toself',
    fillcolor='rgba(65,105,225,0.3)'
)
st.plotly_chart(fig_radar, use_container_width=True)
