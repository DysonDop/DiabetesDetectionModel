import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import io
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="wide", initial_sidebar_state="expanded")

# --- AUTHENTICATION ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    key=config['cookie']['key'],
    expiry_days=config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login('Login', 'sidebar')

# --- LOGIN HANDLING ---
if authentication_status is False:
    st.title("🔒 Access Denied")
    st.error("Invalid username or password.")
    st.stop()

elif authentication_status is None:
    st.title("🔐 Please Log In")
    st.info("Use the sidebar to log in and access the app.")
    st.stop()

authenticator.logout("Logout", "sidebar")
st.sidebar.success(f"Welcome, {name} 👋")

# --- LOAD MODEL & DATA ---
with open("Diabetesmodel.pkl", "rb") as f:
    model = pickle.load(f)

dataset = pd.read_csv("diabetes.csv")

# --- DARK MODE ---
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
if dark_mode:
    st.markdown("""
    <style>
    .reportview-container, .sidebar .sidebar-content {
        background-color: #121212;
        color: white;
    }
    .stButton>button {
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .reportview-container, .sidebar .sidebar-content {
        background-color: white;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR INFO ---
st.title("🩺 AI Diabetes Risk Assessment")
with st.sidebar:
    st.header("Input Guidelines")
    st.markdown("""
    - **Glucose:** 70–140  
    - **Blood Pressure:** 60–120  
    - **BMI:** 18.5–24.9  
    - **Age:** 1–120  
    _These are general health ranges._
    """)
    show_advanced = st.checkbox("Show Advanced Mode (Visualizations)", value=True)

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = False

# --- TABS ---
tab_input, tab_info, tab_viz, tab_history, tab_tips = st.tabs(["Input", "Info", "Visualizations", "History", "💡 Health Tips"])

# --- INPUT TAB ---
with tab_input:
    col1, col2 = st.columns([3, 1])

    with col1:
        glucose = st.number_input("Glucose", 40.0, 300.0, 100.0)
        blood_pressure = st.number_input("Blood Pressure", 20.0, 200.0, 70.0)
        weight = st.number_input("Weight (kg)", 20.0, 200.0, 70.0)
        height_cm = st.number_input("Height (cm)", 100.0, 220.0, 170.0)
        bmi = weight / ((height_cm / 100) ** 2)
        st.metric("Calculated BMI", f"{bmi:.2f}")

        if bmi < 18.5:
            bmi_status = "Underweight"
            bmi_color = "orange"
            bmi_icon = "⚠️"
        elif bmi < 25:
            bmi_status = "Healthy"
            bmi_color = "green"
            bmi_icon = "✅"
        elif bmi < 30:
            bmi_status = "Overweight"
            bmi_color = "darkorange"
            bmi_icon = "⚠️"
        else:
            bmi_status = "Obese"
            bmi_color = "red"
            bmi_icon = "🚨"

        st.markdown(f"<div style='color:{bmi_color}; font-weight:bold;'>BMI Category: {bmi_status} {bmi_icon}</div>", unsafe_allow_html=True)

        age = st.number_input("Age", 1, 120, 30)

    with col2:
        st.markdown("### Decision Threshold")
        threshold = st.slider("", 0, 100, 50)

    if st.button("Predict"):
        errors = []
        if not (40 <= glucose <= 300): errors.append("Glucose must be between 40–300.")
        if not (20 <= blood_pressure <= 200): errors.append("Blood Pressure must be between 20–200.")
        if not (10 <= bmi <= 70): errors.append("BMI must be between 10–70.")
        if not (1 <= age <= 120): errors.append("Age must be between 1–120.")

        if errors:
            for err in errors: st.error(err)
        else:
            features = np.array([[glucose, blood_pressure, bmi, age]])
            proba = model.predict_proba(features)[0][1] * 100
            prediction = 1 if proba >= threshold else 0
            risk = "Low" if proba < 30 else "Medium" if proba < 70 else "High"

            if prediction == 1:
                st.warning(f"⚠️ Likely diabetic.\nConfidence: {proba:.2f}% (Threshold: {threshold}%)\nRisk: {risk}")
            else:
                st.success(f"✅ Not likely diabetic.\nConfidence: {100 - proba:.2f}% (Threshold: {threshold}%)\nRisk: {risk}")

            st.session_state.last_prediction = True
            st.session_state.history.append({
                "Glucose": glucose, "Blood Pressure": blood_pressure, "BMI": bmi, "Age": age,
                "Prediction": "Likely" if prediction else "Not Likely",
                "Confidence": f"{proba:.2f}%", "Threshold": f"{threshold}%", "Risk Level": risk
            })

            # Download report
            report = f"""AI Diabetes Risk Report
-----------------------
Glucose: {glucose}
Blood Pressure: {blood_pressure}
BMI: {bmi:.2f} ({bmi_status})
Age: {age}
Decision Threshold: {threshold}%
Prediction: {"Likely" if prediction else "Not Likely"}
Confidence: {proba:.2f}%
Risk Level: {risk}
"""
            st.download_button("📄 Download Report (.txt)", report, file_name="diabetes_report.txt")

# --- INFO TAB ---
with tab_info:
    st.header("Understanding the Inputs")
    st.markdown("""
    - **Glucose:** Elevated glucose may indicate diabetes.  
    - **Blood Pressure:** High BP increases diabetes complications.  
    - **BMI:** Obesity is a strong diabetes risk factor.  
    - **Age:** Risk increases after age 45.  
    """)

# --- VISUALIZATION TAB ---
if show_advanced:
    with tab_viz:
        st.header("Dataset Visualizations")
        st.subheader("Heatmap of Feature Correlations")
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", ax=ax1)
        st.pyplot(fig1)

        st.subheader("Histograms")
        features_to_plot = ['Glucose', 'BloodPressure', 'BMI', 'Age']
        fig2, axs2 = plt.subplots(len(features_to_plot), 1, figsize=(8, 12))
        for i, feature in enumerate(features_to_plot):
            sns.histplot(dataset[feature], kde=True, ax=axs2[i])
            axs2[i].set_title(f"{feature} Distribution")
        plt.tight_layout()
        st.pyplot(fig2)

        st.subheader("Boxplot Overview")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=dataset[features_to_plot], ax=ax3)
        st.pyplot(fig3)

        st.subheader("Scatterplot: Glucose vs BMI")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="Glucose", y="BMI", hue="Outcome", data=dataset, ax=ax4)
        st.pyplot(fig4)

# --- HISTORY TAB ---
with tab_history:
    st.header("Prediction History")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history)
        csv_buf = io.StringIO()
        df_history.to_csv(csv_buf, index=False)
        st.download_button("Download CSV", csv_buf.getvalue(), "diabetes_history.csv", "text/csv")
    else:
        st.info("No predictions yet.")

# --- TIPS TAB ---
with tab_tips:
    st.header("💡 Personalised Health Tips")
    if not st.session_state.last_prediction:
        st.info("Make a prediction to get tips.")
    else:
        st.markdown("### Based on your input:")
        if glucose > 140: st.warning("🔴 High Glucose: Cut sugar, stay active.")
        if blood_pressure > 130: st.warning("🧠 High BP: Limit salt, reduce stress.")
        if bmi > 25: st.warning("⚖️ High BMI: Adopt a healthier diet and exercise.")
        if age > 50: st.info("👴 Age > 50: Get screened regularly.")
        st.markdown("### General Tips:")
        st.success("✅ Drink water\n✅ Exercise daily\n✅ Sleep 7–8 hrs\n✅ Eat fibre\n✅ Monitor sugar")
