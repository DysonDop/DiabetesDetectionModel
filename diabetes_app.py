import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import io

# Page config: MUST be first Streamlit command
st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="wide", initial_sidebar_state="expanded")

# Load your trained model
with open("Diabetesmodel.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset for visualization
dataset = pd.read_csv("diabetes.csv")

# Dark mode toggle (simple CSS switch)
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
if dark_mode:
    st.markdown(
        """
        <style>
        .reportview-container, .sidebar .sidebar-content {
            background-color: #121212;
            color: white;
        }
        .stButton>button {
            color: black;
        }
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .reportview-container, .sidebar .sidebar-content {
            background-color: white;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True
    )

st.title("🩺 AI Diabetes Risk Assessment")

# Sidebar info & input guidelines
with st.sidebar:
    st.header("Input Guidelines")
    st.markdown("""
    - **Glucose:** Typical healthy range 70–140  
    - **Blood Pressure:** Typical healthy range 60–120  
    - **BMI:** Typical healthy range 18.5–24.9  
    - **Age:** Typical range 1–120  

    _Ranges are approximate and for guidance only._
    """)
    st.markdown("This app uses a machine learning model trained on a diabetes dataset to predict your risk.")

# Initialize session state for history if not already done
if "history" not in st.session_state:
    st.session_state.history = []

# Tabs
tab_input, tab_info, tab_viz, tab_history = st.tabs(["Input", "Info", "Visualizations", "History"])

# Input Tab
with tab_input:
    col1, col2 = st.columns([3, 1])

    with col1:
        glucose = st.number_input("Glucose", min_value=40.0, max_value=300.0, value=100.0, format="%.1f")
        blood_pressure = st.number_input("Blood Pressure", min_value=20.0, max_value=200.0, value=70.0, format="%.1f")
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, format="%.1f")
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

    with col2:
        st.markdown("### Decision Threshold")
        threshold = st.slider("", min_value=0, max_value=100, value=50, help="Adjust sensitivity of diabetes prediction.")

    if st.button("Predict"):
        errors = []
        if glucose < 40 or glucose > 300:
            errors.append("Glucose must be between 40 and 300.")
        if blood_pressure < 20 or blood_pressure > 200:
            errors.append("Blood Pressure must be between 20 and 200.")
        if bmi < 10 or bmi > 70:
            errors.append("BMI must be between 10 and 70.")
        if age < 1 or age > 120:
            errors.append("Age must be between 1 and 120.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            features = np.array([[glucose, blood_pressure, bmi, age]])
            proba = model.predict_proba(features)[0][1] * 100
            prediction = 1 if proba >= threshold else 0

            # Risk category based on confidence
            if proba < 30:
                risk = "Low"
            elif proba < 70:
                risk = "Medium"
            else:
                risk = "High"

            if prediction == 1:
                st.warning(f"⚠️ You are likely to have diabetes.\nConfidence: {proba:.2f}% (Threshold: {threshold}%)\nRisk Level: {risk}")
            else:
                st.success(f"✅ You are not likely to have diabetes.\nConfidence: {100 - proba:.2f}% (Threshold: {threshold}%)\nRisk Level: {risk}")

            # Save prediction to history
            st.session_state.history.append({
                "Glucose": glucose,
                "Blood Pressure": blood_pressure,
                "BMI": bmi,
                "Age": age,
                "Prediction": "Likely" if prediction == 1 else "Not Likely",
                "Confidence": f"{proba:.2f}%",
                "Threshold": f"{threshold}%",
                "Risk Level": risk
            })

# Info Tab
with tab_info:
    st.header("Understanding the Inputs")
    st.markdown("""
    **Glucose:**  
    Higher glucose levels often indicate an increased risk of diabetes.  
    Normal fasting glucose is usually between 70–140 mg/dL.

    **Blood Pressure:**  
    Elevated blood pressure can contribute to complications related to diabetes.  
    Healthy blood pressure generally ranges from 60–120 mmHg.

    **BMI (Body Mass Index):**  
    BMI indicates body fat. Higher BMI (>25) can increase diabetes risk.  
    Healthy BMI range: 18.5–24.9.

    **Age:**  
    Risk of diabetes tends to increase with age, especially after 45.

    > This app uses a machine learning model trained on a diabetes dataset to predict your risk based on these inputs.
    """)

# Visualization Tab
with tab_viz:
    st.header("Dataset Visualizations")

    st.subheader("Heatmap of Feature Correlations")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Histogram of Features")
    features_to_plot = ['Glucose', 'BloodPressure', 'BMI', 'Age']
    fig2, axs2 = plt.subplots(len(features_to_plot), 1, figsize=(8, 3 * len(features_to_plot)))
    for i, feature in enumerate(features_to_plot):
        sns.histplot(dataset[feature], kde=True, ax=axs2[i])
        axs2[i].set_title(f"Distribution of {feature}")
    plt.tight_layout()
    st.pyplot(fig2)

    st.subheader("Boxplot of Features")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=dataset[features_to_plot], ax=ax3)
    st.pyplot(fig3)

    st.subheader("Scatterplot of Glucose vs. BMI colored by Outcome")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="Glucose", y="BMI", hue="Outcome", data=dataset, ax=ax4)
    st.pyplot(fig4)

# History Tab
with tab_history:
    st.header("Prediction History")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history)

        # Download CSV button
        csv_buffer = io.StringIO()
        df_history.to_csv(csv_buffer, index=False)
        st.download_button("Download History as CSV", data=csv_buffer.getvalue(), file_name="diabetes_prediction_history.csv", mime="text/csv")
    else:
        st.info("No prediction history yet. Make a prediction to see it here.")
