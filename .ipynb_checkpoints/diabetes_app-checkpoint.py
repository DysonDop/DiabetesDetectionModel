
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import io
import random

st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="wide", initial_sidebar_state="expanded")

quotes = [
    "“Take care of your body. It’s the only place you have to live.” – Jim Rohn",
    "“Health is not valued until sickness comes.” – Thomas Fuller",
    "“It is health that is real wealth and not pieces of gold and silver.” – Mahatma Gandhi",
    "“Your body hears everything your mind says.” – Naomi Judd",
    "“A fit body, a calm mind, a house full of love. These things cannot be bought – they must be earned.” – Naval Ravikant",
    "“An ounce of prevention is worth a pound of cure.” – Benjamin Franklin",
    "“The greatest wealth is health.” – Virgil",
    "“Don’t dig your grave with your own knife and fork.” – English Proverb",
    "“Discipline is the bridge between goals and accomplishment.” – Jim Rohn",
    "“Self-care is not a luxury—it is a necessity.” – Audre Lorde"
]

with open("Diabetesmodel.pkl", "rb") as f:
    model = pickle.load(f)

dataset = pd.read_csv("diabetes.csv")

st.sidebar.success("Welcome to the AI Diabetes Risk Assessment App! 👋")

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

if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = False

tab_input, tab_info, tab_history, tab_tips = st.tabs(["Input", "Info", "History", "💡 Health Tips"])

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

with tab_info:
    st.header("Understanding the Inputs")
    st.markdown("""
    - **Glucose:** Elevated glucose may indicate diabetes.  
    - **Blood Pressure:** High BP increases diabetes complications.  
    - **BMI:** Obesity is a strong diabetes risk factor.  
    - **Age:** Risk increases after age 45.  
    """)

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

with tab_tips:
    st.header("💡 Personalised Health Tips")
    if not st.session_state.last_prediction:
        st.info("Make a prediction to get personalised tips.")
    else:
        st.markdown(f"""
        <div style='border-left: 5px solid #4CAF50;
                    padding: 15px;
                    background-color: #f0f0f0;
                    color: #333;
                    font-style: italic;
                    font-size: 16px;
                    margin-bottom: 20px;'>
        💬 <b>Motivational Quote:</b><br>
        {random.choice(quotes)}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 🧪 Based on Your Results")

        if glucose > 180:
            st.error("❗ Very High Glucose: Consult a doctor.")
        elif glucose > 140:
            st.warning("🔴 High Glucose: Limit sugar and carbs.")
        elif glucose < 70:
            st.info("🧃 Low Glucose: Eat a healthy snack.")

        if blood_pressure > 140:
            st.error("💣 Very High Blood Pressure: Requires medical attention.")
        elif blood_pressure > 130:
            st.warning("🧠 High Blood Pressure: Reduce sodium and stress.")
        elif blood_pressure < 60:
            st.info("💧 Low Blood Pressure: Stay hydrated.")

        if bmi > 35:
            st.error("⚠️ Severely Obese: Consult a specialist.")
        elif bmi > 30:
            st.warning("📉 Obese: Improve diet and exercise.")
        elif bmi > 25:
            st.info("🚶 Overweight: Start light workouts.")
        elif bmi < 18.5:
            st.info("🍽️ Underweight: Increase nutritious calorie intake.")

        if age > 50:
            st.info("👴 Over 50? Get screened regularly.")

        st.markdown("## 🌿 General Wellness Tips")
        st.success("""
        ✅ Stay hydrated  
        ✅ Exercise 150 minutes/week  
        ✅ Sleep 7–8 hours/night  
        ✅ Eat fibre-rich foods  
        ✅ Avoid sugary drinks  
        ✅ Don’t skip meals  
        ✅ Manage stress  
        ✅ Take breaks from screens  
        ✅ Practise mindful eating  
        ✅ Control portion sizes  
        ✅ Know your family health history  
        ✅ Get regular check-ups  
        """)

        st.image("https://cdn-icons-png.flaticon.com/512/2917/2917991.png", width=100, caption="Stay healthy, stay happy! 🎉")
