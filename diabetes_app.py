import streamlit as st
import numpy as np
import pandas as pd
import pickle
import io
import random

# Load your trained model
with open("Diabetesmodel.pkl", "rb") as f:
    model = pickle.load(f)

# Health tips quotes
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

# Page config
st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="wide")

# Custom CSS
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    font-size: 17px;
    line-height: 1.6;
}
h1, h2, h3 {
    color: #333;
    font-weight: 700;
}
input[type="number"], input[type="text"] {
    border-radius: 8px;
    padding: 10px;
    font-size: 16px;
}
.stButton > button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    padding: 10px 24px;
    border: none;
    border-radius: 8px;
}
.stButton > button:hover {
    background-color: #388e3c;
}
.stDownloadButton > button {
    background-color: #2196f3;
    color: white;
    font-size: 16px;
    padding: 10px 18px;
    border-radius: 6px;
}
.stDownloadButton > button:hover {
    background-color: #1976d2;
}
[data-testid="stMetric"] {
    background: #d9fdd3;
    padding: 12px 18px;
    border-radius: 10px;
    font-weight: bold;
    color: #2e7d32;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    margin-top: 10px;
}
div[role="tablist"] > div {
    font-size: 18px;
}
div[style*='border-left: 5px solid'] {
    border-radius: 8px !important;
    background-color: #f9f9f9 !important;
    box-shadow: 0 0 10px rgba(0,0,0,0.03);
}
</style>
""", unsafe_allow_html=True)

# Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = False

# Risk Summary Generator
def generate_risk_summary(glucose, bp, bmi, age, risk):
    advice = []
    if glucose > 140:
        advice.append("• Your glucose is elevated; consider monitoring your sugar intake.")
    if bp > 120:
        advice.append("• Your blood pressure is high; reducing salt and stress may help.")
    if bmi > 25:
        advice.append("• Your BMI indicates overweight; increasing physical activity can help.")
    if age > 45:
        advice.append("• Age increases your risk; regular screening is recommended.")
    if not advice:
        advice.append("• Your inputs are within healthy ranges. Keep up the good lifestyle!")
    return f"### Overall Risk Level: {risk}\n\n" + "\n".join(advice) + "\n\nPlease consult your healthcare provider for tailored advice."

# Title
st.title("🩺 AI Diabetes Risk Assessment")

# Sidebar
with st.sidebar:
    st.header("Input Guidelines")
    st.markdown("""
    - **Glucose:** 70–140  
    - **Blood Pressure:** 60–120  
    - **BMI:** 18.5–24.9  
    - **Age:** 1–120  
    _These are general health ranges._
    """)

# Tabs
tab_input, tab_info, tab_history, tab_tips = st.tabs(["📥 Input", "📖 Info", "📊 History", "💡 Health Tips"])

# INPUT TAB
with tab_input:
    col1, col2 = st.columns([3, 1])

    with col1:
        user_name = st.text_input("Enter Your Name", "")
        glucose = st.number_input("Glucose", 40.0, 300.0, 100.0)
        blood_pressure = st.number_input("Blood Pressure", 20.0, 200.0, 70.0)
        weight = st.number_input("Weight (kg)", 20.0, 200.0, 70.0)
        height_cm = st.number_input("Height (cm)", 100.0, 220.0, 170.0)

        bmi = weight / ((height_cm / 100) ** 2)

        # BMI Classification
        if bmi < 18.5:
            bmi_label = "Underweight"
            bmi_color = "#ffe082"
            bmi_icon = "🍽️"
            bmi_tip = "Eat more nutritious calories."
        elif bmi < 25:
            bmi_label = "Normal"
            bmi_color = "#d9fdd3"
            bmi_icon = "✅"
            bmi_tip = "You're in a healthy range. Keep it up!"
        elif bmi < 30:
            bmi_label = "Overweight"
            bmi_color = "#fff59d"
            bmi_icon = "🚶"
            bmi_tip = "Consider light exercise and balanced diet."
        elif bmi < 35:
            bmi_label = "Obese"
            bmi_color = "#ffccbc"
            bmi_icon = "📉"
            bmi_tip = "Improve diet and increase physical activity."
        else:
            bmi_label = "Severely Obese"
            bmi_color = "#ef9a9a"
            bmi_icon = "⚠️"
            bmi_tip = "Please consult a healthcare professional."

        st.markdown(f"""
        <div style='background-color:{bmi_color}; color:black; padding:15px; border-radius:10px; margin-top:15px;'>
            <h4 style='margin-bottom:10px;'>🧮 Calculated BMI: <strong>{bmi:.2f}</strong> ({bmi_label})</h4>
            <p style='font-size:16px;'>{bmi_icon} {bmi_tip}</p>
        </div>
        """, unsafe_allow_html=True)

        age = st.number_input("Age", 1, 120, 30)

    with col2:
        st.markdown("### Decision Threshold")
        threshold = st.slider("", 0, 100, 50)

    if st.button("Predict"):
        errors = []
        if user_name.strip() == "":
            errors.append("Please enter your name.")
        if not (40 <= glucose <= 300): errors.append("Glucose must be between 40–300.")
        if not (20 <= blood_pressure <= 200): errors.append("Blood Pressure must be between 20–200.")
        if not (10 <= bmi <= 70): errors.append("BMI must be between 10–70.")
        if not (1 <= age <= 120): errors.append("Age must be between 1–120.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            features = np.array([[glucose, blood_pressure, bmi, age]])
            proba = model.predict_proba(features)[0][1] * 100
            prediction = 1 if proba >= threshold else 0

            if proba < 30:
                risk = "Low"
            elif proba < 70:
                risk = "Medium"
            else:
                risk = "High"

            if prediction == 1:
                st.warning(f"⚠️ {user_name}, you are likely to have diabetes.\nConfidence: {proba:.2f}% (Threshold: {threshold}%)\nRisk Level: {risk}")
            else:
                st.success(f"✅ {user_name}, you are not likely to have diabetes.\nConfidence: {100 - proba:.2f}% (Threshold: {threshold}%)\nRisk Level: {risk}")

            summary = generate_risk_summary(glucose, blood_pressure, bmi, age, risk)
            st.markdown(summary)

            st.session_state.history.append({
                "Name": user_name,
                "Glucose": glucose,
                "Blood Pressure": blood_pressure,
                "BMI": bmi,
                "Age": age,
                "Prediction": "Likely" if prediction == 1 else "Not Likely",
                "Confidence": f"{proba:.2f}%",
                "Threshold": f"{threshold}%",
                "Risk Level": risk
            })

            report = f"""AI Diabetes Risk Report
-----------------------
Name: {user_name}
Glucose: {glucose}
Blood Pressure: {blood_pressure}
BMI: {bmi:.2f}
Age: {age}
Decision Threshold: {threshold}%
Prediction: {"Likely" if prediction == 1 else "Not Likely"}
Confidence: {proba:.2f}%
Risk Level: {risk}

Advice:
{summary.replace('### Overall Risk Level: ' + risk, '').strip()}
"""
            st.download_button("📄 Download Report (.txt)", report, file_name=f"{user_name}_diabetes_report.txt")
            st.session_state.last_prediction = True

# INFO TAB
with tab_info:
    st.header("Understanding the Inputs")
    st.markdown("""
    - **Glucose:** Elevated glucose may indicate diabetes.  
    - **Blood Pressure:** High BP increases diabetes complications.  
    - **BMI:** Obesity is a strong diabetes risk factor.  
    - **Age:** Risk increases after age 45.  
    """)

# HISTORY TAB
with tab_history:
    st.header("Prediction History")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history)
        csv_buf = io.StringIO()
        df_history.to_csv(csv_buf, index=False)
        st.download_button("📥 Download History", csv_buf.getvalue(), "diabetes_history.csv", "text/csv")
    else:
        st.info("No predictions yet.")

# HEALTH TIPS TAB
with tab_tips:
    st.header("💡 Personalised Health Tips")
    if not st.session_state.get("last_prediction", False):
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
