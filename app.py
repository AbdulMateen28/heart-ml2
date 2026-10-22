import streamlit as st
import joblib
import pandas as pd

model = joblib.load("heart_model.pkl")

st.title("Heart Disease Prediction System")

age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)

if st.button("Predict"):
    sex_val = 1 if sex == "Male" else 0
    data = pd.DataFrame([[age, sex_val, chol, thalach]],
                        columns=["age", "sex", "chol", "thalach"])
    result = model.predict(data)[0]
    if result == 1:
        st.error("Heart disease is likely.")
    else:
        st.success("Heart disease is not likely.")