import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

st.title("🩺 Diabetes Prediction using Naive Bayes")

# Load dataset (manual small dataset)
data = {
    "Glucose": [148, 85, 183, 89, 137, 116, 78, 115],
    "BloodPressure": [72, 66, 64, 66, 40, 74, 50, 0],
    "BMI": [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3],
    "Age": [50, 31, 32, 21, 33, 30, 26, 29],
    "Outcome": [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model
model = GaussianNB()
model.fit(X, y)

st.subheader("Enter Patient Details")

glucose = st.number_input("Glucose Level", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
bmi = st.number_input("BMI", 0.0, 50.0, 25.0)
age = st.number_input("Age", 1, 100, 30)

input_data = pd.DataFrame([[glucose, bp, bmi, age]],
                          columns=["Glucose", "BloodPressure", "BMI", "Age"])

if st.button("Predict Diabetes"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
        