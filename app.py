import streamlit as st
import pandas as pd
import joblib

# 🧠 Load your trained model (make sure the .pkl file is in same folder)
model = joblib.load("salary_model.pkl")

# 🌸 Website Title
st.title("💼 Salary Prediction App")
st.write("Predict an employee's salary based on age, gender, education, job title, and experience.")

# 🧾 Input fields
age = st.number_input("Enter Age:", min_value=18, max_value=70, value=25)
gender = st.selectbox("Select Gender:", ["Male", "Female"])
education = st.selectbox("Select Education Level:", ["Bachelor's", "Master's", "PhD"])
job = st.text_input("Enter Job Title:", "Software Engineer")
experience = st.number_input("Years of Experience:", min_value=0, max_value=50, value=2)

# 🧮 Predict Button
if st.button("Predict Salary"):
    # Convert categorical inputs to numeric (same as your training step)
    gender_code = 1 if gender.lower() == "male" else 0
    edu_map = {"bachelor's": 0, "master's": 1, "phd": 2}
    education_code = edu_map.get(education.lower(), 0)

    # Create DataFrame for model input
    X = pd.DataFrame([[age, gender_code, education_code, 0, experience]],
                     columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Predict using model
    salary_pred = model.predict(X)[0]

    # 🎯 Display result
    st.success(f"💰 Predicted Salary: ₹{int(round(salary_pred)):,}")

# 📜 Footer
st.markdown("---")
st.caption("Project by Sudipta Das | Model: Linear Regression | Dataset: Kaggle Salary Data")
