import streamlit as st
import joblib
import numpy as np

model = joblib.load('loan_eligibility_model.pkl')

st.title("Loan Eligibility Predictor")

# Input fields (match the order and encoding used in training!)
Gender = st.selectbox("Gender", ['Male', 'Female'])  # 0=Female, 1=Male
Married = st.selectbox("Married", ['Yes', 'No'])     # 0=No, 1=Yes
Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])  # encoded
Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
ApplicantIncome = st.number_input("Applicant Income")
CoapplicantIncome = st.number_input("Coapplicant Income")
LoanAmount = st.number_input("Loan Amount")
Loan_Amount_Term = st.number_input("Loan Term")
Credit_History = st.selectbox("Credit History", [1.0, 0.0])  # 1 = Good, 0 = Bad
Property_Area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

# Encoding input as per training
gender_map = {'Female': 0, 'Male': 1}
married_map = {'No': 0, 'Yes': 1}
dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
education_map = {'Graduate': 0, 'Not Graduate': 1}
self_employed_map = {'No': 0, 'Yes': 1}
property_area_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}

input_data = [
    gender_map[Gender],
    married_map[Married],
    dependents_map[Dependents],
    education_map[Education],
    self_employed_map[Self_Employed],
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    property_area_map[Property_Area]
]

if st.button("Predict"):
    prediction = model.predict([input_data])
    result = "Eligible ✅" if prediction[0] == 1 else "Not Eligible ❌"
    st.success(result)

