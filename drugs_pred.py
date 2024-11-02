import streamlit as st
import pickle
import numpy as np
# Load the trained model
model = pickle.load(open("Drug.pkl", "rb"))
# Streamlit interface
st.title("Drug Prediction App")
# Input fields
age = st.number_input("Enter Age", min_value=0)
sex = st.selectbox("Enter Sex", ["Male", "Female"])
bp = st.selectbox("Enter BP", ["Low", "Normal", "High"])
cholesterol = st.selectbox("Enter Cholesterol", ["Normal", "High"])
na_to_k = st.number_input("Enter Na_to_K ratio", min_value=0.0, format="%.3f")
# Encode categorical inputs
sex_encoded = 1 if sex == "Male" else 0
bp_encoded = {"Low": 0, "Normal": 1, "High": 2}[bp]
cholesterol_encoded = 1 if cholesterol == "High" else 0
# Prediction button
if st.button("Predict Drug"):
    # Prepare input for prediction
    features = np.array([[age, sex_encoded, bp_encoded, cholesterol_encoded, na_to_k]])
    prediction = model.predict(features)
    # Mapping the predictions to drug name
    predicted_drug = ""
    if prediction[0] == 0:
        predicted_drug = 'drugA'
    elif prediction[0] == 1:
        predicted_drug = 'drugB'
    elif prediction[0] == 2:
        predicted_drug = 'drugC'
    elif prediction[0] == 3:
        predicted_drug = 'drugX'
    elif prediction[0] == 4:
        predicted_drug = 'drugY'
    # Display the result
    st.write(f"The predicted drug is {predicted_drug}")
