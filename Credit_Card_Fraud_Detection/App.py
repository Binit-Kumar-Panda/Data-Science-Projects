import streamlit as st
import numpy as np
import pickle

# Load the saved model
with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App
st.title("XGBoost Model Predictor")
st.write("Provide input features to predict the outcome using the pre-trained XGBoost model.")

# Create input fields for each feature
st.subheader("Input Features")
feature_labels = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# Collect input from the user
input_features = []
for label in feature_labels:
    value = st.number_input(label, value=0.0, format="%.6f")
    input_features.append(value)

# Prediction Button
if st.button("Predict"):
    # Reshape input for the model
    input_data = np.array(input_features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Display the result
    st.subheader("Prediction")
    st.write("Predicted Class:", int(prediction[0]))
