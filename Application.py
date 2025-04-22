import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the trained model and necessary encoders/scaler
model = load_model('Models/model.h5')

# Load encoders and scaler
with open('Models/onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('Models/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('Models/Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Streamlit app
st.title("Customer Churn Prediction")

# Sidebar for project description
st.sidebar.title("Project Overview")
st.sidebar.write("""
This application predicts the likelihood of customer churn using an Artificial Neural Network model. 
It takes in customer details, such as credit score, geography, gender, age, and more, and predicts 
whether a customer is likely to leave the company based on these factors.

### Instructions:
- Enter customer information in the input fields.
- Click on **Predict** to get the probability of churn and the model's prediction.

### About the Model:
- The model was trained on historical data and uses features like geographic region, account balance, 
and customer activity to make predictions.
- Churn Probability: A value above 0.5 indicates high churn risk.
""")

# User inputs for prediction
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Geography", options=label_encoder_geo.categories_[0])
gender = st.selectbox("Gender", options=label_encoder_gender.classes_)
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Account Balance", min_value=0, value=60000)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", options=[0, 1])
is_active_member = st.selectbox("Is Active Member", options=[0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0, value=50000)

# Create input dictionary
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

# Process the data for model prediction
# One-hot encode 'Geography'
geo_encoded = label_encoder_geo.transform([[input_data['Geography']]]).toarray()

# Manually specify column names for the one-hot encoded 'Geography'
geo_encoded_df = pd.DataFrame(geo_encoded, columns=[f"Geography_{cat}" for cat in label_encoder_geo.categories_[0]])

# Convert input_data to a DataFrame and encode gender
input_df = pd.DataFrame([input_data])
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

# Concatenate one-hot encoded columns with input_df
input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Predict churn probability
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Display the prediction
if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

# Display the churn probability
st.write(f"Churn Probability: {prediction_proba:.2f}")
