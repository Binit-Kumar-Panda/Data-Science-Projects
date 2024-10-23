import streamlit as st
import pandas as pd
import pickle

# Load your trained model
model = pickle.load(open(r"C:\Users\binit\Downloads\Ridge.pkl", 'rb'))

# Title and Description
st.title("House Price Prediction App")
st.write("""
### Enter the house details to predict the price:
""")

# Input Features
total_sqft = st.number_input("Total Square Feet", min_value=500, max_value=10000, step=100, value=1000)
bhk = st.slider("Number of BHK", min_value=1, max_value=10, step=1, value=3)
bath = st.slider("Number of Bathrooms", min_value=1, max_value=5, step=1, value=2)
location = st.selectbox("Location", [
    '1st Block Jayanagar', '1st Phase JP Nagar', '2nd Phase Judicial Layout', 
    '2nd Stage Nagarbhavi', '5th Block Hbr Layout', '5th Phase JP Nagar',
    # Add all your locations here...
    'other'  # make sure to include all possible locations
])  # Categorical feature

# Make predictions when the button is clicked
if st.button("Predict Price"):
    # Collect input into a DataFrame
    input_data = pd.DataFrame({
        'total_sqft': [total_sqft],
        'bhk': [bhk],
        'bath': [bath],
        'location': [location]
    })

    # Predict the price using the DataFrame
    predicted_price = model.predict(input_data)[0]
    
    # Show the predicted price
    st.success(f"The predicted house price is ${predicted_price:,.2f}")
