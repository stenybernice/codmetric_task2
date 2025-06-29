
import streamlit as st
import numpy as np
import pandas as pd
import joblib


# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("🏡 California Housing Price Prediction")
st.markdown("Fill in the features below to predict the house price.")

# Input fields for each feature
MedInc = st.number_input("Median Income (10k USD)", min_value=0.0, value=5.0)
HouseAge = st.number_input("House Age", min_value=1.0, value=20.0)
AveRooms = st.number_input("Average Rooms", min_value=0.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)
Population = st.number_input("Population", min_value=1.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy", min_value=1.0, value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

# Create input array
user_input = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                        Population, AveOccup, Latitude, Longitude]])

# Scale input
user_input_scaled = scaler.transform(user_input)

# Predict
if st.button("Predict Price"):
    user_input_scaled = scaler.transform(user_input)

    raw_pred = model.predict(user_input_scaled)[0]
    prediction = max(raw_pred, 0)

    if raw_pred < 0:
        st.warning("Invalid input.This combo predicts a negative price")
    else:
        st.info("Valid input! Model predicts a proper house value.")

    st.success(f"Estimated Median House Value: **${prediction * 100000:,.2f}**")


