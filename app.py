# app.py

import streamlit as st
import pandas as pd
import joblib
import os

# Load model and feature list
model = joblib.load('models/bike_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

st.set_page_config(page_title="Bike Rental Predictor", layout="centered")

st.title("🚴 Bike Rental Count Predictor")
st.markdown("Predict bike rentals for specific hour and conditions by adjusting the inputs.")

# Sidebar Inputs
st.sidebar.header("Select Input Parameters")

# Input controls
season = st.sidebar.selectbox("Season", ['Spring', 'Summer', 'Fall', 'Winter'])
season_map = {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4}

yr = st.sidebar.selectbox("Year", ['2011', '2012'])
yr_map = {'2011': 0, '2012': 1}

mnth = st.sidebar.slider("Month", 1, 12, 6)
hr = st.sidebar.slider("Hour (0–23)", 0, 23, 9)
holiday = st.sidebar.selectbox("Holiday", ['No', 'Yes'])
holiday_map = {'No': 0, 'Yes': 1}

weekday = st.sidebar.slider("Weekday (0=Sun)", 0, 6, 3)
workingday = st.sidebar.selectbox("Working Day", ['No', 'Yes'])
workingday_map = {'No': 0, 'Yes': 1}

weathersit = st.sidebar.selectbox("Weather",
    ['Clear', 'Mist', 'Light Snow/Rain', 'Heavy Rain'])
weathersit_map = {
    'Clear': 1,
    'Mist': 2,
    'Light Snow/Rain': 3,
    'Heavy Rain': 4
}

temp = st.sidebar.slider("Temperature (normalized)", 0.0, 1.0, 0.5)
atemp = st.sidebar.slider("Feels Like Temp (normalized)", 0.0, 1.0, 0.5)
hum = st.sidebar.slider("Humidity (normalized)", 0.0, 1.0, 0.5)
windspeed = st.sidebar.slider("Windspeed (normalized)", 0.0, 1.0, 0.2)

# Convert inputs to a DataFrame
input_data = pd.DataFrame([{
    'season': season_map[season],
    'yr': yr_map[yr],
    'mnth': mnth,
    'hr': hr,
    'holiday': holiday_map[holiday],
    'weekday': weekday,
    'workingday': workingday_map[workingday],
    'weathersit': weathersit_map[weathersit],
    'temp': temp,
    'atemp': atemp,
    'hum': hum,
    'windspeed': windspeed
}])

# One-hot encoding like in training
input_data_encoded = pd.get_dummies(input_data)

# Add any missing columns from training
for col in feature_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns exactly like in training
input_data_encoded = input_data_encoded[feature_columns]

# Predict
if st.button("Predict Bike Rentals"):
    prediction = model.predict(input_data_encoded)[0]
    st.success(f"📈 Predicted Bike Rentals: **{int(prediction)}**")

    # Debug info (Optional)
    st.subheader("Input Sent to Model")
    st.dataframe(input_data_encoded)
