import streamlit as st
import requests
import pandas as pd
import json

st.title("Battery Failure Prediction Dashboard")

# Load data and send request to Flask API
df = pd.read_csv('D:/Battery_Failure_Prediction/nasa_battery_data_preprocessed.csv')
features = ['cycle', 'voltage', 'current', 'temperature', 'capacity', 'time', 'internal_resistance']
sample_data = df[features].values[:25]

url = "http://localhost:5000/predict"
data = {"data": sample_data.tolist()}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)
if response.status_code == 200:
    predictions = response.json()['predictions']
    st.write("### Predictions")
    st.dataframe(pd.DataFrame(predictions))
else:
    st.error(f"Failed to get predictions: {response.text}")