import pandas as pd
import requests
import json

# Load a sample from your preprocessed data
df = pd.read_csv('D:/Battery_Failure_Prediction/nasa_battery_data_preprocessed.csv')
features = ['cycle', 'voltage', 'current', 'temperature', 'capacity', 'time', 'internal_resistance']
sample_data = df[features].values[:25]  # First 25 rows

url = "http://localhost:5000/predict"
data = {"data": sample_data.tolist()}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())