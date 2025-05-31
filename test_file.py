import requests
import json

url = "http://localhost:5000/predict"
data = {
    "data": [[i, 3.8 - i*0.01, -0.5, 25 + i*0.1, 1.8 - i*0.01, 100 + i, 0.1 + i*0.01] for i in range(1, 21)]
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.status_code)
print(response.json())