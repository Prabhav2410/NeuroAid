import requests

url = "http://localhost:5000/api/chat"
payload = {"message": "Hello Gemini"}
resp = requests.post(url, json=payload)

print(resp.json())