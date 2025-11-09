import requests

url = "http://localhost:8000/predict_ticker"
payload = {
    "symbol": "BBAS3.SA",
    "lookback": 60,
    "start_date": "2018-01-01",
    "cache": True
}

r = requests.post(url, json=payload)
print(r.status_code, r.json())
