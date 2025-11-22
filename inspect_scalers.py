import joblib
import os

path = "artifacts/scalers_dict.joblib"
if os.path.exists(path):
    try:
        data = joblib.load(path)
        print(f"Type of loaded data: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for k, v in data.items():
                print(f"Key: {k}, Type: {type(v)}")
        else:
            print(f"Data: {data}")
    except Exception as e:
        print(f"Error loading joblib: {e}")
else:
    print(f"File not found: {path}")
