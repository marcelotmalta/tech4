import joblib
import os

path = "artifacts/scaler_generic.joblib"
if os.path.exists(path):
    try:
        data = joblib.load(path)
        print(f"Type of loaded data: {type(data)}")
    except Exception as e:
        print(f"Error loading joblib: {e}")
else:
    print(f"File not found: {path}")
