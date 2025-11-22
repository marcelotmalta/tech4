import sys
import os
import pandas as pd
import numpy as np

# Add api directory to path
sys.path.append(os.path.join(os.getcwd(), 'api'))

from app_fastapi import prepare_features, get_scalers_dict

# Mock data
df = pd.DataFrame({
    'Close': np.random.rand(100) * 100,
    'Volume': np.random.randint(1000, 10000, 100),
    'RSI': np.random.rand(100) * 100
})

print("Testing with ticker 'VALE3' (not in dict)")
scalers = get_scalers_dict()
print(f"Scalers dict type: {type(scalers)}")
# print(f"Scalers dict content: {scalers}")

try:
    prepare_features(df, 'VALE3')
    print("Success 'VALE3'")
except Exception as e:
    print(f"Error 'VALE3': {e}")

print("\nTesting with ticker 'VALE3.SA' (in dict)")
try:
    prepare_features(df, 'VALE3.SA')
    print("Success 'VALE3.SA'")
except Exception as e:
    print(f"Error 'VALE3.SA': {e}")
