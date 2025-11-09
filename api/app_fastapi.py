from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import os

# Import adiando para evitar custo de import quando não necessário
from tensorflow.keras.models import load_model

app = FastAPI(title="Tech Challenge Fase 4 – LSTM API", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "models/lstm_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "artifacts/scaler.joblib")
LOOKBACK = int(os.getenv("LOOKBACK", "60"))  # janela usada no treino

class PredictRequest(BaseModel):
    # Série de preços (Close) em ordem temporal (antigo -> recente)
    close_prices: List[float]
    # opcionalmente informar lookback usado no treino
    lookback: Optional[int] = None
    horizon: int = 1  # passos à frente (1 = próxima barra)

class PredictResponse(BaseModel):
    prediction: List[float]

def _load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler não encontrado em {SCALER_PATH}")
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    lookback = req.lookback or LOOKBACK
    series = np.array(req.close_prices, dtype=float)

    if len(series) < lookback:
        raise HTTPException(status_code=400, detail=f"São necessários pelo menos {lookback} pontos")

    # Carrega modelo e scaler
    try:
        model, scaler = _load_artifacts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Normaliza e cria a janela mais recente
    series_scaled = scaler.transform(series.reshape(-1, 1)).flatten()
    x = series_scaled[-lookback:].reshape(1, lookback, 1)

    # Previsão (um passo; horizon>1 faz roll)
    preds = []
    last_window = x.copy()
    for _ in range(max(1, req.horizon)):
        yhat_scaled = model.predict(last_window, verbose=0)
        preds.append(yhat_scaled.ravel()[0])
        # atualiza janela para multi-step
        last_window = np.concatenate([last_window[:, 1:, :], yhat_scaled.reshape(1, 1, 1)], axis=1)

    # Desfaz escala
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten().tolist()
    return PredictResponse(prediction=preds_inv)
