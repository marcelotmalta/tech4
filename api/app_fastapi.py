
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

import joblib
from tensorflow.keras.models import load_model

MODEL_PATH = os.getenv("MODEL_PATH", "models/lstm_generic.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "artifacts/scaler_generic.joblib")
LOOKBACK_DEFAULT = int(os.getenv("LOOKBACK", "60"))
DATA_DIR = os.getenv("DATA_DIR", "data")

app = FastAPI(
    title="Tech Challenge Fase 4 - LSTM API",
    version="1.0.0",
    description="API de previsão para séries de preços com modelo LSTM genérico."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    close_prices: List[float] = Field(..., description="Série de fechamentos (antigo -> recente).")
    lookback: Optional[int] = Field(None, description="Se omitido, usa LOOKBACK do servidor.")
    horizon: Optional[int] = Field(1, description="Passos à frente (rollout simples).")

    @validator("close_prices")
    def _validate_series(cls, v):
        if not v or not isinstance(v, list):
            raise ValueError("close_prices deve ser uma lista de números.")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("close_prices deve conter apenas números.")
        return v

class TickerRequest(BaseModel):
    symbol: str = Field(..., description="Ticker, ex.: PETR4.SA (B3) ou AAPL (NASDAQ)")
    lookback: Optional[int] = Field(None, description="Se omitido, usa LOOKBACK do servidor")
    start_date: Optional[str] = Field("2018-01-01", description="AAAA-MM-DD")
    cache: Optional[bool] = Field(True, description="Salvar CSV em data/<symbol>.csv para reuso")

_model = None
_scaler = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}.")
        _model = load_model(MODEL_PATH)
    return _model

def get_scaler():
    global _scaler
    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler não encontrado em {SCALER_PATH}.")
        _scaler = joblib.load(SCALER_PATH)
    return _scaler

def _predict_from_closes(close_prices: List[float], lookback: int, horizon: int = 1) -> List[float]:
    scaler = get_scaler()
    model = get_model()

    if len(close_prices) < lookback:
        raise ValueError(f"São necessários pelo menos {lookback} pontos em close_prices.")

    arr = np.array(close_prices, dtype=float).reshape(-1, 1)
    arr_scaled = scaler.transform(arr)

    window = arr_scaled[-lookback:].reshape(1, lookback, 1)

    horizon = max(1, int(horizon or 1))
    preds = []
    for _ in range(horizon):
        y_scaled = model.predict(window, verbose=0)
        y_inv = scaler.inverse_transform(y_scaled)[0, 0]
        preds.append(float(y_inv))
        window = np.concatenate([window[:, 1:, :], y_scaled.reshape(1, 1, 1)], axis=1)

    return preds

def _csv_cache_path(symbol: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{symbol}.csv")

def _fetch_alpha(symbol: str, start_date: str) -> pd.DataFrame:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY não definida no ambiente.")

    try:
        from alpha_vantage.timeseries import TimeSeries
    except Exception as e:
        raise RuntimeError(f"alpha_vantage não instalado: {e}")

    ts = TimeSeries(key=api_key, output_format="pandas")
    data, meta = ts.get_daily(symbol=symbol, outputsize="full")

    data = (
        data.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
        })
        .reset_index()
        .rename(columns={"date": "Date"})
    )

    if start_date:
        data = data[data["Date"] >= pd.to_datetime(start_date)]

    data = data.sort_values("Date").reset_index(drop=True)
    if data.empty or "Close" not in data.columns:
        raise RuntimeError(f"Alpha Vantage retornou conjunto vazio para {symbol}.")
    return data

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "name": "Tech Challenge Fase 4 - LSTM API",
        "endpoints": ["/health", "/predict", "/predict_ticker", "/docs"],
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "lookback_default": LOOKBACK_DEFAULT
    }

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        lb = int(req.lookback or LOOKBACK_DEFAULT)
        hz = int(req.horizon or 1)
        preds = _predict_from_closes(req.close_prices, lookback=lb, horizon=hz)
        return {"lookback": lb, "horizon": hz, "prediction": preds}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/predict_ticker")
def predict_ticker(req: TickerRequest):
    symbol = req.symbol.strip()
    lb = int(req.lookback or LOOKBACK_DEFAULT)
    start_date = req.start_date or "2018-01-01"
    use_cache = bool(req.cache if req.cache is not None else True)

    df = None
    cache_path = _csv_cache_path(symbol)
    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        except Exception:
            df = None

    if df is None:
        try:
            df = _fetch_alpha(symbol, start_date=start_date)
            if use_cache:
                df.to_csv(cache_path, index=False)
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": f"Falha ao baixar {symbol}: {e}"})

    if df is None or df.empty or "Close" not in df.columns:
        return JSONResponse(status_code=404, content={"error": f"Sem dados para {symbol}."})

    closes = df["Close"].dropna().tolist()
    try:
        preds = _predict_from_closes(closes, lookback=lb, horizon=1)
        return {"symbol": symbol, "lookback": lb, "prediction": preds}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# Eager load para validar artefatos (não bloqueia /health se faltar)
try:
    _ = get_model()
    _ = get_scaler()
except Exception as init_err:
    print(f"[warn] Inicialização parcial: {init_err}")
