import os
from dotenv import load_dotenv

load_dotenv()

from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- Configurações ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/lstm_generic.keras")
# Agora carregamos um dicionário de scalers, não um único scaler
SCALERS_DICT_PATH = os.getenv("SCALER_PATH", "artifacts/scalers_dict.joblib")
LOOKBACK_DEFAULT = int(os.getenv("LOOKBACK", "60"))
DATA_DIR = os.getenv("DATA_DIR", "data")

app = FastAPI(
    title="Tech Challenge Fase 4 - LSTM API (Multivariate)",
    version="2.0.0",
    description="API de previsão para séries de preços com modelo LSTM genérico (Close, Vol, RSI)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic ---
class PredictRequest(BaseModel):
    close_prices: List[float] = Field(..., description="Série de fechamentos (antigo -> recente).")
    volumes: List[float] = Field(..., description="Série de volumes correspondente.")
    lookback: Optional[int] = Field(None, description="Se omitido, usa LOOKBACK do servidor.")
    horizon: Optional[int] = Field(1, description="Passos à frente (rollout simples).")

    @validator("close_prices", "volumes")
    def _validate_list(cls, v):
        if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Deve ser uma lista de números.")
        return v

class TickerRequest(BaseModel):
    symbol: str = Field(..., description="Ticker, ex.: PETR4.SA (B3) ou AAPL (NASDAQ)")

# --- Globais ---
_model = None
_scalers_dict = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}.")
        _model = load_model(MODEL_PATH)
    return _model

def get_scalers_dict():
    global _scalers_dict
    if _scalers_dict is None:
        if os.path.exists(SCALERS_DICT_PATH):
            _scalers_dict = joblib.load(SCALERS_DICT_PATH)
        else:
            print(f"[warn] Dicionário de scalers não encontrado em {SCALERS_DICT_PATH}. Iniciando vazio.")
            _scalers_dict = {}
    return _scalers_dict

# --- Lógica de Features ---
def calculate_rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_features(df: pd.DataFrame, ticker: str):
    """
    Gera features (RSI, LogVol) e aplica scaling.
    Se o ticker já tiver scalers salvos, usa eles.
    Se não, cria novos (fit) na hora.
    Retorna: array (N, 3), scaler_close (para inversão)
    """
    scalers_dict = get_scalers_dict()
    
    # 1. Engenharia de Features
    df = df.copy()
    if 'RSI' not in df.columns:
        df['RSI'] = calculate_rsi(df['Close'], period=14).fillna(50)
    
    # Log Volume
    vol_log = np.log1p(df[['Volume']])
    
    # 2. Scaling
    if ticker in scalers_dict:
        # Usa scalers existentes (treino)
        sc_close = scalers_dict[ticker]['close']
        sc_vol   = scalers_dict[ticker]['vol']
        sc_rsi   = scalers_dict[ticker]['rsi']
        
        close_scaled = sc_close.transform(df[['Close']])
        vol_scaled   = sc_vol.transform(vol_log)
        rsi_scaled   = sc_rsi.transform(df[['RSI']])
    else:
        # Ticker novo: cria scalers on-the-fly
        # print(f"[info] Criando scalers on-the-fly para {ticker}")
        sc_close = MinMaxScaler().fit(df[['Close']])
        sc_vol   = MinMaxScaler().fit(vol_log)
        sc_rsi   = MinMaxScaler().fit(df[['RSI']])
        
        close_scaled = sc_close.transform(df[['Close']])
        vol_scaled   = sc_vol.transform(vol_log)
        rsi_scaled   = sc_rsi.transform(df[['RSI']])
        
    # Monta array final (N, 3) -> [Close, Vol, RSI]
    X_scaled = np.hstack([close_scaled, vol_scaled, rsi_scaled])
    
    return X_scaled, sc_close

def _predict_next_step(model, window_3d):
    # window_3d shape: (1, lookback, 3)
    y_scaled = model.predict(window_3d, verbose=0) # (1, 1)
    return y_scaled[0, 0]

# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.get("/")
def root():
    return {
        "name": "Tech Challenge Fase 4 - LSTM API (Multivariate)",
        "endpoints": ["/health", "/predict", "/predict_ticker"],
        "features": ["Close", "Volume", "RSI"]
    }

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint manual: recebe listas de Close e Volume.
    Assume ticker desconhecido (fit on-the-fly).
    """
    try:
        if len(req.close_prices) != len(req.volumes):
            return JSONResponse(status_code=400, content={"error": "close_prices e volumes devem ter o mesmo tamanho."})
        
        lb = int(req.lookback or LOOKBACK_DEFAULT)
        if len(req.close_prices) < lb + 20: # Precisa de margem para RSI
             return JSONResponse(status_code=400, content={"error": f"Série muito curta. Forneça pelo menos {lb+20} pontos para cálculo de RSI."})

        # Monta DataFrame temporário
        df = pd.DataFrame({
            "Close": req.close_prices,
            "Volume": req.volumes
        })
        
        # Prepara dados (fit on-the-fly)
        X_scaled, scaler_close = prepare_features(df, ticker="MANUAL_REQUEST")
        
        # Pega a última janela
        last_window = X_scaled[-lb:].reshape(1, lb, 3)
        
        # Previsão (apenas 1 passo por enquanto para simplificar a lógica multivariada)
        model = get_model()
        pred_scaled = _predict_next_step(model, last_window)
        pred_real = float(scaler_close.inverse_transform([[pred_scaled]])[0, 0])
        
        return {"lookback": lb, "prediction": [pred_real]}
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/predict_ticker")
def predict_ticker(req: TickerRequest):
    symbol = req.symbol.strip()
    lb = LOOKBACK_DEFAULT
    start_date = "2018-01-01"
    use_cache = True
    # 1. Busca Dados
    try:
        df = _get_data_for_ticker(symbol, start_date, use_cache)
    except Exception as e:
         return JSONResponse(status_code=502, content={"error": f"Erro ao obter dados: {e}"})
         
    if len(df) < lb + 20:
        return JSONResponse(status_code=400, content={"error": f"Histórico insuficiente ({len(df)} linhas) para lookback {lb}."})

    # 2. Prepara Features (usa scalers salvos se existirem)
    try:
        X_scaled, scaler_close = prepare_features(df, ticker=symbol)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erro no processamento de features: {e}"})

    # 3. Previsão
    try:
        last_window = X_scaled[-lb:].reshape(1, lb, 3)
        model = get_model()
        pred_scaled = _predict_next_step(model, last_window)
        pred_real = float(scaler_close.inverse_transform([[pred_scaled]])[0, 0])
        
        last_date = df['Date'].iloc[-1] if 'Date' in df.columns else "N/A"
        
        return {
            "symbol": symbol,
            "last_date_in_history": str(last_date),
            "prediction_next_day": pred_real
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erro na inferência: {e}"})

# --- Helpers de Dados ---
def _get_data_for_ticker(symbol: str, start_date: str, use_cache: bool) -> pd.DataFrame:
    # Tenta cache CSV
    cache_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
            # print(f"[cache] Usando CSV local para {symbol}")
            return df
        except:
            pass
            
    # Tenta Alpha Vantage
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY não definida.")
        
    from alpha_vantage.timeseries import TimeSeries
    ts = TimeSeries(key=api_key, output_format="pandas")
    data, _ = ts.get_daily(symbol=symbol, outputsize="full")
    
    data = data.rename(columns={
        "1. open": "Open", "2. high": "High", "3. low": "Low",
        "4. close": "Close", "5. volume": "Volume"
    }).reset_index().rename(columns={"date": "Date"})
    
    data = data[data["Date"] >= pd.to_datetime(start_date)]
    data = data.sort_values("Date").reset_index(drop=True)
    
    if use_cache:
        os.makedirs(DATA_DIR, exist_ok=True)
        data.to_csv(cache_path, index=False)
        
    return data

# Init Check
try:
    get_model()
    get_scalers_dict()
except Exception as e:
    print(f"[warn] Inicialização lazy: {e}")
