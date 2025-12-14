import os
from dotenv import load_dotenv

load_dotenv()

from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_fastapi_instrumentator import Instrumentator
from evidently import Report
from evidently.presets import DataDriftPreset

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

from fastapi.staticfiles import StaticFiles

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/app", StaticFiles(directory="api/static", html=True), name="static")

# --- Monitoring ---
Instrumentator().instrument(app).expose(app)

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
            loaded = joblib.load(SCALERS_DICT_PATH)
            if isinstance(loaded, dict):
                _scalers_dict = loaded
            else:
                print(f"[warn] Arquivo {SCALERS_DICT_PATH} não contém um dicionário (tipo encontrado: {type(loaded)}). Iniciando vazio.")
                _scalers_dict = {}
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
    # 1. Busca Dados (sempre tenta Alpha Vantage primeiro para dados recentes)
    try:
        df = _get_data_for_ticker(symbol, use_cache=True)
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
        
        # Calculate error margin (Volatility-based: 1.96 * std(returns) * price)
        # Using last 30 days for volatility
        returns = df['Close'].pct_change().dropna()
        volatility = returns.tail(30).std() if len(returns) > 1 else 0.02 # Fallback 2%
        error_margin = float(pred_real * volatility * 1.96)

        return {
            "symbol": symbol,
            "last_date_in_history": str(last_date),
            "prediction_next_day": pred_real,
            "error_margin": error_margin,
            "history_dates": df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "history_prices": df['Close'].tolist()
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erro na inferência: {e}"})

@app.get("/monitoring/drift", response_class=HTMLResponse)
def monitoring_drift(symbol: str = "PETR4.SA"):
    """
    Gera um relatório HTML de Data Drift usando Evidently.
    Usa dados históricos recentes:
    - Reference: dados mais antigos do fetch (ex: iniciais)
    - Current: dados mais recentes (ex: últimos 30 dias)
    """
    try:
        # 1. Busca dados (cache ou API)
        df = _get_data_for_ticker(symbol, use_cache=True)
        
        if len(df) < 60:
             return f"<h1>Erro</h1><p>Dados insuficientes ({len(df)} linhas) para gerar relatório de drift.</p>"
        
        # 2. Split Reference vs Current
        # Vamos assumir os últimos 30 dias como 'Current' e o resto como 'Reference'
        # Se 30 dias for muito pouco em relação ao total, ajustamos.
        
        split_index = len(df) - 30
        if split_index < 30: # Garante pelo menos 30 pontos de referencia
             split_index = len(df) // 2
             
        reference = df.iloc[:split_index]
        current = df.iloc[split_index:]
        
        # 3. Gera Relatório
        report = Report(metrics=[
            DataDriftPreset(), 
        ])
        
        report.run(reference_data=reference, current_data=current)
        
        return report.get_html()
        
    except Exception as e:
        return f"<h1>Erro ao gerar relatório</h1><p>{str(e)}</p>"

@app.get("/monitoring/dashboard", response_class=HTMLResponse)
def monitoring_dashboard():
    with open("api/static/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

# --- Helpers de Dados ---
def _get_data_for_ticker(symbol: str, use_cache: bool) -> pd.DataFrame:
    cache_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    
    # Prioriza buscar dados recentes do Alpha Vantage (sempre tenta primeiro para garantir atualização)
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if api_key:
        try:
            from alpha_vantage.timeseries import TimeSeries
            ts = TimeSeries(key=api_key, output_format="pandas")
            data, _ = ts.get_daily(symbol=symbol, outputsize="compact")  # Últimos ~100 dias (dados recentes)
            
            data = data.rename(columns={
                "1. open": "Open", "2. high": "High", "3. low": "Low",
                "4. close": "Close", "5. volume": "Volume"
            }).reset_index().rename(columns={"date": "Date"})
            
            # Não filtra por start_date, pois compact já traz recentes
            data = data.sort_values("Date").reset_index(drop=True)
            
            # Salva no cache para próximas chamadas
            if use_cache:
                os.makedirs(DATA_DIR, exist_ok=True)
                data.to_csv(cache_path, index=False)
                print(f"[alpha] Dados atualizados para {symbol} (salvo em cache)")
            
            return data
        except Exception as e:
            print(f"[alpha] Falhou para {symbol}: {e}. Tentando cache...")
    
    # Fallback: usa cache se disponível
    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
            print(f"[cache] Usando dados locais para {symbol}")
            return df
        except Exception as e:
            raise RuntimeError(f"Cache corrompido para {symbol}: {e}")
    
    raise RuntimeError(f"Não foi possível obter dados para {symbol} (Alpha Vantage indisponível e sem cache).")

# Init Check
try:
    get_model()
    get_scalers_dict()
except Exception as e:
    print(f"[warn] Inicialização lazy: {e}")
