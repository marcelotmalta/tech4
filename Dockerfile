FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Dependências mínimas de sistema
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Instala somente as deps da API
COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir -r /tmp/requirements-api.txt

# Código
COPY . .

ENV MODEL_PATH=models/lstm_generic.keras \
    SCALER_PATH=artifacts/scaler_generic.joblib \
    LOOKBACK=60 \
    DATA_DIR=/app/data

EXPOSE 8000
CMD ["uvicorn", "api.app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
