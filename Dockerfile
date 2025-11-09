# Base leve com Python 3.10
FROM python:3.10-slim

# Configs de runtime Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Diretório de trabalho
WORKDIR /app

# Dependências de sistema essenciais (mínimas)
# (adicione outras se necessário, ex.: 'tzdata', 'curl' para HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia ambos os requirements e escolhe em build qual usar
# Se você NÃO tiver o requirements-cpu.txt, o build usa o requirements.txt
COPY requirements.txt requirements-cpu.txt* /tmp/

# Escolha do arquivo de dependências via ARG (padrão: requirements-cpu.txt)
ARG REQS=requirements-cpu.txt
RUN if [ -f "/tmp/${REQS}" ]; then \
        echo "Instalando deps de ${REQS}" && pip install -r "/tmp/${REQS}"; \
    else \
        echo "Instalando deps de requirements.txt" && pip install -r /tmp/requirements.txt; \
    fi

# Copia o restante do projeto
COPY . .

# Variáveis de ambiente padrão (ajuste se trocar nomes/locais)
ENV MODEL_PATH=models/lstm_generic.keras \
    SCALER_PATH=artifacts/scaler_generic.joblib \
    LOOKBACK=60 \
    DATA_DIR=data

# Porta da API
EXPOSE 8000

# (Opcional) Healthcheck — requer 'curl'
# RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
# HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
#   CMD curl -f http://localhost:8000/health || exit 1

# Comando padrão para subir a API
CMD ["uvicorn", "api.app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
