# Dockerfile para API FastAPI (execução de inferência)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (opcional: BLAS para numpy/pandas)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos apenas a pasta da API; o modelo será montado por volume
COPY api/ ./api/

# Crie diretórios para montar modelos/artefatos
RUN mkdir -p /app/models /app/artifacts

EXPOSE 8000
CMD ["uvicorn", "api.app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
