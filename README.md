# Tech Challenge – Fase 4 (LSTM + API)

Este repositório contém um **notebook** e um **stub de API** para previsão de fechamento de ações usando **LSTM**.
O fluxo está organizado em **fases** e segue os entregáveis solicitados no desafio.

## Estrutura
```
tech_challenge_fase4/
├─ notebook/
│  └─ 01_lstm_pipeline.ipynb
├─ api/
│  └─ app_fastapi.py
├─ data/
│  └─ sample.csv            # (opcional) cole aqui seus dados históricos se não usar yfinance
├─ Dockerfile
├─ requirements.txt
└─ README.md
```

## Como usar

### Ambiente local
1. Crie um virtualenv e instale as dependências:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Abra o notebook:
   ```bash
   jupyter lab  # ou jupyter notebook
   ```
3. Execute as **Fases 1–5** no notebook para treinar e salvar o modelo (`models/` e `artifacts/`).

### API (FastAPI)
1. Execute localmente (após salvar o modelo):
   ```bash
   uvicorn api.app_fastapi:app --host 0.0.0.0 --port 8000 --reload
   ```
2. Documentação interativa:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker
```bash
docker build -t tech-challenge-fase4 .
docker run -p 8000:8000 -v $(pwd)/models:/app/models -v $(pwd)/artifacts:/app/artifacts tech-challenge-fase4
```

## Observações
- Em ambientes **sem internet**, utilize um CSV em `data/sample.csv`. O notebook detecta e usa automaticamente.
- Em ambientes **com internet**, o notebook pode baixar dados via `yfinance` (basta definir `USE_YFINANCE=True`).
- Métricas: MAE, RMSE, MAPE.
- O app de API está pronto para **carregar o modelo salvo** e aceitar janelas de dados para previsão.

**Última atualização:** 2025-11-09


---

## Notas para Windows (erros de instalação / MemoryError)

Se estiver em **Windows nativo**, recomenda-se **Python 3.10** e usar o arquivo `requirements-windows.txt`:

```powershell
# Verifique versão do Python:
py --version

# Crie o venv com Python 3.10
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# Instale em etapas para reduzir uso de memória
pip install --no-cache-dir -r requirements-windows.txt
```

**Dicas para evitar `MemoryError` durante `pip install`:**
1. Feche IDEs e apps pesados (navegador com muitas abas, etc.).  
2. Aumente o arquivo de paginação (memória virtual) do Windows.  
3. Instale em **duas etapas**:
   - Primeiro stack científico + notebook:
     ```powershell
     pip install --no-cache-dir numpy==1.26.4 pandas==2.0.3 scikit-learn==1.3.2 matplotlib==3.7.3 yfinance==0.2.40 jupyter==1.0.0
     ```
   - Depois TensorFlow CPU:
     ```powershell
     pip install --no-cache-dir tensorflow==2.10.1 h5py==3.8.0
     ```
4. Evite instalar o pacote `keras` separado (use `tf.keras`).  
5. Se persistir, considere usar **WSL2 (Ubuntu)** ou **Conda**:
   - **WSL2**: Python 3.10 + `pip install tensorflow==2.13.*` (CPU) costumam funcionar melhor.
   - **Conda**: `conda create -n tc4 python=3.10 tensorflow==2.10.1`.

