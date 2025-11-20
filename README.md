# Tech Challenge – Fase 4 (LSTM + API)

Este repositório contém a solução para a Fase 4 do Tech Challenge, focada na criação de uma API para previsão de preços de ações utilizando um modelo **LSTM Multivariado**.

## 🌐 Acesso Online

A API está deployada e acessível publicamente no Render. Você pode testar os endpoints diretamente pela documentação interativa (Swagger UI):

👉 **[https://tech4.onrender.com/docs](https://tech4.onrender.com/docs)**

## 🧠 O Modelo Final

O modelo desenvolvido é uma **LSTM (Long Short-Term Memory)** que utiliza três variáveis para prever o preço de fechamento (`Close`) do dia seguinte:

1.  **Close**: Preço de fechamento.
2.  **Volume**: Volume de negociações (aplicado logaritmo `np.log1p`).
3.  **RSI (Relative Strength Index)**: Índice de Força Relativa (calculado com janela de 14 dias).

O modelo foi treinado para aceitar sequências de **60 dias** (lookback) e prever 1 passo à frente.

## 🧪 Processo de Treinamento

O modelo foi treinado seguindo um pipeline rigoroso (detalhado no notebook `pipelinegerenrico_rsi.ipynb`):

1.  **Coleta de Dados**:
    -   **Fonte**: Alpha Vantage (com fallback para CSV local).
    -   **Tickers**: Diversos setores e mercados para generalização (`AAPL`, `MSFT`, `AMZN`, `PETR4.SA`, `VALE3.SA`, `ITUB4.SA`).
    -   **Período**: De 2018-01-01 até o presente.

2.  **Engenharia de Features**:
    -   **RSI**: Índice de Força Relativa calculado com janela de 14 dias.
    -   **Log Volume**: Aplicação de `log1p` no volume para reduzir a variância e tratar outliers.

3.  **Pré-processamento**:
    -   **Scaling**: `MinMaxScaler` aplicado individualmente por ticker para cada feature (Close, Vol, RSI).
    -   **Janelamento**: Sequências de 60 dias (Lookback) para prever o próximo dia.
    -   **Split**: Divisão temporal respeitando a ordem cronológica (Treino / Validação / Teste).

4.  **Arquitetura da Rede Neural**:
    -   `LSTM` (96 unidades)
    -   `LayerNormalization`
    -   `Dropout`
    -   `LSTM` (64 unidades)
    -   `Dense` (1 unidade - Saída linear)

5.  **Treinamento**:
    -   **Loss**: Mean Squared Error (MSE).
    -   **Otimizador**: Adam.
    -   **Monitoramento**: Callbacks para Early Stopping e redução de Learning Rate.

## ⚙️ Configuração

Para rodar o projeto, você precisa definir as variáveis de ambiente. Crie um arquivo `.env` na raiz do projeto (baseado no `.env.example`) com o seguinte conteúdo:

```ini
# Chave da API Alpha Vantage (Obrigatória para novos tickers)
ALPHAVANTAGE_API_KEY=sua_chave_aqui

# Configurações do Modelo (Opcionais - valores padrão abaixo)
MODEL_PATH=models/lstm_generic.keras
SCALER_PATH=artifacts/scalers_dict.joblib
LOOKBACK=60
DATA_DIR=data
```

> [!IMPORTANT]
> **Limite da API Alpha Vantage**: A chave gratuita possui um limite de **25 requisições por dia**. Se o limite for atingido, a API retornará erro ao tentar buscar dados de novos tickers. O sistema faz cache dos dados em CSV na pasta `data/` para economizar requisições.

## 📡 Endpoints Disponíveis

A API oferece os seguintes endpoints para consulta:

### 1. Previsão por Ticker (`POST /predict_ticker`)
Realiza a previsão para um ticker específico (ex: PETR4.SA), baixando dados automaticamente da Alpha Vantage.

**Corpo da Requisição (JSON):**
```json
{
  "symbol": "PETR4.SA"
}
```

**Resposta:**
```json
{
  "symbol": "PETR4.SA",
  "last_date_in_history": "2023-10-27 00:00:00",
  "prediction_next_day": 34.50
}
```

### 2. Previsão Manual (`POST /predict`)
Realiza a previsão com base em dados fornecidos manualmente (lista de preços e volumes). Útil para testar cenários ou usar dados de outras fontes.

**Corpo da Requisição (JSON):**
```json
{
  "close_prices": [10.0, 10.2, ...],  // Mínimo 80 pontos
  "volumes": [1000, 1200, ...],       // Mesmo tamanho de close_prices
  "lookback": 60                      // (Opcional) Janela de tempo
}
```

### 3. Health Check (`GET /health`)
Verifica se a API está online e se o modelo foi carregado com sucesso.

**Resposta:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```


## 🚀 Como Rodar

### Localmente (Python)

1.  Crie e ative o ambiente virtual:
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

2.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3.  Execute a API:
    ```bash
    uvicorn api.app_fastapi:app --host 0.0.0.0 --port 8000 --reload
    ```

4.  Acesse a documentação:
    - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### Via Docker

1.  Construa a imagem:
    ```bash
    docker build -t tech-challenge-fase4 .
    ```

2.  Rode o container (passando a chave de API):
    ```bash
    docker run -p 8000:8000 -e ALPHAVANTAGE_API_KEY=sua_chave_aqui tech-challenge-fase4
    ```

## ☁️ Deploy no Render

Este projeto está configurado para deploy fácil no [Render](https://render.com/).

1.  Crie um novo **Web Service** no Render conectado ao seu repositório GitHub.
2.  Selecione o ambiente **Docker**.
3.  Adicione a variável de ambiente `ALPHAVANTAGE_API_KEY` nas configurações do serviço.
4.  (Opcional) Se quiser persistência do cache de dados, adicione um disco persistente montado em `/app/data`.

## 📂 Estrutura do Projeto

```
tech_challenge_fase4/
├─ api/
│  └─ app_fastapi.py    # Código da API (FastAPI)
├─ artifacts/           # Scalers salvos (.joblib)
├─ data/                # Cache de dados (.csv)
├─ models/              # Modelo treinado (.keras)
├─ notebook/            # Notebooks de treino e análise
├─ Dockerfile           # Configuração Docker
├─ requirements.txt     # Dependências
└─ README.md            # Documentação
```
