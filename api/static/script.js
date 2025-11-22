const tickerInput = document.getElementById('tickerInput');
const predictBtn = document.getElementById('predictBtn');
const loadingDiv = document.getElementById('loading');
const resultDiv = document.getElementById('result');
const errorDiv = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');

let chartInstance = null;

predictBtn.addEventListener('click', async () => {
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) return;

    // Validation for Brazilian stocks (e.g. PETR4 -> PETR4.SA)
    // Regex: 4 letters followed by 1 or 2 digits, and NOT ending in .SA
    const b3Pattern = /^[A-Z]{4}\d{1,2}$/;
    if (b3Pattern.test(ticker)) {
        showError("Para ações brasileiras, por favor adicione '.SA' ao final (ex: " + ticker + ".SA).");
        return;
    }

    // UI State
    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
    predictBtn.disabled = true;

    try {
        const response = await fetch('/predict_ticker', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol: ticker })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to fetch prediction');
        }

        renderResult(data);
    } catch (err) {
        showError(err.message);
    } finally {
        loadingDiv.classList.add('hidden');
        predictBtn.disabled = false;
    }
});

function renderResult(data) {
    // Update Cards
    const lastPrice = data.history_prices[data.history_prices.length - 1];
    document.getElementById('lastPrice').textContent = formatCurrency(lastPrice);
    document.getElementById('lastDate').textContent = data.last_date_in_history;

    document.getElementById('predictedPrice').textContent = formatCurrency(data.prediction_next_day);
    document.getElementById('errorMargin').textContent = `± ${formatCurrency(data.error_margin)} (95% Conf.)`;

    // Render Chart
    renderChart(data);

    resultDiv.classList.remove('hidden');
}

function renderChart(data) {
    const ctx = document.getElementById('stockChart').getContext('2d');

    if (chartInstance) {
        chartInstance.destroy();
    }

    // Prepare Data
    const labels = [...data.history_dates, 'Forecast'];
    const prices = [...data.history_prices, null]; // History only
    const predictionPoint = new Array(data.history_prices.length).fill(null);
    predictionPoint.push(data.prediction_next_day); // Prediction only

    // Confidence Interval Data
    // Start from the last historical point to create a continuous "cone"
    const lastHistoryPrice = data.history_prices[data.history_prices.length - 1];

    const upperBoundData = new Array(data.history_prices.length).fill(null);
    upperBoundData[upperBoundData.length - 1] = lastHistoryPrice; // Start point
    upperBoundData.push(data.prediction_next_day + data.error_margin); // End point

    const lowerBoundData = new Array(data.history_prices.length).fill(null);
    lowerBoundData[lowerBoundData.length - 1] = lastHistoryPrice; // Start point
    lowerBoundData.push(data.prediction_next_day - data.error_margin); // End point

    // Gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(56, 189, 248, 0.5)');
    gradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical Price',
                    data: prices,
                    borderColor: '#38bdf8',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Prediction',
                    data: predictionPoint,
                    borderColor: '#818cf8',
                    backgroundColor: '#818cf8',
                    borderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    borderDash: [5, 5]
                },
                {
                    label: 'Confidence Upper',
                    data: upperBoundData,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(129, 140, 248, 0.2)',
                    fill: '+1', // Fill to next dataset (Lower)
                    pointRadius: 0,
                    tension: 0
                },
                {
                    label: 'Confidence Lower',
                    data: lowerBoundData,
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    fill: false,
                    pointRadius: 0,
                    tension: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#94a3b8' }
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    borderColor: '#334155',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 10 }
                },
                y: {
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

function formatCurrency(value) {
    return new Intl.NumberFormat('pt-BR', { style: 'currency', currency: 'BRL' }).format(value);
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorDiv.classList.remove('hidden');
}
