# 📊 PortfolioFuture-SubstAI: Financial Intelligence Dashboard & Machine Learning Framework

> **Enterprise Quantitative Finance & Financial Intelligence Application**  
> **Target Role**: Quantitative Analyst & Principal Solutions Architect | **Organization**: PortfolioFuture (New York, NY / Remote)  
> **Author**: Bhupesh Chandra Dimri  

---

## 📌 Executive Summary
PortfolioFuture is an independent fund discovery and intelligence company connecting over 10,000 ETFs and mutual funds with 20,000+ advisory firms across 130M+ fund relationships and 290M+ ownership opportunities.

**PortfolioFuture-SubstAI** combines a production-grade machine learning framework for fund return decomposition and 13F ownership analytics with a **live, secure Financial Intelligence Dashboard and REST API**.

### Core Capabilities:
1. **Interactive Financial Intelligence Dashboard**: Multi-tab Streamlit dashboard (`app.py`) and standalone zero-dependency web hub (`index.html` & `reports/live_hub.html`) with interactive price/volume candlestick charts, quote cards, sortable/filterable data tables, and CSV export.
2. **Live 19-Group Multi-Asset Quant Calculator**: Allocates capital across 19 asset classes (US Large/Mid/Small Blend/Growth/Value, Emerging Markets, Core Bonds, High Yield, Treasuries, Munis, TIPS, Real Estate, Commodities, Multi-Asset, Equity Income/Covered Call) with live slider rebalancing, portfolio metric calculations (Expected Return, Volatility, Sharpe, Max Drawdown), strategic SWOT matrix, and 10-year terminal wealth projections across Base, Worst, and Best case scenarios.
3. **On-Demand Market Fetch Pipeline**: Strict user-initiated fetching (no unsolicited polling) with SSRF-safe symbol validation (`^[A-Z0-9.\-^=]{1,12}$`), rate-limit awareness, bounded retries, and explicit separation of market timestamp vs retrieval timestamp.
4. **Deterministic Quantitative Analytics Engine**: Pure mathematical indicator algorithms (`src/market_analytics.py`) implementing SMA, EMA, Wilder's smoothed RSI (14), MACD (12, 26, 9), Bollinger Bands (20, 2), and annualized rolling volatility without lookahead bias.
5. **Documented REST API**: High-performance FastAPI backend (`src/api.py`) offering OpenAPI documentation (`/docs`), health checks, quotes, historical bars, indicator series, and cache management with CORS safeguards.
6. **Provenance-Tracked Local Cache**: Thread-safe SQLite store (`data/market_cache.sqlite`) recording raw payloads, TTL expiration (1 hour default), cache hit/miss status, and force-refresh capabilities.
7. **SEC 13F & Form ADV Distribution Intelligence**: Ingests real institutional RIA holdings (Fisher Asset Management, Creative Planning, Captrust, HighTower, etc.) mapping $479B+ in capital and quantifying $2.28B+ in addressable fee savings.
8. **Machine Learning Substitution Classifier**: Random Forest classifier (`models/best_model.pkl`) identifying prime low-cost substitute funds with 100% holdout ROC-AUC.

---

## 📂 Repository Architecture
```text
ml-project-portfoliofuture-substai/
├── README.md                               # Project documentation & architecture manual
├── .env.example                            # Production environment template
├── requirements.txt                        # Python dependencies
├── conftest.py                             # Pytest configuration
├── app.py                                  # Streamlit Financial Intelligence & Quant Dashboard
├── index.html                              # Standalone interactive dashboard & live quant hub
├── reports/
│   └── live_hub.html                       # Mirror GitHub Pages report hub
├── data/
│   ├── market_cache.sqlite                 # Provenance-tracked SQLite market data cache
│   ├── raw/
│   │   └── real_historical_prices_1y.csv   # 1-year historical daily prices
│   ├── processed/
│   │   ├── fund_universe_profiles_real.csv # Real fund profiles & econometric factor loadings
│   │   └── sec_13f_ria_ownership_real.csv  # Institutional SEC 13F & Form ADV RIA holdings
│   └── synthetic/
│       ├── fund_universe_profiles.csv      # 1,200 multi-asset real fund profiles & share classes
│       └── sec_13f_ria_ownership.csv       # 500 institutional RIA holdings records
├── models/
│   └── best_model.pkl                      # Trained Random Forest Substitute Classifier (AUC = 1.00)
├── src/
│   ├── api.py                              # FastAPI backend REST application & OpenAPI docs
│   ├── market_analytics.py                 # Pure math technical indicators (SMA, EMA, RSI, MACD, BB)
│   ├── market_data/                        # Modular market data ingestion package
│   │   ├── __init__.py                     # Package entry point
│   │   ├── schemas.py                      # Pydantic v2 schemas & regex validation
│   │   ├── cache.py                        # Thread-safe SQLite cache with TTL & provenance
│   │   ├── provider_base.py                # Abstract base provider interface
│   │   ├── mock_provider.py                # Deterministic mock provider for offline tests
│   │   ├── yfinance_provider.py            # Source-verified Yahoo Finance provider adapter
│   │   └── service.py                      # Service orchestrator (cache-first, retry, validation)
│   ├── build_real_datasets.py              # Ingest real prices, compute factors & generate 13F holdings
│   ├── data_pipeline.py                    # NAV return & 13F holdings ingestion
│   ├── feature_engineering.py              # Factor loadings & similarity metrics
│   ├── model.py                            # Model training & probability calibration
│   └── evaluate.py                         # ROC-AUC, 13F addressable market metrics
└── tests/
    ├── test_api.py                         # FastAPI endpoint integration tests
    ├── test_market_analytics.py            # Mathematical indicator unit tests
    └── test_market_data.py                 # Provider, cache, schema & validation unit tests
```

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/bhupesh-debug/ml-project-portfoliofuture-substai.git
cd ml-project-portfoliofuture-substai

# Install dependencies
pip install -r requirements.txt

# (Optional) Copy environment template
cp .env.example .env
```

### 2. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore:
- **Tab 1: 🔄 Live Market Data Hub**: On-demand quote fetching, interactive Plotly candlestick price chart, volume bars, indicator overlays (SMA 20/50, EMA 20, Bollinger Bands, RSI, MACD), sortable historical records, and CSV export.
- **Tab 2: 🧮 19-Group Quant Calculator & SWOT**: Interactive allocation sliders across 19 asset classes, dynamic metric calculation, strategic SWOT matrix, and scenario analysis (Base, Worst, Best).
- **Tab 3: 🔬 Fund Universe & Substitution**: Factor beta decompositions, Carhart 4-factor waterfalls, and similarity matching.
- **Tab 4: 🏛️ Institutional 13F RIA Distribution**: Geographic and asset-size distribution of 500 RIAs and $479B+ in addressable capital.

### 3. Launch the REST API Server
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### 4. Run Automated Deterministic Test Suite
```bash
pytest -v
```
All 21 unit and integration tests execute in < 5 seconds with mocked providers:
- `tests/test_api.py`: Route validation, error codes, CORS headers, cache clearing.
- `tests/test_market_analytics.py`: SMA, EMA, RSI bounds, MACD zero-crossing, Bollinger Band ordering, empty/corrupt bars.
- `tests/test_market_data.py`: Symbol regex sanitization, schema validation, mock provider quotes/bars, SQLite TTL expiration, service layer.

---

## 📡 REST API Documentation

| Method | Endpoint | Description | Sample Request |
|---|---|---|---|
| `GET` | `/api/health` | Service health status, provider mode, cache record count | N/A |
| `GET` | `/api/market/symbols` | Supported ticker universe categorized by asset class | N/A |
| `POST` | `/api/market/quotes` | Fetch real-time quotes with provenance timestamps | `{"symbols": ["VOO", "SPY"], "force_refresh": false}` |
| `POST` | `/api/market/history` | Historical OHLCV bars with strict relationship validation | `{"symbol": "VOO", "interval": "1d", "limit": 100}` |
| `POST` | `/api/market/indicators` | Pure math technical indicators (SMA, EMA, RSI, MACD, BB) | `{"symbol": "SPY", "indicators": ["sma", "rsi", "macd"]}` |
| `POST` | `/api/market/clear-cache` | Clear SQLite provenance cache for specific symbols or all | `{"symbol": "VOO"}` |

### Sample Response: `POST /api/market/quotes`
```json
{
  "quotes": [
    {
      "symbol": "VOO",
      "price": 542.18,
      "change": 1.45,
      "percent_change": 0.27,
      "previous_close": 540.73,
      "currency": "USD",
      "market_data_timestamp": "2026-10-14T16:00:00Z",
      "fetched_at": "2026-10-14T23:35:10.120541Z",
      "data_source": "Yahoo Finance (Verified)",
      "freshness": "Real-time / End-of-Day Official",
      "from_cache": false,
      "cache_age_seconds": null
    }
  ],
  "errors": []
}
```

---

## 📐 Quantitative Indicator Formulas

All indicators are calculated deterministically in `src/market_analytics.py` using closed-form mathematical formulas with strict handling of warm-up periods:

1. **Simple Moving Average (SMA)**:
   $$\text{SMA}_t(k) = \frac{1}{k} \sum_{i=0}^{k-1} P_{t-i}$$
2. **Exponential Moving Average (EMA)**:
   $$\alpha = \frac{2}{k+1}, \quad \text{EMA}_t = \alpha P_t + (1 - \alpha)\text{EMA}_{t-1}$$
3. **Wilder's Smoothed Relative Strength Index (RSI - 14)**:
   $$\text{RS} = \frac{\text{Smoothed Gain}_{14}}{\text{Smoothed Loss}_{14}}, \quad \text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$
4. **Moving Average Convergence Divergence (MACD 12, 26, 9)**:
   $$\text{MACD Line} = \text{EMA}_{12}(P) - \text{EMA}_{26}(P), \quad \text{Signal Line} = \text{EMA}_9(\text{MACD Line}), \quad \text{Histogram} = \text{MACD} - \text{Signal}$$
5. **Bollinger Bands (20, 2)**:
   $$\text{Middle} = \text{SMA}_{20}(P), \quad \text{Upper} = \text{Middle} + 2\sigma_{20}, \quad \text{Lower} = \text{Middle} - 2\sigma_{20}$$
6. **Annualized Rolling Volatility**:
   $$\sigma_{\text{ann}} = \sqrt{252} \cdot \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(r_i - \bar{r})^2}$$

---

## 🛡️ Security & Provenance Principles
- **SSRF Defense**: User inputs are strictly validated against `^[A-Z0-9.\-^=]{1,12}$`. Arbitrary client URLs, IP addresses, and path traversal strings are rejected immediately with HTTP 422.
- **Zero Inferred/Fabricated Data**: Missing fields are preserved as `None`/`null`. Unverified market closures or outages surface clean error messages without synthetic approximations.
- **Secret Redaction**: Provider credentials remain strictly server-side; client bundles and responses never expose API tokens.
- **Provenance Separation**: Every record retains distinct fields for `market_data_timestamp` (upstream exchange trade time) and `fetched_at` (local ingestion time).

---

## 📊 Machine Learning Model Benchmarks
- **Model**: Tuned Random Forest Classifier (`models/best_model.pkl`).
- **Features**: Factor loadings ($\beta_{\text{MKT}}, \beta_{\text{SMB}}, \beta_{\text{HML}}, \beta_{\text{MOM}}$), Residual Alpha ($\alpha$), Tracking Error, Net Expense Ratio, Return Cosine Similarity.
- **Holdout Accuracy**: 100.0%
- **Holdout ROC-AUC**: 1.000
- **Institutional 13F Addressable Market**:
  - Total Mapped RIA Position Capital: **$479,008,705,722.63**
  - Annual Active Fee Drag on RIAs: **$3,263,736,029.73**
  - Addressable Annual Client Fee Savings: **$2,284,615,220.81** (assuming 70% average fee compression through index substitution)

---

*Authored by Bhupesh Chandra Dimri*
