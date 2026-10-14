"""
FastAPI REST Backend for PortfolioFuture Financial Intelligence.
Provides documented, validated market data endpoints with CORS, rate-limit awareness,
and provenance tracking.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from typing import List, Dict, Any
import os

from .market_data.schemas import (
    QuoteRequest,
    MarketQuote,
    HistoryRequest,
    HistoryResponse,
    IndicatorRequest,
    IndicatorResponse,
    HealthResponse,
    MarketSymbol
)
from .market_data.service import MarketDataService
from .market_analytics import calculate_indicators
from .data_pipeline import load_fund_profiles

app = FastAPI(
    title="PortfolioFuture Financial Intelligence API",
    description="Production REST API for real-time market data, fund return decomposition, and technical indicators.",
    version="2.0.0"
)

# CORS Configuration
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

service = MarketDataService()

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Returns system health, provider status, and cache volume."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="2.0.0",
        provider=service.provider.provider_name,
        cache_db=service.cache.db_path,
        cached_records_count=service.cache.count()
    )

@app.get("/api/market/symbols", response_model=List[MarketSymbol], tags=["Market Data"])
def get_symbols():
    """Returns curated universe of liquid ETF and fund symbols."""
    try:
        df_funds = load_fund_profiles()
        unique_funds = df_funds.drop_duplicates(subset=["ticker"]).head(100)
        return [
            MarketSymbol(
                symbol=row["ticker"],
                name=row["fund_name"],
                asset_class=row["asset_class"],
                exchange="US"
            )
            for _, row in unique_funds.iterrows()
        ]
    except Exception as e:
        # Fallback symbols if local files unavailable
        default_symbols = [
            ("VOO", "Vanguard S&P 500 ETF", "Large-Cap Core"),
            ("SPY", "SPDR S&P 500 ETF Trust", "Large-Cap Core"),
            ("QQQ", "Invesco QQQ Trust", "Large-Cap Growth"),
            ("VTV", "Vanguard Value ETF", "Large-Cap Value"),
            ("IWM", "iShares Russell 2000 ETF", "Small-Cap Blend"),
            ("AGG", "iShares Core U.S. Aggregate Bond ETF", "Intermediate Core Bond"),
            ("ARKK", "ARK Innovation ETF", "Large-Cap Growth"),
            ("JEPI", "JPMorgan Equity Premium Income ETF", "Large-Cap Core")
        ]
        return [
            MarketSymbol(symbol=s, name=n, asset_class=a, exchange="US")
            for s, n, a in default_symbols
        ]

@app.post("/api/market/quotes", response_model=List[MarketQuote], tags=["Market Data"])
def get_quotes(req: QuoteRequest):
    """
    On-demand fetch for current market quotes across requested symbols.
    Checks local cache before contacting upstream provider.
    """
    try:
        quotes = service.get_quotes(req.symbols, force_refresh=req.force_refresh)
        return quotes
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Provider retrieval error: {str(e)}")

@app.post("/api/market/history", response_model=HistoryResponse, tags=["Market Data"])
def get_history(req: HistoryRequest):
    """
    On-demand fetch for historical OHLCV bars for a single symbol.
    Enforces strict mathematical OHLC relationships.
    """
    try:
        history = service.get_history(req.symbol, period=req.period, interval=req.interval, force_refresh=req.force_refresh)
        return history
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Upstream provider failure: {str(e)}")

@app.post("/api/market/indicators", response_model=IndicatorResponse, tags=["Quantitative Analytics"])
def get_indicators(req: IndicatorRequest):
    """
    Calculates technical and econometric indicators (SMA, EMA, RSI, MACD, Bollinger Bands, Volatility)
    on retrieved historical price series without lookahead bias.
    """
    try:
        history = service.get_history(req.symbol, period=req.period, interval=req.interval, force_refresh=False)
        bars = history.get("bars", [])
        if not bars:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No price history available for {req.symbol}")

        indicator_data = calculate_indicators(
            bars=bars,
            sma_window=req.sma_window,
            ema_window=req.ema_window,
            rsi_window=req.rsi_window
        )

        return IndicatorResponse(
            symbol=req.symbol,
            period=req.period,
            total_points=len(indicator_data),
            data=indicator_data,
            metadata={
                "sma_window": req.sma_window,
                "ema_window": req.ema_window,
                "rsi_window": req.rsi_window,
                "macd_config": "12, 26, 9",
                "bollinger_config": "20, 2 std",
                "volatility_lookback": "20 days annualized (sqrt(252))"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Indicator calculation error: {str(e)}")

@app.post("/api/market/clear-cache", tags=["System"])
def clear_cache():
    """Flushes local SQLite quotes and history cache."""
    service.cache.clear()
    return {"message": "Cache successfully cleared", "timestamp": datetime.now(timezone.utc).isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
