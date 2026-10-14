"""
Integration tests for FastAPI REST API endpoints.
Uses TestClient with Mock Provider to verify request/response schemas, status codes,
and error handling.
"""

import pytest
import os
from starlette.testclient import TestClient

# Configure mock provider for tests
os.environ["USE_MOCK_PROVIDER"] = "true"

from src.api import app, service
from src.market_data.mock_provider import MockMarketDataProvider

# Ensure service uses MockMarketDataProvider in test session
service.provider = MockMarketDataProvider()
client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "provider" in data

def test_symbols_endpoint():
    response = client.get("/api/market/symbols")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "symbol" in data[0]

def test_quotes_endpoint_valid():
    payload = {"symbols": ["SPY", "VOO"], "force_refresh": True}
    response = client.post("/api/market/quotes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["symbol"] in ["SPY", "VOO"]
    assert data[0]["price"] > 0
    assert data[0]["currency"] == "USD"

def test_quotes_endpoint_invalid_symbol():
    payload = {"symbols": ["INVALID;DROP TABLE"], "force_refresh": True}
    response = client.post("/api/market/quotes", json=payload)
    # Must fail schema regex validation with 422 or 400
    assert response.status_code in [400, 422]

def test_history_endpoint_valid():
    payload = {"symbol": "SPY", "period": "1y", "interval": "1d"}
    response = client.post("/api/market/history", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "SPY"
    assert data["total_bars"] > 100
    assert len(data["bars"]) > 100
    assert "open" in data["bars"][0]
    assert "close" in data["bars"][0]

def test_history_endpoint_invalid_symbol():
    payload = {"symbol": "BAD/TICKER???"}
    response = client.post("/api/market/history", json=payload)
    assert response.status_code in [400, 422]

def test_indicators_endpoint_valid():
    payload = {
        "symbol": "SPY",
        "period": "1y",
        "indicators": ["SMA", "RSI", "MACD"],
        "sma_window": 20,
        "rsi_window": 14
    }
    response = client.post("/api/market/indicators", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "SPY"
    assert data["total_points"] > 100
    assert "indicators" in data["data"][-1]
    assert "sma" in data["data"][-1]["indicators"]
    assert "rsi" in data["data"][-1]["indicators"]

def test_clear_cache_endpoint():
    response = client.post("/api/market/clear-cache")
    assert response.status_code == 200
    assert response.json()["message"] == "Cache successfully cleared"
