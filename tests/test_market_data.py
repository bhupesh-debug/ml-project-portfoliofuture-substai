"""
Unit tests for Market Data Provider, Schemas, Validation, and SQLite Cache.
"""

import pytest
import os
import time
from src.market_data.schemas import (
    QuoteRequest, HistoryRequest, MarketQuote, OHLCVBar, SYMBOL_REGEX
)
from src.market_data.mock_provider import MockMarketDataProvider
from src.market_data.cache import MarketCache
from src.market_data.service import MarketDataService

def test_symbol_regex_validation():
    assert SYMBOL_REGEX.match("SPY")
    assert SYMBOL_REGEX.match("BRK.B")
    assert SYMBOL_REGEX.match("BF-B")
    assert SYMBOL_REGEX.match("^VIX")
    
    # Invalids (prevent SSRF, injection, and invalid formats)
    assert not SYMBOL_REGEX.match("http://evil.com")
    assert not SYMBOL_REGEX.match("SPY; DROP TABLE")
    assert not SYMBOL_REGEX.match("VERY_LONG_SYMBOL_NAME_123456")
    assert not SYMBOL_REGEX.match("<script>")

def test_quote_request_validation():
    req = QuoteRequest(symbols=[" spy ", "qqq", "SPY"])
    assert req.symbols == ["SPY", "QQQ"]
    
    with pytest.raises(ValueError):
        QuoteRequest(symbols=["INVALID/SYMBOL?"])

def test_history_request_validation():
    req = HistoryRequest(symbol="aapl", period="1y", interval="1d")
    assert req.symbol == "AAPL"
    assert req.period == "1y"

    with pytest.raises(ValueError):
        HistoryRequest(symbol="BAD_TICKER$$$")

def test_mock_provider_quotes():
    provider = MockMarketDataProvider()
    quotes = provider.get_quotes(["SPY", "QQQ"])
    assert len(quotes) == 2
    assert quotes[0]["symbol"] == "SPY"
    assert quotes[0]["price"] > 0
    assert quotes[0]["source_provider"] == "MOCK_PROVIDER"

def test_mock_provider_history_ohlc_relationships():
    provider = MockMarketDataProvider()
    history = provider.get_history("SPY", period="1y", interval="1d")
    assert history["symbol"] == "SPY"
    assert history["total_bars"] > 100
    
    for bar in history["bars"]:
        # Mathematical integrity: High must be highest, Low lowest
        assert bar["high"] >= max(bar["open"], bar["close"])
        assert bar["low"] <= min(bar["open"], bar["close"])
        assert bar["volume"] >= 0

def test_sqlite_cache_ttl_and_hit(tmp_path):
    test_db = str(tmp_path / "test_cache.sqlite")
    cache = MarketCache(db_path=test_db, default_quote_ttl_sec=1, default_history_ttl_sec=2)
    
    # Store quote
    quote_payload = {
        "symbol": "SPY",
        "price": 500.0,
        "change": 2.5,
        "change_percent": 0.5,
        "volume": 1000000,
        "source_provider": "TEST",
        "freshness": "REAL_TIME"
    }
    cache.set_quote("SPY", quote_payload)
    
    # Instant retrieval -> HIT
    retrieved = cache.get_quote("SPY")
    assert retrieved is not None
    assert retrieved["price"] == 500.0
    assert retrieved["cache_status"] == "HIT"
    
    # Wait for TTL expiration
    time.sleep(1.1)
    expired = cache.get_quote("SPY")
    assert expired is None

def test_market_service_with_mock_provider(tmp_path):
    test_db = str(tmp_path / "test_service_cache.sqlite")
    cache = MarketCache(db_path=test_db)
    provider = MockMarketDataProvider()
    service = MarketDataService(provider=provider, cache=cache)
    
    # First fetch -> MISS
    quotes_1 = service.get_quotes(["SPY", "VOO"])
    assert len(quotes_1) == 2
    assert quotes_1[0]["cache_status"] == "MISS"
    
    # Second fetch -> HIT
    quotes_2 = service.get_quotes(["SPY", "VOO"])
    assert quotes_2[0]["cache_status"] == "HIT"
    
    # Force refresh -> BYPASS
    quotes_3 = service.get_quotes(["SPY"], force_refresh=True)
    assert quotes_3[0]["cache_status"] == "BYPASS"
