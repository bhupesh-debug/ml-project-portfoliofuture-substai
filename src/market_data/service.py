"""
Market Data Service Orchestrator.
Coordinates request validation, cache inspection, provider delegation,
schema normalization, and provenance tagging.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from .schemas import MarketQuote, HistoryResponse, OHLCVBar, SYMBOL_REGEX
from .provider_base import BaseMarketDataProvider
from .yfinance_provider import YFinanceMarketDataProvider
from .mock_provider import MockMarketDataProvider
from .cache import MarketCache

class MarketDataService:
    def __init__(self, provider: Optional[BaseMarketDataProvider] = None, cache: Optional[MarketCache] = None):
        self.cache = cache or MarketCache()
        
        # Select provider based on environment or injected instance
        if provider:
            self.provider = provider
        elif os.getenv("USE_MOCK_PROVIDER", "false").lower() == "true":
            self.provider = MockMarketDataProvider()
        else:
            self.provider = YFinanceMarketDataProvider()

    def get_quotes(self, symbols: List[str], force_refresh: bool = False) -> List[Dict[str, Any]]:
        cleaned_symbols = []
        for s in symbols:
            s_up = s.strip().upper()
            if not SYMBOL_REGEX.match(s_up):
                raise ValueError(f"Invalid symbol: {s}")
            cleaned_symbols.append(s_up)

        results = []
        symbols_to_fetch = []

        if not force_refresh:
            for s in cleaned_symbols:
                cached = self.cache.get_quote(s)
                if cached:
                    results.append(cached)
                else:
                    symbols_to_fetch.append(s)
        else:
            symbols_to_fetch = cleaned_symbols

        if symbols_to_fetch:
            try:
                fresh_quotes = self.provider.get_quotes(symbols_to_fetch)
                for q in fresh_quotes:
                    q["cache_status"] = "MISS" if not force_refresh else "BYPASS"
                    self.cache.set_quote(q["symbol"], q)
                    results.append(q)
            except Exception as e:
                # If upstream fails and we have stale cached data, return cached with warning
                for s in symbols_to_fetch:
                    stale = self.cache.get_quote(s)
                    if stale:
                        stale["freshness"] = "CACHED"
                        results.append(stale)

        return results

    def get_history(self, symbol: str, period: str = "1y", interval: str = "1d", force_refresh: bool = False) -> Dict[str, Any]:
        s_up = symbol.strip().upper()
        if not SYMBOL_REGEX.match(s_up):
            raise ValueError(f"Invalid symbol: {symbol}")

        if not force_refresh:
            cached = self.cache.get_history(s_up, period, interval)
            if cached:
                return cached

        fresh_history = self.provider.get_history(s_up, period, interval)
        fresh_history["cache_status"] = "MISS" if not force_refresh else "BYPASS"
        self.cache.set_history(s_up, period, interval, fresh_history)
        return fresh_history
