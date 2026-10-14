"""
Abstract Base Class for Financial Market Data Providers.
Defines unified interface for fetching real quotes and historical bars.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseMarketDataProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch quotes for a list of validated symbols.
        Returns normalized list of dictionaries complying with MarketQuote schema.
        """
        pass

    @abstractmethod
    def get_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        """
        Fetch OHLCV historical bars for a single symbol.
        Returns normalized dictionary complying with HistoryResponse schema.
        """
        pass
