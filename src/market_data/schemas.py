"""
Pydantic Schemas for Financial Market Intelligence.
Defines strict data models for quotes, OHLCV historical bars, indicators, and metadata.
"""

from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re

SYMBOL_REGEX = re.compile(r"^[A-Z0-9.\-^=]{1,12}$")

class MarketSymbol(BaseModel):
    symbol: str
    name: str
    asset_class: str
    exchange: Optional[str] = "US"

class QuoteRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, max_length=25)
    force_refresh: bool = False

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: List[str]) -> List[str]:
        cleaned = []
        for s in v:
            sym = s.strip().upper()
            if not SYMBOL_REGEX.match(sym):
                raise ValueError(f"Invalid symbol format: {s}")
            cleaned.append(sym)
        return list(dict.fromkeys(cleaned))

class MarketQuote(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: Optional[int] = 0
    currency: str = "USD"
    market_data_timestamp: str
    fetched_at: str
    freshness: Literal["REAL_TIME", "DELAYED_15MIN", "END_OF_DAY", "CACHED"]
    source_provider: str
    cache_status: Literal["HIT", "MISS", "BYPASS"]
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None

class HistoryRequest(BaseModel):
    symbol: str
    period: Literal["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"] = "1y"
    interval: Literal["1d", "1wk", "1mo"] = "1d"
    force_refresh: bool = False

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        sym = v.strip().upper()
        if not SYMBOL_REGEX.match(sym):
            raise ValueError(f"Invalid symbol format: {v}")
        return sym

class OHLCVBar(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

class HistoryResponse(BaseModel):
    symbol: str
    period: str
    interval: str
    total_bars: int
    fetched_at: str
    source_provider: str
    cache_status: Literal["HIT", "MISS", "BYPASS"]
    bars: List[OHLCVBar]

class IndicatorRequest(BaseModel):
    symbol: str
    period: Literal["3mo", "6mo", "1y", "2y", "5y"] = "1y"
    interval: Literal["1d", "1wk"] = "1d"
    indicators: List[Literal["SMA", "EMA", "RSI", "MACD", "BOLLINGER", "VOLATILITY"]] = ["SMA", "RSI", "MACD"]
    sma_window: int = Field(20, ge=2, le=200)
    ema_window: int = Field(20, ge=2, le=200)
    rsi_window: int = Field(14, ge=2, le=50)

class IndicatorPoint(BaseModel):
    timestamp: str
    close: float
    indicators: Dict[str, Optional[float]]

class IndicatorResponse(BaseModel):
    symbol: str
    period: str
    total_points: int
    data: List[IndicatorPoint]
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: str
    version: str
    provider: str
    cache_db: str
    cached_records_count: int
