"""
Deterministic Mock Financial Provider for Unit Testing & Offline Verification.
Produces mathematical, schema-compliant quotes and historical bars without network calls.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from .provider_base import BaseMarketDataProvider

class MockMarketDataProvider(BaseMarketDataProvider):
    @property
    def provider_name(self) -> str:
        return "MOCK_PROVIDER"

    def get_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        results = []
        now = datetime.now(timezone.utc).isoformat()
        
        base_prices = {
            "SPY": 510.50,
            "VOO": 470.20,
            "QQQ": 440.10,
            "IWM": 205.80,
            "AGG": 98.40,
            "ARKK": 48.60,
            "AAPL": 182.50
        }
        
        for s in symbols:
            p = base_prices.get(s, 100.0)
            chg = round(p * 0.0075, 2)
            pct = round((chg / p) * 100, 2)
            results.append({
                "symbol": s,
                "price": p,
                "change": chg,
                "change_percent": pct,
                "volume": 25000000,
                "currency": "USD",
                "market_data_timestamp": now,
                "fetched_at": now,
                "freshness": "REAL_TIME",
                "source_provider": self.provider_name,
                "cache_status": "MISS",
                "high_52w": round(p * 1.15, 2),
                "low_52w": round(p * 0.85, 2)
            })
        return results

    def get_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        num_days = 252 if period in ["1y", "max"] else (63 if period == "3mo" else 126)
        
        base_price = 100.0
        bars = []
        for i in range(num_days):
            day = now - timedelta(days=(num_days - i))
            # Sine wave with drift for determinism
            drift = 1.0 + (i * 0.0005)
            open_p = round(base_price * drift * (1.0 + (i % 5) * 0.002), 2)
            close_p = round(open_p * (1.0 + ((i % 7) - 3) * 0.003), 2)
            high_p = round(max(open_p, close_p) * 1.008, 2)
            low_p = round(min(open_p, close_p) * 0.992, 2)
            volume = 5000000 + (i * 10000)
            
            bars.append({
                "timestamp": day.strftime("%Y-%m-%d"),
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume,
                "adjusted_close": close_p
            })
            
        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "total_bars": len(bars),
            "fetched_at": now.isoformat(),
            "source_provider": self.provider_name,
            "cache_status": "MISS",
            "bars": bars
        }
