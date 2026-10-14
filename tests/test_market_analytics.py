"""
Unit tests for Quantitative Technical Indicators:
SMA, EMA, Wilder's RSI, MACD, Bollinger Bands, and Rolling Volatility.
"""

import pytest
import numpy as np
from src.market_analytics import calculate_indicators

def generate_mock_bars(num_days=60, start_price=100.0, trend=0.5):
    bars = []
    for i in range(num_days):
        p = start_price + (i * trend)
        bars.append({
            "timestamp": f"2026-01-{i+1:02d}" if i < 30 else f"2026-02-{i-29:02d}",
            "open": p - 0.2,
            "high": p + 0.5,
            "low": p - 0.5,
            "close": p,
            "volume": 1000000
        })
    return bars

def test_sma_calculation():
    bars = generate_mock_bars(num_days=30, start_price=100.0, trend=1.0)
    # closes are 100, 101, 102, ..., 129
    res = calculate_indicators(bars, sma_window=5)
    
    # First 4 should be None (warm-up period)
    for i in range(4):
        assert res[i]["indicators"]["sma"] is None
        
    # 5th point (index 4) should be mean of [100, 101, 102, 103, 104] = 102.0
    assert res[4]["indicators"]["sma"] == 102.0

def test_rsi_bounds_and_direction():
    # Strictly increasing sequence -> RSI should approach 100
    up_bars = generate_mock_bars(num_days=50, start_price=50.0, trend=2.0)
    res_up = calculate_indicators(up_bars, rsi_window=14)
    valid_rsi_up = [r["indicators"]["rsi"] for r in res_up if r["indicators"]["rsi"] is not None]
    assert len(valid_rsi_up) > 0
    assert valid_rsi_up[-1] >= 95.0
    
    # Strictly decreasing sequence -> RSI should approach 0
    down_bars = generate_mock_bars(num_days=50, start_price=200.0, trend=-2.0)
    res_down = calculate_indicators(down_bars, rsi_window=14)
    valid_rsi_down = [r["indicators"]["rsi"] for r in res_down if r["indicators"]["rsi"] is not None]
    assert len(valid_rsi_down) > 0
    assert valid_rsi_down[-1] <= 5.0

def test_macd_calculation():
    bars = generate_mock_bars(num_days=70, start_price=100.0, trend=0.2)
    res = calculate_indicators(bars)
    
    # Check that MACD line and signal exist for later bars
    latest = res[-1]["indicators"]
    assert latest["macd_line"] is not None
    assert latest["macd_signal"] is not None
    assert latest["macd_histogram"] is not None
    # Histogram must equal Line - Signal
    assert round(latest["macd_histogram"], 2) == round(latest["macd_line"] - latest["macd_signal"], 2)

def test_bollinger_bands_ordering():
    bars = generate_mock_bars(num_days=40, start_price=150.0, trend=0.1)
    res = calculate_indicators(bars)
    
    for pt in res[25:]:
        inds = pt["indicators"]
        if inds["bb_upper"] is not None and inds["bb_lower"] is not None:
            assert inds["bb_upper"] >= inds["bb_middle"]
            assert inds["bb_middle"] >= inds["bb_lower"]

def test_empty_bars_handling():
    assert calculate_indicators([]) == []

def test_invalid_bars_missing_fields():
    with pytest.raises(ValueError):
        calculate_indicators([{"price": 100}])
