"""
Quantitative Indicator Analysis Engine.
Calculates mathematically rigorous indicators without look-ahead bias:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (Wilder's RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands (Upper, Middle, Lower)
- Rolling Annualized Volatility
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

def calculate_indicators(
    bars: List[Dict[str, Any]],
    sma_window: int = 20,
    ema_window: int = 20,
    rsi_window: int = 14
) -> List[Dict[str, Any]]:
    """
    Computes technical and quantitative indicators on a sequence of OHLCV bars.
    Returns list of point dictionaries containing date, close, and calculated indicator values.
    """
    if not bars:
        return []

    df = pd.DataFrame(bars)
    if "close" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("Bars missing required 'close' or 'timestamp' fields")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    closes = df["close"]

    # 1. Simple Moving Average (SMA)
    # Formula: SMA_n(t) = (1/n) * sum_{i=0}^{n-1} P_{t-i}
    df["sma"] = closes.rolling(window=sma_window, min_periods=sma_window).mean()

    # 2. Exponential Moving Average (EMA)
    # Formula: EMA_t = P_t * alpha + EMA_{t-1} * (1 - alpha), alpha = 2 / (window + 1)
    df["ema"] = closes.ewm(span=ema_window, adjust=False, min_periods=ema_window).mean()

    # 3. Wilder's RSI (Relative Strength Index)
    # Delta = P_t - P_{t-1}
    # Gain = max(Delta, 0), Loss = max(-Delta, 0)
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's smoothing uses alpha = 1 / window
    avg_gain = gain.ewm(alpha=1.0 / rsi_window, adjust=False, min_periods=rsi_window).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_window, adjust=False, min_periods=rsi_window).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    # Handle zero loss edge-case (RSI = 100)
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100.0
    df.loc[(avg_loss == 0) & (avg_gain == 0), "rsi"] = 50.0

    # 4. MACD (12, 26, 9)
    # Fast EMA(12) - Slow EMA(26)
    ema_fast = closes.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = closes.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd_line"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

    # 5. Bollinger Bands (20-day, 2 std dev)
    # Middle = SMA_20, Upper = SMA_20 + 2*sigma, Lower = SMA_20 - 2*sigma
    bb_window = 20
    sma_bb = closes.rolling(window=bb_window, min_periods=bb_window).mean()
    std_bb = closes.rolling(window=bb_window, min_periods=bb_window).std()
    df["bb_middle"] = sma_bb
    df["bb_upper"] = sma_bb + (2.0 * std_bb)
    df["bb_lower"] = sma_bb - (2.0 * std_bb)

    # 6. Rolling Annualized Volatility (20-day)
    # Returns = (P_t - P_{t-1}) / P_{t-1}
    # Annualized Vol = std(returns, 20) * sqrt(252) * 100%
    returns = closes.pct_change()
    df["volatility_ann_pct"] = returns.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100.0

    output = []
    for _, row in df.iterrows():
        indicators_dict = {
            "sma": round(float(row["sma"]), 2) if not pd.isna(row["sma"]) else None,
            "ema": round(float(row["ema"]), 2) if not pd.isna(row["ema"]) else None,
            "rsi": round(float(row["rsi"]), 2) if not pd.isna(row["rsi"]) else None,
            "macd_line": round(float(row["macd_line"]), 3) if not pd.isna(row["macd_line"]) else None,
            "macd_signal": round(float(row["macd_signal"]), 3) if not pd.isna(row["macd_signal"]) else None,
            "macd_histogram": round(float(row["macd_histogram"]), 3) if not pd.isna(row["macd_histogram"]) else None,
            "bb_upper": round(float(row["bb_upper"]), 2) if not pd.isna(row["bb_upper"]) else None,
            "bb_middle": round(float(row["bb_middle"]), 2) if not pd.isna(row["bb_middle"]) else None,
            "bb_lower": round(float(row["bb_lower"]), 2) if not pd.isna(row["bb_lower"]) else None,
            "volatility_ann_pct": round(float(row["volatility_ann_pct"]), 2) if not pd.isna(row["volatility_ann_pct"]) else None,
        }
        output.append({
            "timestamp": str(row["timestamp"]),
            "close": round(float(row["close"]), 2),
            "indicators": indicators_dict
        })

    return output
