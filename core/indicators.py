"""
Technical indicator helpers wrapping the `ta` library.
Each function takes a price DataFrame and returns a same-indexed Series or DataFrame.
"""
from __future__ import annotations

import pandas as pd
import ta


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    return ta.momentum.RSIIndicator(close, window=window).rsi()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Returns DataFrame with columns: macd, signal, histogram."""
    ind = ta.trend.MACD(close, window_fast=fast, window_slow=slow, window_sign=signal)
    return pd.DataFrame({
        "macd": ind.macd(),
        "signal": ind.macd_signal(),
        "histogram": ind.macd_diff(),
    })


def bollinger(close: pd.Series, window: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Returns DataFrame with columns: upper, mid, lower, pct_b (position within bands)."""
    ind = ta.volatility.BollingerBands(close, window=window, window_dev=std)
    return pd.DataFrame({
        "upper": ind.bollinger_hband(),
        "mid": ind.bollinger_mavg(),
        "lower": ind.bollinger_lband(),
        "pct_b": ind.bollinger_pband(),
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range — used for position sizing and stop placement."""
    return ta.volatility.AverageTrueRange(high, low, close, window=window).average_true_range()


def ema(close: pd.Series, window: int = 20) -> pd.Series:
    return ta.trend.EMAIndicator(close, window=window).ema_indicator()


def sma(close: pd.Series, window: int = 20) -> pd.Series:
    return ta.trend.SMAIndicator(close, window=window).sma_indicator()


def stoch_rsi(close: pd.Series, window: int = 14, smooth: int = 3) -> pd.DataFrame:
    """Stochastic RSI — returns k and d lines."""
    ind = ta.momentum.StochRSIIndicator(close, window=window, smooth1=smooth, smooth2=smooth)
    return pd.DataFrame({"stoch_rsi_k": ind.stochrsi_k(), "stoch_rsi_d": ind.stochrsi_d()})


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    return ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index — trend strength 0-100."""
    return ta.trend.ADXIndicator(high, low, close, window=window).adx()


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends RSI, MACD histogram, BB %B, ATR, EMA20, EMA50, EMA8, EMA55, OBV, ADX
    to an OHLCV DataFrame.
    Expects columns: open, high, low, close, volume.
    """
    out = df.copy()
    out["rsi"] = rsi(df["close"])
    macd_df = macd(df["close"])
    out["macd_hist"] = macd_df["histogram"]
    out["macd_line"] = macd_df["macd"]
    out["macd_signal"] = macd_df["signal"]
    bb = bollinger(df["close"])
    out["bb_pct_b"] = bb["pct_b"]
    out["bb_upper"] = bb["upper"]
    out["bb_lower"] = bb["lower"]
    out["atr"] = atr(df["high"], df["low"], df["close"])
    out["ema20"] = ema(df["close"])
    out["ema50"] = ema(df["close"], window=50)
    out["ema8"] = ema(df["close"], window=8)
    out["ema55"] = ema(df["close"], window=55)
    out["obv"] = obv(df["close"], df["volume"])
    out["adx"] = adx(df["high"], df["low"], df["close"])
    return out
