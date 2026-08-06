"""
Bot 6 — Triple EMA Trend Following + OBV Confirmation
Strategy: Bull when price > EMA8 > EMA21 > EMA55 AND OBV trending up.
More active than bot1 — fires on partial EMA alignment without needing deep oversold.
"""
from __future__ import annotations
from typing import Dict
import pandas as pd
import numpy as np
from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed, WATCHLIST
from core.indicators import compute_all, ema, obv


class EMABot(BaseBot):
    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot6_ema_trend", client, feed)
        from core.dynamic_watchlist import active_symbols
        self.watchlist = list(dict.fromkeys(WATCHLIST + active_symbols()))

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        signals = {}
        for symbol in self.watchlist:
            try:
                sym_bars = bars_df.xs(symbol, level=0).copy()
            except KeyError:
                continue
            if len(sym_bars) < 60:
                continue
            df = compute_all(sym_bars).dropna()
            if df.empty or len(df) < 3:
                continue
            last = df.iloc[-1]
            prev = df.iloc[-2]
            close = last["close"]
            e8 = last["ema8"] if "ema8" in df.columns else last["ema20"]
            e21 = last["ema20"]
            e55 = last["ema50"]
            # OBV slope: positive = volume confirming
            obv_now = obv(df["close"], df["volume"]).iloc[-1]
            obv_prev = obv(df["close"], df["volume"]).iloc[-6]
            obv_rising = obv_now > obv_prev

            long_score = 0.0
            if close > e8 > e21 > e55:       # full bull alignment
                long_score += 0.5
            elif close > e8 > e21:            # partial alignment
                long_score += 0.3
            elif close > e8:
                long_score += 0.1
            if obv_rising:
                long_score += 0.3
            if last.get("macd_hist", 0) > 0:
                long_score += 0.2

            exit_score = 0.0
            if close < e8:
                exit_score += 0.3
            if e8 < e21:                      # EMA death cross
                exit_score += 0.4
            if last.get("macd_hist", 0) < 0 and prev.get("macd_hist", 0) >= 0:
                exit_score += 0.3

            if exit_score > 0.3:
                signals[symbol] = -exit_score
            elif long_score > 0.2:
                signals[symbol] = long_score
            else:
                signals[symbol] = 0.0
            self.logger.log_signal(symbol, signals[symbol],
                                   reason=f"ema_align={'bull' if close>e8>e21 else 'none'},obv={'up' if obv_rising else 'dn'}")
        return signals
