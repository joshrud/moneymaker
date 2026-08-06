"""
Bot 1 — Classical Technical Momentum
Strategy: RSI + MACD + Bollinger Bands crossover.
No ML. Pure rules-based baseline used as the benchmark to beat.

Entry  (long): RSI < 35 AND MACD histogram turning positive AND price near lower BB.
Exit   (long): RSI > 65 OR MACD histogram turns negative.
Position size: ATR-based (via RiskManager).
"""
from __future__ import annotations
from typing import Dict

import pandas as pd

from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed, WATCHLIST
from core.indicators import compute_all


class MomentumBot(BaseBot):
    """RSI / MACD / Bollinger Band momentum strategy."""

    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot1_momentum", client, feed)
        from core.dynamic_watchlist import active_symbols
        self.watchlist = list(dict.fromkeys(WATCHLIST + active_symbols()))

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        """
        Returns a signal in [-1, +1] per symbol.
        We only go long (no shorting in paper account setup).
        """
        signals = {}

        for symbol in self.watchlist:
            try:
                sym_bars = bars_df.xs(symbol, level=0).copy()
            except KeyError:
                continue
            if len(sym_bars) < 30:
                continue

            df = compute_all(sym_bars).dropna()
            if df.empty:
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2]

            rsi_val = last["rsi"]
            macd_hist_now = last["macd_hist"]
            macd_hist_prev = prev["macd_hist"]
            bb_pct_b = last["bb_pct_b"]

            # Long signal: oversold + MACD crossing up + near lower band
            long_score = 0.0
            if rsi_val < 40:
                long_score += 0.4
            if rsi_val < 35:
                long_score += 0.2  # stronger oversold
            if macd_hist_now > 0 and macd_hist_prev <= 0:
                long_score += 0.3  # MACD zero-line crossover
            elif macd_hist_now > macd_hist_prev and macd_hist_now > 0:
                long_score += 0.1
            if bb_pct_b < 0.2:
                long_score += 0.2  # price near lower band

            # Exit / short signal: overbought or MACD crossing down
            exit_score = 0.0
            if rsi_val > 65:
                exit_score += 0.5
            if macd_hist_now < 0 and macd_hist_prev >= 0:
                exit_score += 0.4
            if bb_pct_b > 0.9:
                exit_score += 0.2

            if exit_score > 0.2:
                signals[symbol] = -exit_score  # signal to close / avoid
            elif long_score > 0.2:
                signals[symbol] = long_score
            else:
                signals[symbol] = 0.0

            self.logger.log_signal(symbol, signals[symbol],
                                   reason=f"rsi={rsi_val:.1f},bb={bb_pct_b:.2f}")

        return signals
