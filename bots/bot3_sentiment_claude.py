"""
Bot 3 — LLM Sentiment (Claude API)
Strategy: Fetches pre-market news via Alpaca, sends headlines to Claude Haiku,
receives per-symbol bullish/bearish conviction scores, sizes positions accordingly.
Claude acts as a reasoning layer — not a black-box predictor.
Falls back gracefully if ANTHROPIC_API_KEY is not set.
"""
from __future__ import annotations
import os
from typing import Dict

import pandas as pd

from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed, WATCHLIST
from core.sentiment import claude_sentiment_score, finbert_score
from core.indicators import rsi, macd


class ClaudeSentimentBot(BaseBot):
    """
    Uses Claude Haiku to reason over news headlines before each trading session.
    Only trades symbols where Claude assigns high-conviction sentiment.
    A technical filter (RSI + MACD trend direction) must agree with the sentiment.
    """

    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot3_sentiment_claude", client, feed)
        self.watchlist = WATCHLIST
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.use_claude = bool(self.api_key)
        if not self.use_claude:
            self.logger.log("warning", msg="ANTHROPIC_API_KEY not set — using FinBERT fallback")

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        """
        Fetches news → scores sentiment → filters with technicals → returns signals.
        """
        # Bail out if market closes within 10 min — Claude API latency can push execution past 4pm
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.hour == 15 and now.minute >= 50:
            self.logger.log("skipped", reason="within 10 min of market close")
            return {sym: 0.0 for sym in self.watchlist}

        # Fetch recent news headlines
        try:
            news_items = self.feed.get_news(self.watchlist, limit=30)
        except Exception as e:
            self.logger.log("news_fetch_error", error=str(e))
            return {}

        # Group headlines by symbol
        from collections import defaultdict
        symbol_headlines: dict[str, list[str]] = defaultdict(list)
        for item in news_items:
            if isinstance(item, dict):
                headline = item.get("headline", "")
                syms = item.get("symbols", [])
            else:
                headline = getattr(item, "headline", str(item))
                syms = getattr(item, "symbols", [])
            for sym in syms:
                if sym in self.watchlist:
                    symbol_headlines[sym].append(headline)

        signals = {}
        for symbol in self.watchlist:
            headlines = symbol_headlines.get(symbol, [])

            # Sentiment score
            if headlines:
                if self.use_claude:
                    sent_score, reason = claude_sentiment_score(headlines, symbol, self.api_key)
                else:
                    sent_score = finbert_score(headlines)
                    reason = "finbert"
            else:
                sent_score, reason = 0.0, "no news"

            # Technical confirmation filter
            try:
                sym_bars = bars_df.xs(symbol, level=0)
                close_s = sym_bars["close"]
                if len(close_s) >= 14:
                    rsi_val = float(rsi(close_s).iloc[-1])
                    macd_hist = float(macd(close_s)["histogram"].iloc[-1])
                else:
                    rsi_val, macd_hist = 50.0, 0.0
            except (KeyError, Exception):
                rsi_val, macd_hist = 50.0, 0.0

            # Only trade when sentiment and technicals agree
            tech_bullish = rsi_val < 60 and macd_hist > 0
            tech_bearish = rsi_val > 40 and macd_hist < 0

            if sent_score > 0.3 and tech_bullish:
                signals[symbol] = sent_score
            elif sent_score < -0.3 and tech_bearish:
                signals[symbol] = sent_score
            else:
                signals[symbol] = 0.0

            self.logger.log_signal(symbol, signals[symbol],
                                   reason=f"sent={sent_score:.2f},{reason},rsi={rsi_val:.1f}")

        return signals
