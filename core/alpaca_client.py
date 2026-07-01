"""Thin wrapper around alpaca-py for paper trading. All bots share one instance."""
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

NY = ZoneInfo("America/New_York")


class AlpacaClient:
    """Wraps Alpaca trading + data APIs. Uses paper=True by default."""

    def __init__(self):
        key = os.environ["ALPACA_API_KEY"]
        secret = os.environ["ALPACA_SECRET_KEY"]
        self.trader = TradingClient(key, secret, paper=True)
        self.data = StockHistoricalDataClient(key, secret)
        self.news_client = NewsClient(key, secret)

    # ── Account ──────────────────────────────────────────────────────────────

    def get_account(self):
        return self.trader.get_account()

    def get_positions(self) -> dict:
        """Returns {symbol: position_object}."""
        return {p.symbol: p for p in self.trader.get_all_positions()}

    # ── Orders ────────────────────────────────────────────────────────────────

    def market_buy(self, symbol: str, qty: float) -> object:
        req = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
        )
        return self.trader.submit_order(req)

    def market_sell(self, symbol: str, qty: float) -> object:
        req = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY
        )
        return self.trader.submit_order(req)

    def close_position(self, symbol: str) -> object:
        return self.trader.close_position(symbol)

    def close_all_positions(self):
        return self.trader.close_all_positions(cancel_orders=True)

    # ── Market data ───────────────────────────────────────────────────────────

    def get_bars(
        self,
        symbols: list,
        timeframe: TimeFrame = TimeFrame.Day,
        start: datetime = None,
        end: datetime = None,
        limit: int = 252,
    ) -> pd.DataFrame:
        """Returns a multi-index DataFrame (symbol, timestamp) with OHLCV columns."""
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
        )
        bars = self.data.get_stock_bars(req)
        return bars.df

    def get_latest_quotes(self, symbols: list) -> dict:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        return self.data.get_stock_latest_quote(req)

    def get_news(self, symbols: list, limit: int = 10) -> list:
        """
        Returns recent news articles as plain dicts: {headline, symbols, source, summary, ...}.
        NewsRequest expects symbols as a comma-joined string.
        """
        req = NewsRequest(symbols=",".join(symbols), limit=limit)
        news_set = self.news_client.get_news(req)
        data = news_set.data
        # API returns either {"news": [...]} or [...] depending on SDK version
        if isinstance(data, list):
            return data
        return data.get("news", [])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        clock = self.trader.get_clock()
        return clock.is_open

    def get_clock(self):
        return self.trader.get_clock()
