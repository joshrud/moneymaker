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

    # ── Options ───────────────────────────────────────────────────────────────

    def get_options_chain(self, symbol: str, expiry_after: str, expiry_before: str,
                          option_type: str = "call", limit: int = 20) -> list:
        """Returns list of option contract objects for given symbol and criteria."""
        try:
            from alpaca.trading.requests import GetOptionContractsRequest
            req = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                expiration_date_gte=expiry_after,
                expiration_date_lte=expiry_before,
                type=option_type,
                limit=limit,
            )
            resp = self.trader.get_option_contracts(req)
            return list(resp) if resp else []
        except Exception:
            return []

    def option_sell_to_open(self, contract_symbol: str, qty: int = 1) -> object:
        """Sell to open an option contract (covered call or CSP)."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        req = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return self.trader.submit_order(req)

    def option_buy_to_close(self, contract_symbol: str, qty: int = 1) -> object:
        """Buy to close an existing short option position."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        req = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        return self.trader.submit_order(req)

    def get_option_positions(self) -> list:
        """Returns current open option positions."""
        try:
            positions = self.trader.get_all_positions()
            return [p for p in positions if getattr(p, 'asset_class', '') == 'us_option']
        except Exception:
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        clock = self.trader.get_clock()
        return clock.is_open

    def get_clock(self):
        return self.trader.get_clock()
