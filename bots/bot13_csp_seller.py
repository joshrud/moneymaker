"""
Bot 13 — Cash-Secured Put Seller (Value Entry)
For stocks > 3% below their 20-day MA, sells a 21-30 DTE cash-secured put
at a strike 2-3% below current price. Generates premium; if assigned, owns
the stock at a double discount.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict
import pandas as pd
import numpy as np
from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed, WATCHLIST
from core.indicators import ema

NY = ZoneInfo("America/New_York")
MA_DEVIATION_THRESHOLD = -0.03   # stock must be 3%+ below 20-day MA
DTE_MIN = 21
DTE_MAX = 30
STRIKE_DISCOUNT = 0.97           # sell put at 97% of current price
MAX_CONCURRENT_PUTS = 3          # max simultaneous open put positions


class CSPBot(BaseBot):
    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot13_csp_seller", client, feed)
        self.watchlist = WATCHLIST
        self._open_puts: dict[str, str] = {}  # symbol → contract

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        return {}

    def run_once(self) -> dict:
        prices = self.feed.latest_prices()
        portfolio_val = self.portfolio.mark_to_market(prices)

        if len(self._open_puts) >= MAX_CONCURRENT_PUTS:
            self._check_put_expirations()
            return self._end_of_cycle(prices)

        bars = self.feed.daily_bars(symbols=self.watchlist, lookback=30)
        if bars.empty:
            return self._end_of_cycle(prices)
        closes = bars["close"].unstack(level=0) if isinstance(bars.index, pd.MultiIndex) else bars

        today = datetime.now(NY)
        expiry_after = (today + timedelta(days=DTE_MIN)).strftime("%Y-%m-%d")
        expiry_before = (today + timedelta(days=DTE_MAX)).strftime("%Y-%m-%d")

        for symbol in self.watchlist:
            if symbol in self._open_puts:
                continue
            if symbol not in closes.columns:
                continue
            try:
                close_s = closes[symbol].dropna()
                if len(close_s) < 20:
                    continue
                ma20 = float(ema(close_s, window=20).iloc[-1])
                current_price = prices.get(symbol)
                if current_price is None or current_price <= 0:
                    continue
                deviation = (current_price - ma20) / (ma20 + 1e-8)
                if deviation >= MA_DEVIATION_THRESHOLD:
                    continue  # not discounted enough

                target_strike = current_price * STRIKE_DISCOUNT
                contracts = self.client.get_options_chain(
                    symbol, expiry_after, expiry_before, option_type="put", limit=10
                )
                if not contracts:
                    continue

                best = min(contracts,
                           key=lambda c: abs(float(getattr(c, "strike_price", target_strike)) - target_strike),
                           default=None)
                if best is None:
                    continue

                contract_symbol = getattr(best, "symbol", None)
                if not contract_symbol:
                    continue

                order = self.client.option_sell_to_open(contract_symbol, qty=1)
                self._open_puts[symbol] = contract_symbol
                self.logger.log("csp_written", symbol=symbol, contract=contract_symbol,
                                deviation=round(deviation, 4), strike=target_strike)
            except Exception as e:
                self.logger.log("csp_error", symbol=symbol, error=str(e))

        return self._end_of_cycle(prices)

    def _check_put_expirations(self):
        """Remove puts that have been closed or expired."""
        try:
            option_positions = self.client.get_option_positions()
            open_contracts = {getattr(p, "symbol", "") for p in option_positions}
            expired = [sym for sym, contract in self._open_puts.items()
                       if contract not in open_contracts]
            for sym in expired:
                self.logger.log("csp_expired_or_assigned", symbol=sym,
                                contract=self._open_puts[sym])
                del self._open_puts[sym]
        except Exception:
            pass
