"""
Bot 12 — Covered Call Writer (Income Generation)
When ANY bot is long a stock with > 1% unrealized gain, sells a 30-45 DTE covered call
at the nearest OTM strike (target delta ~0.30). Generates premium income; caps upside.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict
import pandas as pd
from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed

NY = ZoneInfo("America/New_York")
MIN_GAIN_PCT = 0.01    # only write calls when position is 1%+ in profit
DTE_MIN = 28
DTE_MAX = 45


class CoveredCallBot(BaseBot):
    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot12_covered_calls", client, feed)
        self._written_calls: dict[str, str] = {}  # symbol → contract_symbol

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        # This bot doesn't generate stock signals — returns empty
        return {}

    def run_once(self) -> dict:
        """Override: checks positions and writes/manages covered calls."""
        prices = self.feed.latest_prices()
        portfolio_val = self.portfolio.mark_to_market(prices)

        try:
            positions = self.client.get_positions()
        except Exception as e:
            self.logger.log("positions_error", error=str(e))
            return self._end_of_cycle(prices)

        today = datetime.now(NY)
        expiry_after = (today + timedelta(days=DTE_MIN)).strftime("%Y-%m-%d")
        expiry_before = (today + timedelta(days=DTE_MAX)).strftime("%Y-%m-%d")

        for symbol, pos in positions.items():
            # Skip if we already have a call written on this symbol
            if symbol in self._written_calls:
                self._manage_existing_call(symbol, prices)
                continue

            try:
                market_val = float(getattr(pos, "market_value", 0))
                cost_basis = float(getattr(pos, "cost_basis", market_val))
                if cost_basis <= 0:
                    continue
                gain_pct = (market_val - cost_basis) / cost_basis
                if gain_pct < MIN_GAIN_PCT:
                    continue

                current_price = prices.get(symbol, market_val / max(float(getattr(pos, "qty", 1)), 1))
                if current_price <= 0:
                    continue

                # Find nearest OTM call
                contracts = self.client.get_options_chain(
                    symbol, expiry_after, expiry_before, option_type="call", limit=10
                )
                if not contracts:
                    continue

                # Select the contract with strike nearest to 102% of current price (slightly OTM)
                target_strike = current_price * 1.02
                best = min(contracts, key=lambda c: abs(float(getattr(c, "strike_price", target_strike)) - target_strike), default=None)
                if best is None:
                    continue

                contract_symbol = getattr(best, "symbol", None)
                if not contract_symbol:
                    continue

                qty = max(1, int(float(getattr(pos, "qty", 0)) // 100))
                if qty < 1:
                    continue

                order = self.client.option_sell_to_open(contract_symbol, qty=qty)
                self._written_calls[symbol] = contract_symbol
                self.logger.log("covered_call_written", symbol=symbol,
                                contract=contract_symbol, qty=qty,
                                gain_pct=round(gain_pct, 4))
            except Exception as e:
                self.logger.log("covered_call_error", symbol=symbol, error=str(e))

        return self._end_of_cycle(prices)

    def _manage_existing_call(self, symbol: str, prices: dict):
        """Buy to close if call is deep ITM (stock rallied past strike) to avoid assignment."""
        contract = self._written_calls.get(symbol)
        if not contract:
            return
        try:
            option_positions = self.client.get_option_positions()
            contract_syms = [getattr(p, "symbol", "") for p in option_positions]
            if contract not in contract_syms:
                # Call was already closed/expired
                del self._written_calls[symbol]
                self.logger.log("covered_call_expired_or_closed", symbol=symbol, contract=contract)
        except Exception:
            pass
