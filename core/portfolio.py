"""
Virtual portfolio tracker: simulates trades without placing real Alpaca orders.
Each competing bot owns one VirtualPortfolio instance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

import pandas as pd


COMMISSION_PER_SHARE = 0.0  # Alpaca is commission-free; add slippage below
SLIPPAGE_BPS = 5            # 5 basis-point slippage model


@dataclass
class Position:
    symbol: str
    qty: float
    avg_cost: float

    @property
    def market_value(self) -> float:
        return self.qty * self.avg_cost  # updated via mark_to_market


class VirtualPortfolio:
    """
    Tracks cash, positions, and running PnL for one bot.

    All prices are mid-prices from the data feed (slippage is applied on trade).
    """

    def __init__(self, name: str, starting_capital: float = 20_000.0):
        self.name = name
        self.cash = starting_capital
        self.starting_capital = starting_capital
        self.positions: Dict[str, Position] = {}
        self._equity_history: list[tuple[datetime, float]] = []
        self._trade_history: list[dict] = []

    # ── Trading ───────────────────────────────────────────────────────────────

    def buy(self, symbol: str, qty: float, price: float) -> float:
        """Executes a virtual buy. Returns cost including slippage."""
        if qty <= 0:
            return 0.0
        fill_price = price * (1 + SLIPPAGE_BPS / 10_000)
        cost = qty * fill_price
        if cost > self.cash:
            qty = self.cash / fill_price  # reduce to available cash
            cost = self.cash
        if qty <= 0:
            return 0.0

        self.cash -= cost
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos.qty + qty
            pos.avg_cost = (pos.qty * pos.avg_cost + qty * fill_price) / total_qty
            pos.qty = total_qty
        else:
            self.positions[symbol] = Position(symbol, qty, fill_price)

        self._record_trade("buy", symbol, qty, fill_price)
        return cost

    def sell(self, symbol: str, qty: float, price: float) -> float:
        """Executes a virtual sell. Returns proceeds after slippage."""
        if symbol not in self.positions or qty <= 0:
            return 0.0
        pos = self.positions[symbol]
        qty = min(qty, pos.qty)
        fill_price = price * (1 - SLIPPAGE_BPS / 10_000)
        proceeds = qty * fill_price

        self.cash += proceeds
        pos.qty -= qty
        if pos.qty < 1e-6:
            del self.positions[symbol]

        self._record_trade("sell", symbol, qty, fill_price)
        return proceeds

    def close_all(self, prices: dict):
        """Liquidates all positions at given prices (e.g., end-of-day)."""
        for symbol in list(self.positions):
            price = prices.get(symbol)
            if price:
                self.sell(symbol, self.positions[symbol].qty, price)

    # ── Valuation ─────────────────────────────────────────────────────────────

    def mark_to_market(self, prices: dict) -> float:
        """Updates position prices and returns total portfolio value."""
        equity = self.cash
        for sym, pos in self.positions.items():
            if sym in prices:
                pos.avg_cost = prices[sym]  # use current price for mark
            equity += pos.qty * pos.avg_cost
        self._equity_history.append((datetime.utcnow(), equity))
        return equity

    @property
    def total_value(self) -> float:
        if self._equity_history:
            return self._equity_history[-1][1]
        return self.cash

    @property
    def daily_pnl(self) -> float:
        if len(self._equity_history) < 2:
            return 0.0
        return self._equity_history[-1][1] - self._equity_history[0][1]

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.starting_capital

    @property
    def total_return_pct(self) -> float:
        return self.total_pnl / self.starting_capital * 100

    @property
    def equity_series(self) -> pd.Series:
        if not self._equity_history:
            return pd.Series(dtype=float)
        times, values = zip(*self._equity_history)
        return pd.Series(values, index=times)

    def daily_returns(self) -> pd.Series:
        s = self.equity_series
        return s.pct_change().dropna()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _record_trade(self, side: str, symbol: str, qty: float, price: float):
        self._trade_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "side": side,
            "symbol": symbol,
            "qty": round(qty, 4),
            "price": round(price, 4),
            "portfolio_value": round(self.total_value, 2),
        })

    @property
    def trades(self) -> list[dict]:
        return self._trade_history

    def reset_day(self):
        """Keep equity history but mark start of new day for daily PnL calc."""
        current_val = self.total_value
        self._equity_history = [(datetime.utcnow(), current_val)]
