"""Abstract base class every bot inherits from. Enforces a shared interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
from typing import Dict

from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed
from core.risk import RiskManager
from core.portfolio import VirtualPortfolio
from core.logger import TradeLogger


class BaseBot(ABC):
    """
    Each subclass implements generate_signals() returning {symbol: float}.
    Positive signal → buy, negative → sell/short, 0 → hold.
    Signal magnitude (0–1) is used for position sizing.
    """

    def __init__(
        self,
        name: str,
        client: AlpacaClient,
        feed: DataFeed,
        starting_capital: float = 20_000.0,
    ):
        self.name = name
        self.client = client
        self.feed = feed
        self.portfolio = VirtualPortfolio(name, starting_capital)
        self.risk = RiskManager(capital=starting_capital)
        self.logger = TradeLogger(name)

    @abstractmethod
    def generate_signals(self, bars: dict) -> Dict[str, float]:
        """
        Given current market data, return a signal per symbol.
        Returns {symbol: signal} where signal ∈ [-1, +1].
        """
        ...

    def run_once(self) -> dict:
        """
        Full cycle: fetch data → generate signals → execute virtual trades → log.
        Returns summary dict.
        """
        # Fetch data
        bars_df = self.feed.daily_bars()
        prices = self.feed.latest_prices()

        # Circuit breaker
        portfolio_val = self.portfolio.mark_to_market(prices)
        if self.risk.check_circuit_breaker(portfolio_val):
            self.logger.log("circuit_breaker", reason="daily loss limit hit")
            return self._end_of_cycle(prices)

        # Generate and execute signals
        signals = self.generate_signals(bars_df)
        for symbol, signal in signals.items():
            price = prices.get(symbol)
            if price is None:
                continue
            self._execute_signal(symbol, signal, price, portfolio_val)

        return self._end_of_cycle(prices)

    def _execute_signal(self, symbol: str, signal: float, price: float, pv: float):
        """Converts a [-1,+1] signal into a virtual portfolio trade."""
        bars_df = self.feed.daily_bars([symbol])
        atr_val = 0.0
        try:
            from core.indicators import atr as calc_atr
            sym_bars = bars_df.xs(symbol, level=0) if symbol in bars_df.index.get_level_values(0) else None
            if sym_bars is not None and len(sym_bars) >= 14:
                atr_val = float(calc_atr(sym_bars["high"], sym_bars["low"], sym_bars["close"]).iloc[-1])
        except Exception:
            pass

        qty = self.risk.position_size_atr(price, atr_val, pv) * abs(signal)
        if qty < 0.01:
            return

        has_position = symbol in self.portfolio.positions
        if signal > 0.1 and not has_position:
            self.portfolio.buy(symbol, qty, price)
            self.logger.log_trade("buy", symbol, qty, price, reason=f"signal={signal:.3f}")
        elif signal < -0.1 and has_position:
            pos_qty = self.portfolio.positions[symbol].qty
            self.portfolio.sell(symbol, pos_qty, price)
            self.logger.log_trade("sell", symbol, pos_qty, price, reason=f"signal={signal:.3f}")

    def _end_of_cycle(self, prices: dict) -> dict:
        pv = self.portfolio.mark_to_market(prices)
        daily_pnl = self.portfolio.daily_pnl
        returns = self.portfolio.daily_returns()
        sharpe = self.risk.sharpe_ratio(returns) if len(returns) > 1 else 0.0
        mdd = self.risk.max_drawdown(self.portfolio.equity_series) if len(self.portfolio.equity_series) > 1 else 0.0

        self.logger.log_summary(
            portfolio_value=pv,
            daily_pnl=daily_pnl,
            sharpe=round(sharpe, 4),
            max_drawdown=round(mdd * 100, 4),
        )
        return {"bot": self.name, "portfolio_value": pv, "daily_pnl": daily_pnl, "sharpe": sharpe}

    def reset_day(self):
        """Call at market open to reset circuit breaker and daily PnL baseline."""
        pv = self.portfolio.total_value
        self.risk.reset_day(pv)
        self.portfolio.reset_day()
