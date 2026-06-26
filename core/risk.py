"""
Risk management utilities: position sizing, drawdown tracking, circuit breakers.
All bots call these before submitting any order.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class RiskManager:
    """
    Centralised risk controls applied uniformly across bots.

    Parameters
    ----------
    capital         Starting virtual capital for the bot.
    max_position    Max fraction of capital in any single position (default 20%).
    max_daily_loss  Circuit-breaker: halt trading if daily PnL < -max_daily_loss (default 3%).
    min_atr_mult    Stop-loss distance in ATR multiples (default 1.5).
    """

    def __init__(
        self,
        capital: float = 20_000.0,
        max_position: float = 0.20,
        max_daily_loss: float = 0.03,
        min_atr_mult: float = 1.5,
    ):
        self.capital = capital
        self.max_position = max_position
        self.max_daily_loss = max_daily_loss
        self.min_atr_mult = min_atr_mult
        self._daily_start_value: float = capital
        self._halted: bool = False

    def reset_day(self, current_value: float):
        """Call at market open each day to reset the circuit breaker baseline."""
        self._daily_start_value = current_value
        self._halted = False

    def check_circuit_breaker(self, current_value: float) -> bool:
        """Returns True (halted) if daily loss exceeds threshold."""
        loss_pct = (current_value - self._daily_start_value) / self._daily_start_value
        if loss_pct <= -self.max_daily_loss:
            self._halted = True
        return self._halted

    def position_size_atr(
        self, price: float, atr_value: float, portfolio_value: float
    ) -> float:
        """
        Kelly-inspired ATR position sizing.
        Risk 1% of portfolio per trade; stop = 1.5 * ATR below entry.
        Returns number of shares (fractional allowed).
        """
        if atr_value <= 0 or price <= 0:
            return 0.0
        risk_per_share = self.min_atr_mult * atr_value
        max_risk_dollars = portfolio_value * 0.01  # risk 1% of portfolio
        shares = max_risk_dollars / risk_per_share
        # Cap to max_position of portfolio
        max_shares = (portfolio_value * self.max_position) / price
        return min(shares, max_shares)

    def position_size_fixed(self, price: float, portfolio_value: float) -> float:
        """Fixed fraction: allocate max_position % of portfolio to this asset."""
        if price <= 0:
            return 0.0
        return (portfolio_value * self.max_position) / price

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Annualised Sharpe from a Series of daily returns."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    def max_drawdown(self, values: pd.Series) -> float:
        """Maximum peak-to-trough drawdown fraction (negative value)."""
        peak = values.cummax()
        drawdown = (values - peak) / peak
        return float(drawdown.min())

    def sortino_ratio(self, returns: pd.Series) -> float:
        """Annualised Sortino ratio (uses downside deviation only)."""
        downside = returns[returns < 0]
        if len(downside) < 2 or downside.std() == 0:
            return 0.0
        return float(returns.mean() / downside.std() * np.sqrt(252))

    def calmar_ratio(self, returns: pd.Series) -> float:
        """Annualised return divided by max drawdown magnitude."""
        ann_return = returns.mean() * 252
        values = (1 + returns).cumprod()
        mdd = abs(self.max_drawdown(values))
        return float(ann_return / mdd) if mdd > 0 else 0.0
