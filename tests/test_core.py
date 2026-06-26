"""Unit tests for core utilities. Run with: .venv/bin/pytest tests/"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from core.risk import RiskManager
from core.portfolio import VirtualPortfolio
from core.logger import TradeLogger
from core.indicators import rsi, macd, bollinger, atr, compute_all


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_prices():
    """30 days of synthetic price data with realistic OHLCV."""
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(60) * 0.5)
    high = close + np.abs(np.random.randn(60) * 0.3)
    low = close - np.abs(np.random.randn(60) * 0.3)
    open_ = close + np.random.randn(60) * 0.1
    volume = np.random.randint(1_000_000, 5_000_000, 60).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    })


@pytest.fixture
def portfolio():
    return VirtualPortfolio("test_bot", starting_capital=10_000.0)


@pytest.fixture
def risk_mgr():
    return RiskManager(capital=10_000.0)


# ── Indicator tests ───────────────────────────────────────────────────────────

def test_rsi_range(sample_prices):
    result = rsi(sample_prices["close"])
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all(), "RSI must be in [0, 100]"


def test_rsi_length(sample_prices):
    result = rsi(sample_prices["close"], window=14)
    assert len(result) == len(sample_prices["close"])


def test_macd_columns(sample_prices):
    result = macd(sample_prices["close"])
    assert set(result.columns) == {"macd", "signal", "histogram"}


def test_bollinger_bands_ordering(sample_prices):
    result = bollinger(sample_prices["close"])
    valid = result.dropna()
    assert (valid["upper"] >= valid["mid"]).all()
    assert (valid["mid"] >= valid["lower"]).all()


def test_atr_positive(sample_prices):
    result = atr(sample_prices["high"], sample_prices["low"], sample_prices["close"])
    # ta fills zeros during the warm-up period; check only the values after warm-up
    warmed = result.dropna().iloc[14:]
    assert (warmed > 0).all(), "ATR must be positive after warm-up period"


def test_compute_all_returns_expected_columns(sample_prices):
    result = compute_all(sample_prices)
    for col in ["rsi", "macd_hist", "bb_pct_b", "atr", "ema20"]:
        assert col in result.columns, f"Missing column: {col}"


# ── Portfolio tests ───────────────────────────────────────────────────────────

def test_portfolio_buy_reduces_cash(portfolio):
    initial_cash = portfolio.cash
    portfolio.buy("AAPL", 10, 150.0)
    assert portfolio.cash < initial_cash


def test_portfolio_buy_creates_position(portfolio):
    portfolio.buy("AAPL", 10, 150.0)
    assert "AAPL" in portfolio.positions
    assert portfolio.positions["AAPL"].qty == pytest.approx(10.0, abs=0.1)


def test_portfolio_sell_removes_position(portfolio):
    portfolio.buy("AAPL", 10, 150.0)
    portfolio.sell("AAPL", 10, 155.0)
    assert "AAPL" not in portfolio.positions


def test_portfolio_sell_increases_cash(portfolio):
    portfolio.buy("AAPL", 10, 150.0)
    cash_after_buy = portfolio.cash
    portfolio.sell("AAPL", 10, 155.0)
    assert portfolio.cash > cash_after_buy


def test_portfolio_cannot_oversell(portfolio):
    portfolio.buy("AAPL", 5, 100.0)
    portfolio.sell("AAPL", 100, 110.0)  # try to sell 100 when only 5 held
    assert "AAPL" not in portfolio.positions


def test_portfolio_mark_to_market(portfolio):
    portfolio.buy("AAPL", 10, 100.0)
    val = portfolio.mark_to_market({"AAPL": 120.0})
    assert val > portfolio.cash  # position should be worth more now


def test_portfolio_starting_capital(portfolio):
    assert portfolio.starting_capital == 10_000.0
    assert portfolio.cash == 10_000.0


def test_portfolio_pnl_tracking(portfolio):
    portfolio.buy("AAPL", 10, 100.0)
    portfolio.mark_to_market({"AAPL": 100.0})  # first mark
    portfolio.mark_to_market({"AAPL": 110.0})  # up $100
    assert portfolio.daily_pnl == pytest.approx(100.0 * (1 - 5/10000), abs=5.0)


# ── Risk Manager tests ────────────────────────────────────────────────────────

def test_risk_position_size_atr(risk_mgr):
    qty = risk_mgr.position_size_atr(price=100.0, atr_value=2.0, portfolio_value=10_000.0)
    assert qty > 0
    # Should not exceed max_position (20%) of portfolio
    assert qty * 100.0 <= 10_000.0 * risk_mgr.max_position + 1.0


def test_risk_circuit_breaker_triggers(risk_mgr):
    risk_mgr.reset_day(10_000.0)
    halted = risk_mgr.check_circuit_breaker(9_600.0)  # 4% loss, threshold is 3%
    assert halted is True


def test_risk_circuit_breaker_ok(risk_mgr):
    risk_mgr.reset_day(10_000.0)
    halted = risk_mgr.check_circuit_breaker(9_800.0)  # 2% loss, under threshold
    assert halted is False


def test_risk_sharpe_ratio(risk_mgr):
    returns = pd.Series([0.01, -0.005, 0.008, 0.012, -0.002, 0.007])
    sharpe = risk_mgr.sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)


def test_risk_max_drawdown(risk_mgr):
    values = pd.Series([100, 110, 105, 95, 100, 115])
    mdd = risk_mgr.max_drawdown(values)
    assert mdd < 0  # drawdown is always negative
    assert mdd >= -1.0


# ── Logger tests ──────────────────────────────────────────────────────────────

def test_logger_writes_and_reads(tmp_path, monkeypatch):
    monkeypatch.setattr("core.logger.LOGS_DIR", tmp_path)
    logger = TradeLogger("test_bot")
    logger.log_trade("buy", "AAPL", 10, 150.0, reason="test")
    records = logger.read_today()
    assert len(records) == 1
    assert records[0]["symbol"] == "AAPL"
    assert records[0]["event"] == "trade"


def test_logger_appends_multiple(tmp_path, monkeypatch):
    monkeypatch.setattr("core.logger.LOGS_DIR", tmp_path)
    logger = TradeLogger("test_bot")
    logger.log_trade("buy", "AAPL", 5, 150.0)
    logger.log_trade("sell", "AAPL", 5, 160.0)
    records = logger.read_today()
    assert len(records) == 2
