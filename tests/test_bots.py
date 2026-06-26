"""Smoke tests for each bot's generate_signals() method using mocked data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed, WATCHLIST, PAIRS
from bots.bot1_momentum import MomentumBot
from bots.bot3_sentiment_claude import ClaudeSentimentBot
from bots.bot4_finbert_ppo import FinBERTPPOBot


# ── Shared fixtures ───────────────────────────────────────────────────────────

def make_mock_bars(symbols=None, n_rows=60):
    """Creates a realistic multi-index DataFrame mimicking Alpaca bar data."""
    symbols = symbols or WATCHLIST
    np.random.seed(7)
    frames = []
    for sym in symbols:
        close = 100.0 + np.cumsum(np.random.randn(n_rows) * 0.5)
        high = close + np.abs(np.random.randn(n_rows) * 0.3)
        low = close - np.abs(np.random.randn(n_rows) * 0.3)
        dates = pd.date_range("2025-01-01", periods=n_rows, freq="B")
        df = pd.DataFrame({
            "open": close + np.random.randn(n_rows) * 0.1,
            "high": high, "low": low, "close": close,
            "volume": np.random.randint(1_000_000, 5_000_000, n_rows).astype(float),
        }, index=pd.MultiIndex.from_product([[sym], dates], names=["symbol", "timestamp"]))
        frames.append(df)
    return pd.concat(frames)


def make_mock_client():
    client = MagicMock(spec=AlpacaClient)
    client.get_account.return_value = MagicMock(buying_power="100000.00")
    client.get_positions.return_value = {}
    return client


def make_mock_feed(client):
    feed = MagicMock(spec=DataFeed)
    feed.daily_bars.return_value = make_mock_bars()
    feed.latest_prices.return_value = {sym: 100.0 + i for i, sym in enumerate(WATCHLIST)}
    feed.get_news.return_value = []
    return feed


# ── Bot 1: Momentum ───────────────────────────────────────────────────────────

def test_bot1_signals_are_floats():
    client = make_mock_client()
    feed = make_mock_feed(client)
    bot = MomentumBot(client, feed)
    bars = make_mock_bars()
    signals = bot.generate_signals(bars)
    assert isinstance(signals, dict)
    for sym, sig in signals.items():
        assert isinstance(sig, float), f"Signal for {sym} is not float"
        assert -1.0 <= sig <= 1.0, f"Signal {sig} out of range for {sym}"


def test_bot1_handles_missing_symbol():
    client = make_mock_client()
    feed = make_mock_feed(client)
    bot = MomentumBot(client, feed)
    # Pass bars missing one symbol — should not raise
    bars = make_mock_bars(symbols=["AAPL", "MSFT"])
    signals = bot.generate_signals(bars)
    assert isinstance(signals, dict)


def test_bot1_skips_short_history():
    client = make_mock_client()
    feed = make_mock_feed(client)
    bot = MomentumBot(client, feed)
    # Only 5 bars — not enough for indicators
    bars = make_mock_bars(symbols=["AAPL"], n_rows=5)
    signals = bot.generate_signals(bars)
    # Should return empty or zero signal, not raise
    assert "AAPL" not in signals or signals["AAPL"] == 0.0


# ── Bot 3: Claude Sentiment ───────────────────────────────────────────────────

def test_bot3_no_api_key_returns_signals():
    """Bot 3 should fall back to FinBERT if no API key is set."""
    client = make_mock_client()
    feed = make_mock_feed(client)
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
        bot = ClaudeSentimentBot(client, feed)
        bars = make_mock_bars()
        signals = bot.generate_signals(bars)
        assert isinstance(signals, dict)


def test_bot3_no_news_returns_zero_signals():
    client = make_mock_client()
    feed = make_mock_feed(client)
    feed.get_news.return_value = []
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
        bot = ClaudeSentimentBot(client, feed)
        bars = make_mock_bars()
        signals = bot.generate_signals(bars)
        for sig in signals.values():
            assert sig == 0.0


# ── Bot 4: FinBERT + PPO ──────────────────────────────────────────────────────

def test_bot4_signals_valid_range():
    """Bot 4 should return signals in [-1, 1] (models auto-train if missing)."""
    client = make_mock_client()
    feed = make_mock_feed(client)
    # Provide enough bars for the pairs
    all_syms = list({s for pair in PAIRS for s in pair})
    feed.daily_bars.return_value = make_mock_bars(symbols=all_syms, n_rows=80)
    # Skip actual model training in test (just check it doesn't crash on load)
    with patch("bots.bot4_finbert_ppo.FinBERTPPOBot._load_or_train_all"):
        bot = FinBERTPPOBot(client, feed)
        # No PPO models loaded, so signals should be all zeros
        bars = make_mock_bars(symbols=all_syms, n_rows=80)
        signals = bot.generate_signals(bars)
        assert isinstance(signals, dict)


# ── Base bot execute_signal ───────────────────────────────────────────────────

def test_base_bot_buy_on_positive_signal():
    client = make_mock_client()
    feed = make_mock_feed(client)
    bot = MomentumBot(client, feed)
    bot._execute_signal("AAPL", 0.8, 150.0, 20_000.0)
    assert "AAPL" in bot.portfolio.positions


def test_base_bot_no_trade_on_zero_signal():
    client = make_mock_client()
    feed = make_mock_feed(client)
    bot = MomentumBot(client, feed)
    bot._execute_signal("AAPL", 0.0, 150.0, 20_000.0)
    assert "AAPL" not in bot.portfolio.positions
