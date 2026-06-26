"""Tests for Gym trading environments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from envs.trading_env import TradingEnv, PairsTradingEnv


@pytest.fixture
def sample_prices_df():
    np.random.seed(0)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    assets = ["AAPL", "MSFT", "NVDA"]
    data = {}
    for sym in assets:
        data[sym] = 100.0 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_indicators_df(sample_prices_df):
    """Simple indicator df: RSI and MACD for each asset."""
    cols = {}
    for sym in sample_prices_df.columns:
        cols[f"{sym}_rsi"] = np.full(len(sample_prices_df), 0.5)
        cols[f"{sym}_macd"] = np.zeros(len(sample_prices_df))
    return pd.DataFrame(cols, index=sample_prices_df.index)


@pytest.fixture
def trading_env(sample_prices_df, sample_indicators_df):
    return TradingEnv(sample_prices_df, sample_indicators_df, lookback=10)


@pytest.fixture
def spread_series():
    np.random.seed(1)
    spread = np.cumsum(np.random.randn(200) * 0.1)
    return pd.Series(spread)


@pytest.fixture
def pairs_env(spread_series):
    return PairsTradingEnv(spread_series, lookback_z=10)


# ── TradingEnv tests ──────────────────────────────────────────────────────────

def test_trading_env_reset_returns_valid_obs(trading_env):
    obs, info = trading_env.reset()
    assert obs.shape == trading_env.observation_space.shape
    assert not np.any(np.isnan(obs))


def test_trading_env_step_returns_correct_types(trading_env):
    trading_env.reset()
    action = trading_env.action_space.sample()
    obs, reward, done, truncated, info = trading_env.step(action)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(done, bool)
    assert obs.shape == trading_env.observation_space.shape


def test_trading_env_softmax_sums_to_one(trading_env):
    trading_env.reset()
    action = np.array([1.0, 2.0, -1.0, 0.5])
    weights = TradingEnv._softmax(action)
    assert abs(weights.sum() - 1.0) < 1e-6
    assert (weights >= 0).all()


def test_trading_env_episode_completes(trading_env):
    trading_env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        action = trading_env.action_space.sample()
        _, _, done, _, _ = trading_env.step(action)
        steps += 1
    assert done or steps == 500


# ── PairsTradingEnv tests ─────────────────────────────────────────────────────

def test_pairs_env_reset_returns_valid_obs(pairs_env):
    obs, info = pairs_env.reset()
    assert obs.shape == pairs_env.observation_space.shape
    assert not np.any(np.isnan(obs))


def test_pairs_env_action_space(pairs_env):
    assert pairs_env.action_space.n == 4


def test_pairs_env_hold_action(pairs_env):
    pairs_env.reset()
    obs, reward, done, _, _ = pairs_env.step(0)  # hold
    assert isinstance(reward, (float, np.floating))


def test_pairs_env_enter_and_exit(pairs_env):
    pairs_env.reset()
    pairs_env.step(1)  # enter long
    assert pairs_env._position == 1
    pairs_env.step(3)  # exit
    assert pairs_env._position == 0


def test_pairs_env_zscore_finite(pairs_env):
    pairs_env.reset()
    z = pairs_env._zscore(pairs_env._t)
    assert np.isfinite(z)
