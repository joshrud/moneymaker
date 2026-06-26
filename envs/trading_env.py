"""
OpenAI Gymnasium-compatible trading environments for RL training.

TradingEnv      — multi-asset portfolio management (SAC, continuous weights)
PairsTradingEnv — pairs mean-reversion (PPO, discrete entry/exit)
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

TRANSACTION_COST_BPS = 5  # 5 bps slippage per trade


class TradingEnv(gym.Env):
    """
    Multi-asset continuous portfolio management environment for SAC.

    State:  [lookback window of log returns per asset (lookback×n_assets),
             RSI normalised per asset (n_assets),
             MACD histogram normalised per asset (n_assets),
             current portfolio weights (n_assets + 1 cash)]
    Action: portfolio weight vector (n_assets + 1), softmax applied internally.
    Reward: incremental Sharpe contribution minus transaction cost penalty.
    Episode: one full pass through the price history (walk-forward split).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,       # (dates × symbols) closing prices
        indicators: pd.DataFrame,   # (dates × [sym_rsi, sym_macd_hist, ...])
        lookback: int = 20,
        starting_capital: float = 20_000.0,
    ):
        super().__init__()
        self.prices = prices.values.astype(np.float32)
        self.indicator_data = indicators.values.astype(np.float32)
        self.lookback = lookback
        self.n_assets = prices.shape[1]
        self.starting_capital = starting_capital

        # Action: pre-softmax logits per asset + cash. SAC requires finite bounds;
        # [-4, 4] covers the full useful range since softmax(4) ≈ 0.98 weight.
        self.action_space = spaces.Box(
            low=-4.0, high=4.0, shape=(self.n_assets + 1,), dtype=np.float32
        )
        # Observation dimension
        # np.diff reduces the lookback window by 1 row, so returns = (lookback-1) * n_assets
        obs_dim = (
            (lookback - 1) * self.n_assets  # rolling log returns
            + 2 * self.n_assets             # RSI + MACD hist per asset
            + self.n_assets + 1             # current weights
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._step = 0
        self._weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self._weights[-1] = 1.0  # start fully in cash
        self._portfolio_value = starting_capital
        self._returns_buffer: list[float] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = self.lookback
        self._weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self._weights[-1] = 1.0
        self._portfolio_value = self.starting_capital
        self._returns_buffer = []
        return self._obs(), {}

    def step(self, action: np.ndarray):
        # Softmax to get valid portfolio weights (sum to 1, non-negative)
        new_weights = self._softmax(action)
        turnover = np.sum(np.abs(new_weights - self._weights))
        tc_penalty = turnover * TRANSACTION_COST_BPS / 10_000

        # Compute period return
        t = self._step
        if t >= len(self.prices) - 1:
            return self._obs(), 0.0, True, False, {}

        prev_prices = self.prices[t - 1]
        curr_prices = self.prices[t]
        asset_returns = (curr_prices - prev_prices) / (prev_prices + 1e-8)

        portfolio_return = float(np.dot(new_weights[:-1], asset_returns)) - tc_penalty
        self._portfolio_value *= (1 + portfolio_return)
        self._weights = new_weights
        self._returns_buffer.append(portfolio_return)
        self._step += 1

        # Reward: incremental Sharpe-like signal (return / rolling std)
        if len(self._returns_buffer) >= 5:
            r = np.array(self._returns_buffer[-20:])
            reward = float(r.mean() / (r.std() + 1e-8))
        else:
            reward = portfolio_return

        done = self._step >= len(self.prices) - 1
        return self._obs(), reward, done, False, {}

    def _obs(self) -> np.ndarray:
        t = self._step
        # Rolling log returns window
        window_prices = self.prices[max(0, t - self.lookback): t]
        if len(window_prices) < self.lookback:
            pad = np.zeros((self.lookback - len(window_prices), self.n_assets), dtype=np.float32)
            window_prices = np.vstack([pad, window_prices])
        log_rets = np.diff(np.log(window_prices + 1e-8), axis=0)
        if log_rets.shape[0] < self.lookback - 1:
            pad = np.zeros((self.lookback - 1 - log_rets.shape[0], self.n_assets), dtype=np.float32)
            log_rets = np.vstack([pad, log_rets])

        # Indicator slice for current step
        ind_slice = self.indicator_data[t] if t < len(self.indicator_data) else np.zeros(self.indicator_data.shape[1])

        return np.concatenate([
            log_rets.flatten(),
            ind_slice[:2 * self.n_assets],  # RSI + MACD columns
            self._weights,
        ]).astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()


class PairsTradingEnv(gym.Env):
    """
    Pairs mean-reversion environment for PPO.

    Trades the spread between two co-integrated assets.
    State:  [spread_z_score, current_position (-1/0/1), bars_in_trade, recent_spread_changes(5)]
    Action: 0=hold, 1=enter_long_spread, 2=enter_short_spread, 3=exit
    Reward: PnL from spread reversion minus transaction costs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        spread: pd.Series,     # pre-computed normalised spread series
        lookback_z: int = 20,  # rolling window for z-score calculation
        entry_z: float = 2.0,  # z-score threshold to consider entering
    ):
        super().__init__()
        self.spread_raw = spread.values.astype(np.float32)
        self.lookback_z = lookback_z
        self.entry_z = entry_z

        self.action_space = spaces.Discrete(4)  # hold, long, short, exit
        # [z_score, position, bars_in_trade_norm, 5 recent spread changes]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32)

        self._t = 0
        self._position = 0   # -1, 0, +1
        self._entry_spread = 0.0
        self._bars_in_trade = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.lookback_z
        self._position = 0
        self._entry_spread = 0.0
        self._bars_in_trade = 0
        return self._obs(), {}

    def step(self, action: int):
        t = self._t
        if t >= len(self.spread_raw) - 1:
            return self._obs(), 0.0, True, False, {}

        z = self._zscore(t)
        current_spread = float(self.spread_raw[t])
        next_spread = float(self.spread_raw[t + 1])
        reward = 0.0
        tc = TRANSACTION_COST_BPS / 10_000

        if action == 1 and self._position == 0:  # enter long spread
            self._position = 1
            self._entry_spread = current_spread
            self._bars_in_trade = 0
            reward = -tc
        elif action == 2 and self._position == 0:  # enter short spread
            self._position = -1
            self._entry_spread = current_spread
            self._bars_in_trade = 0
            reward = -tc
        elif action == 3 and self._position != 0:  # exit
            reward = self._position * (current_spread - self._entry_spread) - tc
            self._position = 0
            self._bars_in_trade = 0
        elif self._position != 0:  # hold — unrealised PnL delta
            reward = self._position * (next_spread - current_spread)
            self._bars_in_trade += 1
            # Penalise very long trades (mean reversion should be quick)
            if self._bars_in_trade > 20:
                reward -= 0.001 * self._bars_in_trade

        self._t += 1
        done = self._t >= len(self.spread_raw) - 1
        return self._obs(), float(reward), done, False, {}

    def _zscore(self, t: int) -> float:
        window = self.spread_raw[max(0, t - self.lookback_z): t + 1]
        if window.std() < 1e-8:
            return 0.0
        return float((window[-1] - window.mean()) / window.std())

    def _obs(self) -> np.ndarray:
        t = self._t
        z = self._zscore(t)
        recent = np.diff(self.spread_raw[max(0, t - 5): t + 1])
        if len(recent) < 5:
            recent = np.pad(recent, (5 - len(recent), 0))
        return np.array([
            z, float(self._position), min(self._bars_in_trade / 20.0, 1.0),
            *recent[:5]
        ], dtype=np.float32)
