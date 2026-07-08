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


class SortinoTradingEnv(gym.Env):
    """
    Multi-asset continuous portfolio management environment — identical to TradingEnv
    except the reward uses Sortino ratio (penalises only downside volatility).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,
        indicators: pd.DataFrame,
        lookback: int = 20,
        starting_capital: float = 20_000.0,
    ):
        super().__init__()
        self.prices = prices.values.astype(np.float32)
        self.indicator_data = indicators.values.astype(np.float32)
        self.lookback = lookback
        self.n_assets = prices.shape[1]
        self.starting_capital = starting_capital

        self.action_space = spaces.Box(
            low=-4.0, high=4.0, shape=(self.n_assets + 1,), dtype=np.float32
        )
        obs_dim = (
            (lookback - 1) * self.n_assets
            + 2 * self.n_assets
            + self.n_assets + 1
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._step = 0
        self._weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self._weights[-1] = 1.0
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
        new_weights = self._softmax(action)
        turnover = np.sum(np.abs(new_weights - self._weights))
        tc_penalty = turnover * TRANSACTION_COST_BPS / 10_000

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

        # Sortino reward: divide by downside std only
        if len(self._returns_buffer) >= 5:
            r = np.array(self._returns_buffer[-20:])
            downside = r[r < 0]
            downside_std = float(downside.std()) if len(downside) > 1 else 1e-8
            reward = float(r.mean() / (downside_std + 1e-8))
        else:
            reward = portfolio_return

        done = self._step >= len(self.prices) - 1
        return self._obs(), reward, done, False, {}

    def _obs(self) -> np.ndarray:
        t = self._step
        window_prices = self.prices[max(0, t - self.lookback): t]
        if len(window_prices) < self.lookback:
            pad = np.zeros((self.lookback - len(window_prices), self.n_assets), dtype=np.float32)
            window_prices = np.vstack([pad, window_prices])
        log_rets = np.diff(np.log(window_prices + 1e-8), axis=0)
        if log_rets.shape[0] < self.lookback - 1:
            pad = np.zeros((self.lookback - 1 - log_rets.shape[0], self.n_assets), dtype=np.float32)
            log_rets = np.vstack([pad, log_rets])
        ind_slice = self.indicator_data[t] if t < len(self.indicator_data) else np.zeros(self.indicator_data.shape[1])
        return np.concatenate([
            log_rets.flatten(),
            ind_slice[:2 * self.n_assets],
            self._weights,
        ]).astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()


class VWAPTradingEnv(gym.Env):
    """Single-asset VWAP mean-reversion environment for DQN.

    State: [price_vs_vwap_zscore, rsi_norm, position (-1/0/1), bars_held_norm, recent_returns_5]  → shape (9,)
    Action: 0=hold, 1=strong_buy, 2=buy, 3=sell, 4=strong_sell  (5 discrete)
    Reward: PnL minus transaction cost, with mean-reversion bonus when closing toward VWAP.
    """

    metadata = {"render_modes": []}

    def __init__(self, prices: pd.DataFrame, indicators: pd.DataFrame, lookback: int = 20):
        super().__init__()
        self.prices = prices.values.astype(np.float32).flatten()
        self.rsi = indicators["rsi"].values.astype(np.float32) / 100.0
        # Approximate VWAP with 5-day rolling typical price if not provided
        self.vwap = indicators.get("vwap", indicators.get("ema20", pd.Series(self.prices))).values.astype(np.float32)
        self.lookback = lookback

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(9,), dtype=np.float32)

        self._t = lookback
        self._position = 0
        self._entry_price = 0.0
        self._bars_held = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.lookback
        self._position = 0
        self._entry_price = 0.0
        self._bars_held = 0
        return self._obs(), {}

    def step(self, action: int):
        t = self._t
        if t >= len(self.prices) - 1:
            return self._obs(), 0.0, True, False, {}

        price = float(self.prices[t])
        next_price = float(self.prices[t + 1])
        tc = TRANSACTION_COST_BPS / 10_000
        reward = 0.0

        # Map actions to position changes
        target_pos = {0: self._position, 1: 1, 2: 1, 3: -1 if self._position > 0 else 0, 4: 0}[action]

        if target_pos != self._position:
            reward -= tc
            if self._position != 0:
                reward += self._position * (price - self._entry_price) / (self._entry_price + 1e-8)
            self._position = target_pos
            self._entry_price = price
            self._bars_held = 0
        elif self._position != 0:
            reward += self._position * (next_price - price) / (price + 1e-8)
            self._bars_held += 1

        self._t += 1
        done = self._t >= len(self.prices) - 1
        return self._obs(), float(reward), done, False, {}

    def _obs(self) -> np.ndarray:
        t = self._t
        price = float(self.prices[t])
        vwap = float(self.vwap[t]) if t < len(self.vwap) else price
        vwap_std = float(np.std(self.prices[max(0, t - 20):t + 1])) + 1e-8
        vwap_z = (price - vwap) / vwap_std
        rsi_v = float(self.rsi[t]) if t < len(self.rsi) else 0.5
        recent = np.diff(self.prices[max(0, t - 5):t + 1]) / (self.prices[max(0, t - 5):t] + 1e-8)
        if len(recent) < 5:
            recent = np.pad(recent, (5 - len(recent), 0))
        return np.array([
            np.clip(vwap_z, -5, 5),
            rsi_v,
            float(self._position),
            min(self._bars_held / 20.0, 1.0),
            *recent[:5],
        ], dtype=np.float32)


class DeepHedgingEnv(gym.Env):
    """
    Simplified Deep Hedging environment (Bühlmann et al. 2019).

    Agent holds a long ATM call option and must hedge the delta exposure
    by trading the underlying stock. Learns optimal hedge ratio.

    State: [moneyness, time_to_expiry_norm, current_delta_approx,
            hedge_ratio, recent_stock_returns_5, implied_vol_proxy] → shape (10,)
    Action: hedge_fraction ∈ [-1, 1] — fraction of theoretical delta to hedge
    Reward: -(hedging_pnl_variance) — minimise P&L variance (CVaR proxy)
    """

    metadata = {"render_modes": []}

    def __init__(self, prices: pd.DataFrame, lookback: int = 30, expiry_days: int = 30):
        super().__init__()
        self.prices = prices.values.astype(np.float32).flatten()
        self.lookback = lookback
        self.expiry_days = expiry_days

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self._t = lookback
        self._hedge_ratio = 0.5
        self._episode_pnls: list[float] = []
        self._days_elapsed = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.lookback
        self._hedge_ratio = 0.5
        self._episode_pnls = []
        self._days_elapsed = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        t = self._t
        if t >= len(self.prices) - 1:
            return self._obs(), 0.0, True, False, {}

        new_hedge = float(np.clip(action[0], -1.0, 1.0))
        hedge_change = abs(new_hedge - self._hedge_ratio)
        tc = hedge_change * TRANSACTION_COST_BPS / 10_000

        stock_ret = (float(self.prices[t + 1]) - float(self.prices[t])) / (float(self.prices[t]) + 1e-8)

        vol = float(np.std(self.prices[max(0, t - 20):t + 1]) / (float(self.prices[t]) + 1e-8)) * np.sqrt(252)
        time_left = max(0, (self.expiry_days - self._days_elapsed) / 252)
        moneyness = 0.0  # ATM assumed
        theoretical_delta = 0.5 + moneyness / (vol * np.sqrt(time_left + 1e-8) + 1e-8)
        theoretical_delta = float(np.clip(theoretical_delta, 0.0, 1.0))

        option_pnl = theoretical_delta * stock_ret
        hedge_pnl = -new_hedge * theoretical_delta * stock_ret
        net_pnl = option_pnl + hedge_pnl - tc

        self._episode_pnls.append(net_pnl)
        self._hedge_ratio = new_hedge
        self._days_elapsed += 1
        self._t += 1

        if len(self._episode_pnls) >= 5:
            recent_pnls = np.array(self._episode_pnls[-20:])
            reward = -float(np.var(recent_pnls)) + float(np.mean(recent_pnls)) * 0.1
        else:
            reward = net_pnl

        done = self._t >= len(self.prices) - 1 or self._days_elapsed >= self.expiry_days
        return self._obs(), float(reward), done, False, {}

    def _obs(self) -> np.ndarray:
        t = self._t
        price = float(self.prices[t])
        vol = float(np.std(self.prices[max(0, t - 20):t + 1]) / (price + 1e-8)) * np.sqrt(252)
        time_left = max(0, (self.expiry_days - self._days_elapsed) / self.expiry_days)
        recent = np.diff(self.prices[max(0, t - 5):t + 1]) / (self.prices[max(0, t - 5):t] + 1e-8)
        if len(recent) < 5:
            recent = np.pad(recent.astype(np.float32), (5 - len(recent), 0))
        return np.array([
            0.0,          # moneyness (ATM)
            time_left,
            0.5 + 0.0,    # approximate delta for ATM
            self._hedge_ratio,
            *recent[:5],
            np.clip(vol, 0, 2),
        ], dtype=np.float32)
