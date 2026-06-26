"""
Bot 2 — SAC Reinforcement Learning Portfolio Manager
Algorithm: Soft Actor-Critic (off-policy, continuous actions) via stable-baselines3.
Action: portfolio weights across N assets (softmax normalised).
Reward: incremental Sharpe ratio minus transaction cost penalty.
Model is trained offline on 2 years of historical data; loaded at runtime.
Auto-trains if no saved model is found in models/.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed, WATCHLIST
from core.indicators import rsi, macd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bot2_sac"
SAC_ASSETS = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]  # 5-asset portfolio
LOOKBACK = 20
TRAIN_STEPS = 50_000


class SACBot(BaseBot):
    """
    SAC-based continuous portfolio manager.
    Allocates capital across SAC_ASSETS using a trained SAC policy.
    """

    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot2_sac_rl", client, feed)
        self.assets = SAC_ASSETS
        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        """Load existing model or train from scratch on historical data."""
        from stable_baselines3 import SAC

        if MODEL_PATH.with_suffix(".zip").exists():
            self.model = SAC.load(str(MODEL_PATH))
            self.logger.log("model_loaded", path=str(MODEL_PATH))
            return

        self.logger.log("training_start", steps=TRAIN_STEPS)
        env = self._build_env()
        if env is None:
            self.logger.log("training_skipped", reason="insufficient historical data")
            return

        self.model = SAC("MlpPolicy", env, verbose=0, learning_rate=3e-4,
                         buffer_size=10_000, batch_size=256, gamma=0.99,
                         ent_coef="auto", device="cpu")
        self.model.learn(total_timesteps=TRAIN_STEPS)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        self.model.save(str(MODEL_PATH))
        self.logger.log("training_done", path=str(MODEL_PATH))

    def _build_env(self):
        """Constructs a TradingEnv from 2 years of historical data."""
        from envs.trading_env import TradingEnv
        try:
            bars = self.feed.daily_bars(self.assets, lookback=504)  # ~2 years
            closes = bars["close"].unstack(level=0).reindex(columns=self.assets).dropna()
            if len(closes) < LOOKBACK + 10:
                return None

            # Build indicator matrix: RSI and MACD hist for each asset
            ind_cols = {}
            for sym in self.assets:
                close_s = closes[sym]
                ind_cols[f"{sym}_rsi"] = rsi(close_s).fillna(50) / 100.0  # normalise to [0,1]
                mdf = macd(close_s)
                # Normalise MACD histogram by price scale
                ind_cols[f"{sym}_macd"] = (mdf["histogram"] / closes[sym].mean()).fillna(0)
            ind_df = pd.DataFrame(ind_cols, index=closes.index).fillna(0)

            return TradingEnv(closes, ind_df, lookback=LOOKBACK)
        except Exception as e:
            self.logger.log("env_build_error", error=str(e))
            return None

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        """Uses the SAC policy to compute portfolio weights, returns per-asset signals."""
        if self.model is None:
            return {sym: 0.0 for sym in self.assets}

        obs = self._build_observation(bars_df)
        if obs is None:
            return {sym: 0.0 for sym in self.assets}

        action, _ = self.model.predict(obs, deterministic=True)
        # Convert action to portfolio weights via softmax
        exp_a = np.exp(action - action.max())
        weights = exp_a / exp_a.sum()

        # weights[-1] is cash allocation; first N are assets
        signals = {}
        for i, sym in enumerate(self.assets):
            target_weight = float(weights[i])
            current_weight = self._current_weight(sym)
            # Signal = difference from current allocation, capped to [-1, +1]
            signals[sym] = np.clip(target_weight - current_weight, -1.0, 1.0)
            self.logger.log_signal(sym, signals[sym],
                                   reason=f"sac_weight={target_weight:.3f}")

        return signals

    def _build_observation(self, bars_df: pd.DataFrame):
        """Constructs the observation vector matching the training env."""
        try:
            closes = bars_df["close"].unstack(level=0).reindex(columns=self.assets).dropna()
            if len(closes) < LOOKBACK:
                return None

            window = closes.iloc[-LOOKBACK:].values.astype(np.float32)
            log_rets = np.diff(np.log(window + 1e-8), axis=0).flatten()

            ind_parts = []
            for sym in self.assets:
                close_s = closes[sym]
                r = float(rsi(close_s).iloc[-1]) / 100.0 if len(close_s) >= 14 else 0.5
                m = macd(close_s)["histogram"].iloc[-1]
                m_norm = float(m / (close_s.mean() + 1e-8))
                ind_parts.extend([r, m_norm])

            # Current weights
            current_weights = np.array([self._current_weight(s) for s in self.assets] + [0.0],
                                        dtype=np.float32)
            obs = np.concatenate([log_rets, ind_parts, current_weights]).astype(np.float32)
            return obs
        except Exception:
            return None

    def _current_weight(self, symbol: str) -> float:
        """Returns symbol's current weight in virtual portfolio."""
        pv = self.portfolio.total_value
        if pv <= 0 or symbol not in self.portfolio.positions:
            return 0.0
        pos = self.portfolio.positions[symbol]
        return (pos.qty * pos.avg_cost) / pv
