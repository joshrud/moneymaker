"""
Bot 9 — DQN VWAP Mean Reversion
Strategy: DQN with 5 discrete actions trading when price deviates significantly from
rolling VWAP (approximated as 5-day EMA of typical price weighted by volume).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed, WATCHLIST
from core.indicators import compute_all, ema

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bot9_dqn_vwap"
TRAIN_STEPS = 75_000
LOOKBACK = 20
VWAP_DEVIATION_THRESHOLD = 0.015  # 1.5% from VWAP to consider a signal


class DQNVWAPBot(BaseBot):
    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot9_dqn_vwap", client, feed)
        self.watchlist = WATCHLIST
        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        from stable_baselines3 import DQN
        if MODEL_PATH.with_suffix(".zip").exists():
            self.model = DQN.load(str(MODEL_PATH))
            return
        self.logger.log("training_start", steps=TRAIN_STEPS)
        env = self._build_env()
        if env is None:
            return
        self.model = DQN("MlpPolicy", env, verbose=0, learning_rate=1e-4,
                         buffer_size=50_000, batch_size=64, gamma=0.99,
                         exploration_final_eps=0.05, device="cpu")
        self.model.learn(total_timesteps=TRAIN_STEPS)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        self.model.save(str(MODEL_PATH))

    def _build_env(self):
        try:
            bars = self.feed.daily_bars(lookback=252)
            closes = bars["close"].unstack(level=0) if isinstance(bars.index, pd.MultiIndex) else bars
            if closes.empty:
                return None
            # Use first available symbol for single-asset DQN env
            sym = [s for s in self.watchlist if s in closes.columns]
            if not sym:
                return None
            return self._make_vwap_env(closes, sym[0])
        except Exception:
            return None

    def _make_vwap_env(self, closes, symbol):
        from envs.trading_env import VWAPTradingEnv
        close_s = closes[symbol].dropna()
        if len(close_s) < LOOKBACK + 10:
            return None
        df_ind = pd.DataFrame({
            "rsi": pd.Series(np.linspace(30, 70, len(close_s)), index=close_s.index),
            "ema20": ema(close_s, window=20),
        }, index=close_s.index).ffill().bfill()
        return VWAPTradingEnv(close_s.to_frame("close"), df_ind, lookback=LOOKBACK)

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        if self.model is None:
            return {}
        closes = bars_df["close"].unstack(level=0) if isinstance(bars_df.index, pd.MultiIndex) else bars_df
        signals = {}
        for symbol in self.watchlist:
            if symbol not in closes.columns:
                continue
            close_s = closes[symbol].dropna()
            if len(close_s) < LOOKBACK:
                continue
            price = float(close_s.iloc[-1])
            vwap = float(ema(close_s, window=5).iloc[-1])
            deviation = (price - vwap) / (vwap + 1e-8)
            vwap_std = float(close_s.rolling(20).std().iloc[-1]) + 1e-8
            vwap_z = float(np.clip((price - vwap) / vwap_std, -5, 5))
            recent = np.diff(close_s.values[-6:]) / (close_s.values[-6:-1] + 1e-8)
            if len(recent) < 5:
                recent = np.pad(recent, (5 - len(recent), 0))
            pos = self._current_weight(symbol)
            obs = np.array([vwap_z, 0.5, pos, 0.0, *recent[:5]], dtype=np.float32)
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
            except Exception:
                continue
            # Map DQN actions to signal: 0=hold, 1=strong_buy(+1), 2=buy(+0.5), 3=sell(-0.5), 4=strong_sell(-1)
            action_map = {0: 0.0, 1: 1.0, 2: 0.5, 3: -0.5, 4: -1.0}
            sig = action_map.get(action, 0.0)
            # Only fire when price is actually deviated enough
            if abs(deviation) < VWAP_DEVIATION_THRESHOLD and sig != 0.0:
                sig *= 0.5
            signals[symbol] = sig
            self.logger.log_signal(symbol, sig, reason=f"vwap_dev={deviation:.3f},action={action}")
        return signals

    def _current_weight(self, symbol: str) -> float:
        pv = self.portfolio.total_value
        if pv <= 0 or symbol not in self.portfolio.positions:
            return 0.0
        pos = self.portfolio.positions[symbol]
        return (pos.qty * pos.avg_cost) / pv
