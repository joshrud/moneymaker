"""
Bot 14 — Deep Hedging RL (Options + SAC)
Implements a simplified version of Bühlmann et al. (2019) Deep Hedging paradigm.
Holds synthetic long ATM calls on top universe picks. PPO agent learns optimal
delta hedging by dynamically sizing underlying stock position. Most experimental bot.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed
from core.stock_selector import StockSelector
from core.indicators import rsi

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bot14_deep_hedging"
TOP_N = 5    # hedge positions in top 5 selected stocks
LOOKBACK = 30
TRAIN_STEPS = 75_000
EXPIRY_DAYS = 30


class DeepHedgingBot(BaseBot):
    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot14_deep_hedging", client, feed)
        self.selector = StockSelector(top_n=20, min_per_sector=1)
        self._selected: list[str] = []
        self.model = None
        self._days_since_reset = 0
        self._load_or_train()

    def _load_or_train(self):
        from stable_baselines3 import PPO
        if MODEL_PATH.with_suffix(".zip").exists():
            self.model = PPO.load(str(MODEL_PATH))
            return
        self.logger.log("training_start", steps=TRAIN_STEPS)
        env = self._build_env()
        if env is None:
            return
        self.model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4,
                         n_steps=1024, batch_size=64, gamma=0.99,
                         ent_coef=0.01, device="cpu")
        self.model.learn(total_timesteps=TRAIN_STEPS)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        self.model.save(str(MODEL_PATH))
        self.logger.log("training_done")

    def _build_env(self):
        try:
            from envs.trading_env import DeepHedgingEnv
            bars = self.feed.universe_bars(lookback=252)
            if bars.empty:
                return None
            closes_all = bars["close"].unstack(level=0)
            selected = self.selector.select(closes_all)
            if not selected:
                return None
            close_s = closes_all[selected[0]].dropna()
            return DeepHedgingEnv(close_s.to_frame(), lookback=LOOKBACK, expiry_days=EXPIRY_DAYS)
        except Exception as e:
            self.logger.log("env_build_error", error=str(e))
            return None

    def run_once(self) -> dict:
        bars_df = self.feed.universe_bars(lookback=LOOKBACK * 2)
        prices = self.feed.latest_prices()
        portfolio_val = self.portfolio.mark_to_market(prices)
        if self.risk.check_circuit_breaker(portfolio_val):
            self.logger.log("circuit_breaker", reason="daily loss limit hit")
            return self._end_of_cycle(prices)
        signals = self.generate_signals(bars_df)
        for symbol, signal in signals.items():
            price = prices.get(symbol)
            if price is None:
                continue
            self._execute_signal(symbol, signal, price, portfolio_val)
        self._days_since_reset += 1
        if self._days_since_reset >= EXPIRY_DAYS:
            self._days_since_reset = 0
        return self._end_of_cycle(prices)

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        if self.model is None:
            return {}
        closes_all = bars_df["close"].unstack(level=0) if isinstance(
            bars_df.index, pd.MultiIndex) else bars_df
        self._selected = self.selector.select(closes_all)[:TOP_N]
        if not self._selected:
            return {}

        signals = {}
        for sym in self._selected:
            if sym not in closes_all.columns:
                continue
            try:
                close_s = closes_all[sym].dropna()
                if len(close_s) < LOOKBACK:
                    continue
                vol = float(close_s.pct_change().dropna().iloc[-20:].std() * np.sqrt(252))
                time_left = max(0, (EXPIRY_DAYS - self._days_since_reset) / EXPIRY_DAYS)
                hedge_ratio = self._current_weight(sym)
                recent = np.diff(close_s.values[-6:]) / (close_s.values[-6:-1] + 1e-8)
                if len(recent) < 5:
                    recent = np.pad(recent.astype(np.float32), (5 - len(recent), 0))
                obs = np.array([
                    0.0,         # moneyness (ATM)
                    time_left,
                    0.5,         # approximate delta for ATM
                    hedge_ratio,
                    *recent[:5],
                    float(np.clip(vol, 0, 2)),
                ], dtype=np.float32)
                action, _ = self.model.predict(obs, deterministic=True)
                # Action is hedge fraction [-1, 1]; convert to position signal
                hedge_frac = float(np.clip(action[0] if hasattr(action, '__len__') else action, -1.0, 1.0))
                # Long call = long delta; hedge by going long underlying proportional to delta
                signals[sym] = float(np.clip(hedge_frac * 0.5, -1.0, 1.0))
                self.logger.log_signal(sym, signals[sym],
                                       reason=f"deep_hedge_frac={hedge_frac:.3f},vol={vol:.3f}")
            except Exception:
                continue
        return signals

    def _current_weight(self, symbol: str) -> float:
        pv = self.portfolio.total_value
        if pv <= 0 or symbol not in self.portfolio.positions:
            return 0.0
        pos = self.portfolio.positions[symbol]
        return (pos.qty * pos.avg_cost) / pv


def _build_deep_hedging_env(universe_bars: pd.DataFrame, selector: StockSelector, lookback: int):
    from envs.trading_env import DeepHedgingEnv
    if universe_bars.empty:
        return None
    closes_all = universe_bars["close"].unstack(level=0)
    selected = selector.select(closes_all)
    if not selected:
        return None
    close_s = closes_all[selected[0]].dropna()
    if len(close_s) < lookback + 10:
        return None
    return DeepHedgingEnv(close_s.to_frame(), lookback=lookback, expiry_days=EXPIRY_DAYS)
