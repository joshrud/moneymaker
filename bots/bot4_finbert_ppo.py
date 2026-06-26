"""
Bot 4 — FinBERT Sentiment + PPO Pairs Mean Reversion
Strategy: Statistical pairs trading (AAPL/MSFT, NVDA/AMD, AMZN/GOOGL).
PPO agent learns optimal entry/exit thresholds on the spread z-score.
FinBERT filters out pairs where news fundamentally breaks co-integration.
Auto-trains PPO if no model exists.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed, PAIRS
from core.sentiment import finbert_score

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PPO_TRAIN_STEPS = 30_000


class FinBERTPPOBot(BaseBot):
    """
    Pairs mean-reversion bot with PPO policy + FinBERT news filter.
    Each pair gets its own PPO model trained on historical spread data.
    """

    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot4_finbert_ppo", client, feed)
        self.pairs = PAIRS
        self.all_symbols = list({s for pair in PAIRS for s in pair})
        self.ppo_models: dict[str, object] = {}
        self._load_or_train_all()

    def _load_or_train_all(self):
        """Load or train a PPO model per pair."""
        from stable_baselines3 import PPO
        from envs.trading_env import PairsTradingEnv

        for sym_a, sym_b in self.pairs:
            model_path = MODELS_DIR / f"bot4_ppo_{sym_a}_{sym_b}"
            key = f"{sym_a}_{sym_b}"

            if model_path.with_suffix(".zip").exists():
                self.ppo_models[key] = PPO.load(str(model_path))
                self.logger.log("model_loaded", pair=key)
                continue

            spread = self._compute_spread(sym_a, sym_b)
            if spread is None or len(spread) < 50:
                self.logger.log("training_skipped", pair=key, reason="insufficient data")
                continue

            env = PairsTradingEnv(spread)
            model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4,
                        n_steps=512, batch_size=64, gamma=0.99, device="cpu")
            model.learn(total_timesteps=PPO_TRAIN_STEPS)
            MODELS_DIR.mkdir(exist_ok=True)
            model.save(str(model_path))
            self.ppo_models[key] = model
            self.logger.log("training_done", pair=key)

    def _compute_spread(self, sym_a: str, sym_b: str, lookback: int = 504) -> pd.Series | None:
        """OLS regression spread: spread = close_A - beta * close_B."""
        try:
            bars = self.feed.daily_bars([sym_a, sym_b], lookback=lookback)
            closes = bars["close"].unstack(level=0).dropna()
            if sym_a not in closes.columns or sym_b not in closes.columns:
                return None
            slope, intercept, *_ = stats.linregress(closes[sym_b], closes[sym_a])
            spread = closes[sym_a] - slope * closes[sym_b] - intercept
            return spread
        except Exception:
            return None

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        """
        For each pair: compute current spread z-score, use PPO to decide action,
        filter with FinBERT, then translate pair action to per-symbol signals.
        """
        from envs.trading_env import PairsTradingEnv

        # Fetch news for all symbols once
        try:
            news_items = self.feed.get_news(self.all_symbols, limit=20)
        except Exception:
            news_items = []

        # FinBERT scores per symbol
        from collections import defaultdict
        symbol_headlines: dict[str, list[str]] = defaultdict(list)
        for item in news_items:
            if isinstance(item, dict):
                headline = item.get("headline", "")
                syms = item.get("symbols", [])
            else:
                headline = getattr(item, "headline", str(item))
                syms = getattr(item, "symbols", [])
            for sym in syms:
                if sym in self.all_symbols:
                    symbol_headlines[sym].append(headline)

        finbert_scores = {
            sym: finbert_score(symbol_headlines[sym]) if symbol_headlines[sym] else 0.0
            for sym in self.all_symbols
        }

        signals: Dict[str, float] = {sym: 0.0 for sym in self.all_symbols}

        for sym_a, sym_b in self.pairs:
            key = f"{sym_a}_{sym_b}"
            model = self.ppo_models.get(key)
            if model is None:
                continue

            # Skip pair if FinBERT detects strong fundamental divergence
            sent_a = finbert_scores.get(sym_a, 0.0)
            sent_b = finbert_scores.get(sym_b, 0.0)
            if abs(sent_a - sent_b) > 0.6:  # strong divergence = avoid mean reversion
                self.logger.log("pair_skipped", pair=key,
                                reason=f"fundamental divergence {sent_a:.2f} vs {sent_b:.2f}")
                continue

            spread = self._compute_spread(sym_a, sym_b, lookback=60)
            if spread is None or len(spread) < 25:
                continue

            env = PairsTradingEnv(spread)
            obs, _ = env.reset()
            # Fast-forward to last timestep to get current obs
            env._t = max(env.lookback_z, len(spread) - 2)
            obs = env._obs()

            action, _ = model.predict(obs, deterministic=True)

            # Map PPO pair action to per-symbol signals
            # long spread = long sym_a, short sym_b
            # short spread = short sym_a, long sym_b
            if action == 1:    # long spread
                signals[sym_a] = 0.7
                signals[sym_b] = -0.7
            elif action == 2:  # short spread
                signals[sym_a] = -0.7
                signals[sym_b] = 0.7
            elif action == 3:  # exit
                signals[sym_a] = -0.5 if sym_a in self.portfolio.positions else 0.0
                signals[sym_b] = -0.5 if sym_b in self.portfolio.positions else 0.0

            z = float(env._zscore(env._t))
            self.logger.log_signal(sym_a, signals[sym_a],
                                   reason=f"ppo_action={action},z={z:.2f},fb={sent_a:.2f}")
            self.logger.log_signal(sym_b, signals[sym_b],
                                   reason=f"ppo_action={action},z={z:.2f},fb={sent_b:.2f}")

        return signals
