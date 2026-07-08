"""
Bot 7 — TD3 Expanded Pairs Stat Arb
Strategy: TD3 (more stable than PPO/DDPG) trading spread between 10 co-integrated pairs.
Continuous position sizing via spread z-score. Pairs include financials, energy, consumer staples.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

PAIRS = [
    ("AAPL", "MSFT"), ("NVDA", "AMD"), ("AMZN", "GOOGL"),
    ("JPM", "BAC"), ("XOM", "CVX"), ("PEP", "KO"),
    ("GS", "MS"), ("V", "MA"), ("HD", "TGT"), ("META", "NFLX"),
]


class TD3PairsBot(BaseBot):
    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot7_td3_pairs", client, feed)
        self.pairs = PAIRS
        self.models: dict = {}
        self._load_models()

    def _load_models(self):
        from stable_baselines3 import TD3
        for sym_a, sym_b in self.pairs:
            key = f"{sym_a}_{sym_b}"
            path = MODEL_DIR / f"bot7_td3_{key}"
            if path.with_suffix(".zip").exists():
                try:
                    self.models[key] = TD3.load(str(path))
                except Exception:
                    pass

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        from scipy import stats
        signals: Dict[str, float] = {}
        closes_all = bars_df["close"].unstack(level=0) if isinstance(bars_df.index, pd.MultiIndex) else bars_df

        for sym_a, sym_b in self.pairs:
            key = f"{sym_a}_{sym_b}"
            model = self.models.get(key)
            if model is None:
                continue
            if sym_a not in closes_all.columns or sym_b not in closes_all.columns:
                continue
            ca = closes_all[sym_a].dropna()
            cb = closes_all[sym_b].dropna()
            idx = ca.index.intersection(cb.index)
            if len(idx) < 30:
                continue
            ca, cb = ca[idx], cb[idx]
            slope, intercept, *_ = stats.linregress(cb.values, ca.values)
            spread = (ca - slope * cb - intercept)
            spread_z = float((spread.iloc[-1] - spread.mean()) / (spread.std() + 1e-8))
            recent_z = np.array([
                float((spread.iloc[-i] - spread.mean()) / (spread.std() + 1e-8))
                for i in range(1, 6)
            ], dtype=np.float32)
            pos_a = self._current_weight(sym_a)
            pos_b = self._current_weight(sym_b)
            obs = np.array([spread_z, pos_a, pos_b, *recent_z], dtype=np.float32)
            try:
                action, _ = model.predict(obs, deterministic=True)
                action_val = float(np.clip(action[0], -1.0, 1.0))
            except Exception:
                continue
            # positive action = long spread (long A, short B); negative = short spread
            signals[sym_a] = float(np.clip(action_val, -1.0, 1.0))
            signals[sym_b] = float(np.clip(-action_val * 0.8, -1.0, 1.0))
            self.logger.log_signal(sym_a, signals[sym_a], reason=f"td3_pairs,z={spread_z:.2f}")
            self.logger.log_signal(sym_b, signals[sym_b], reason=f"td3_pairs,z={-spread_z:.2f}")
        return signals

    def _current_weight(self, symbol: str) -> float:
        pv = self.portfolio.total_value
        if pv <= 0 or symbol not in self.portfolio.positions:
            return 0.0
        pos = self.portfolio.positions[symbol]
        return (pos.qty * pos.avg_cost) / pv
