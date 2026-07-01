"""
Bot 2 — Two-Stage SAC Portfolio Manager

Stage 1 (StockSelector): scores all 261 universe stocks on momentum + volatility
  factors and picks the top TOP_N with at least one per GICS sector.

Stage 2 (SAC): allocates capital across the selected TOP_N stocks using a
  trained Soft Actor-Critic policy. Action/obs dimensions are fixed at TOP_N,
  so the model is retrained whenever TOP_N changes.

The model is trained offline via scripts/train_rl_models.py. It auto-trains
from scratch if no saved model is found.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed
from core.indicators import rsi, macd
from core.stock_selector import StockSelector

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bot2_sac"
TOP_N = 20       # number of stocks SAC allocates across
LOOKBACK = 20    # observation window (trading days)
TRAIN_STEPS = 50_000


class SACBot(BaseBot):
    """
    Two-stage portfolio manager: factor-select TOP_N stocks, then SAC allocates.
    """

    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot2_sac_rl", client, feed)
        self.selector = StockSelector(top_n=TOP_N, min_per_sector=1)
        self._selected: list[str] = []   # refreshed each run_once()
        self.model = None
        self._load_or_train()

    # ── model lifecycle ────────────────────────────────────────────────────────

    def _load_or_train(self):
        from stable_baselines3 import SAC

        if MODEL_PATH.with_suffix(".zip").exists():
            self.model = SAC.load(str(MODEL_PATH))
            self.logger.log("model_loaded", path=str(MODEL_PATH))
            return

        self.logger.log("training_start", steps=TRAIN_STEPS)
        env = self._build_env_from_feed()
        if env is None:
            self.logger.log("training_skipped", reason="insufficient data")
            return

        self.model = SAC(
            "MlpPolicy", env, verbose=0, learning_rate=3e-4,
            buffer_size=10_000, batch_size=256, gamma=0.99,
            ent_coef="auto", device="cpu",
        )
        self.model.learn(total_timesteps=TRAIN_STEPS)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        self.model.save(str(MODEL_PATH))
        self.logger.log("training_done", path=str(MODEL_PATH))

    def _build_env_from_feed(self):
        """Fetches universe bars, selects top stocks, returns a TradingEnv."""
        try:
            bars = self.feed.universe_bars(lookback=504)
            return _build_env(bars, self.selector, LOOKBACK)
        except Exception as e:
            self.logger.log("env_build_error", error=str(e))
            return None

    def run_once(self) -> dict:
        """Overrides base: fetches universe bars (not just WATCHLIST) before signalling."""
        from core.data_feed import DataFeed
        bars_df = self.feed.universe_bars(lookback=LOOKBACK * 3)
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
        return self._end_of_cycle(prices)

    # ── signal generation ──────────────────────────────────────────────────────

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        if self.model is None:
            return {}

        # Stage 1: select today's top-N stocks from universe bars
        closes_all = bars_df["close"].unstack(level=0) if isinstance(
            bars_df.index, pd.MultiIndex) else bars_df
        self._selected = self.selector.select(closes_all)
        if not self._selected:
            return {}

        # Stage 2: build obs and run SAC policy
        obs = self._build_observation(closes_all, self._selected)
        if obs is None:
            return {sym: 0.0 for sym in self._selected}

        action, _ = self.model.predict(obs, deterministic=True)
        exp_a = np.exp(action - action.max())
        weights = exp_a / exp_a.sum()  # softmax → portfolio weights

        signals = {}
        for i, sym in enumerate(self._selected):
            target_w = float(weights[i])
            current_w = self._current_weight(sym)
            signals[sym] = float(np.clip(target_w - current_w, -1.0, 1.0))
            self.logger.log_signal(sym, signals[sym],
                                   reason=f"sac_weight={target_w:.3f}")
        return signals

    def _build_observation(self, closes: pd.DataFrame, symbols: list[str]):
        try:
            sub = closes.reindex(columns=symbols).dropna(axis=1)
            if len(sub) < LOOKBACK or sub.shape[1] < len(symbols):
                return None

            window = sub.iloc[-LOOKBACK:].values.astype(np.float32)
            log_rets = np.diff(np.log(window + 1e-8), axis=0).flatten()

            ind_parts = []
            for sym in symbols:
                close_s = sub[sym]
                r = float(rsi(close_s).iloc[-1]) / 100.0 if len(close_s) >= 14 else 0.5
                m_hist = macd(close_s)["histogram"].iloc[-1]
                ind_parts.extend([r, float(m_hist / (close_s.mean() + 1e-8))])

            current_weights = np.array(
                [self._current_weight(s) for s in symbols] + [0.0], dtype=np.float32
            )
            return np.concatenate([log_rets, ind_parts, current_weights]).astype(np.float32)
        except Exception:
            return None

    def _current_weight(self, symbol: str) -> float:
        pv = self.portfolio.total_value
        if pv <= 0 or symbol not in self.portfolio.positions:
            return 0.0
        pos = self.portfolio.positions[symbol]
        return (pos.qty * pos.avg_cost) / pv


# ── shared helper used by both SACBot and train_rl_models.py ──────────────────

def _build_env(universe_bars: pd.DataFrame, selector: StockSelector, lookback: int):
    """
    Given raw universe bars, runs stock selection and returns a TradingEnv
    over the selected stocks. Returns None if data is insufficient.
    """
    from envs.trading_env import TradingEnv

    if universe_bars.empty:
        return None

    closes_all = universe_bars["close"].unstack(level=0)
    selected = selector.select(closes_all)
    if not selected:
        return None

    closes = closes_all.reindex(columns=selected).dropna()
    if len(closes) < lookback + 10:
        return None

    ind_cols = {}
    for sym in selected:
        close_s = closes[sym]
        ind_cols[f"{sym}_rsi"] = rsi(close_s).fillna(50) / 100.0
        mdf = macd(close_s)
        ind_cols[f"{sym}_macd"] = (mdf["histogram"] / (close_s.mean() + 1e-8)).fillna(0)
    ind_df = pd.DataFrame(ind_cols, index=closes.index).fillna(0)

    return TradingEnv(closes, ind_df, lookback=lookback)
