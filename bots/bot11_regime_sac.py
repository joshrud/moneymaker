"""
Bot 11 — Regime-Switching SAC
Two SAC models: one optimised on trending market conditions (ADX > 25),
one on ranging conditions (ADX < 20). A regime detector routes to the right model.
Addresses the #1 RL pitfall: non-stationarity.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed
from core.indicators import rsi, macd, adx
from core.stock_selector import StockSelector

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
TOP_N = 20
LOOKBACK = 20
TRAIN_STEPS = 50_000
ADX_TREND_THRESHOLD = 25.0
ADX_RANGE_THRESHOLD = 20.0


class RegimeSACBot(BaseBot):
    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot11_regime_sac", client, feed)
        self.selector = StockSelector(top_n=TOP_N, min_per_sector=1)
        self._selected: list[str] = []
        self.model_trend = None
        self.model_range = None
        self._load_models()

    def _load_models(self):
        from stable_baselines3 import SAC
        for regime in ("trend", "range"):
            path = MODEL_DIR / f"bot11_sac_{regime}"
            if path.with_suffix(".zip").exists():
                try:
                    if regime == "trend":
                        self.model_trend = SAC.load(str(path))
                    else:
                        self.model_range = SAC.load(str(path))
                except Exception:
                    pass
        self.logger.log("models_loaded",
                        trend=self.model_trend is not None,
                        ranging=self.model_range is not None)

    def _detect_regime(self, bars_df: pd.DataFrame) -> str:
        """Returns 'trend' or 'range' based on average ADX across watchlist."""
        from core.data_feed import WATCHLIST
        adx_vals = []
        for sym in WATCHLIST[:5]:  # sample 5 liquid stocks
            try:
                sym_bars = bars_df.xs(sym, level=0) if isinstance(bars_df.index, pd.MultiIndex) else None
                if sym_bars is None or len(sym_bars) < 20:
                    continue
                adx_val = float(adx(sym_bars["high"], sym_bars["low"], sym_bars["close"]).iloc[-1])
                if not np.isnan(adx_val):
                    adx_vals.append(adx_val)
            except Exception:
                continue
        if not adx_vals:
            return "trend"
        avg_adx = float(np.mean(adx_vals))
        return "trend" if avg_adx >= ADX_RANGE_THRESHOLD else "range"

    def run_once(self) -> dict:
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

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        regime = self._detect_regime(bars_df)
        model = self.model_trend if regime == "trend" else self.model_range
        if model is None:
            # Fall back to the other model if one isn't trained
            model = self.model_range if regime == "trend" else self.model_trend
        if model is None:
            return {}

        closes_all = bars_df["close"].unstack(level=0) if isinstance(
            bars_df.index, pd.MultiIndex) else bars_df
        self._selected = self.selector.select(closes_all)
        if not self._selected:
            return {}

        obs = self._build_obs(closes_all, self._selected)
        if obs is None:
            return {sym: 0.0 for sym in self._selected}

        action, _ = model.predict(obs, deterministic=True)
        exp_a = np.exp(action - action.max())
        weights = exp_a / exp_a.sum()
        signals = {}
        for i, sym in enumerate(self._selected):
            target_w = float(weights[i])
            current_w = self._current_weight(sym)
            signals[sym] = float(np.clip(target_w - current_w, -1.0, 1.0))
            self.logger.log_signal(sym, signals[sym], reason=f"regime={regime},w={target_w:.3f}")
        return signals

    def _build_obs(self, closes, symbols):
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
                [self._current_weight(s) for s in symbols] + [0.0], dtype=np.float32)
            return np.concatenate([log_rets, ind_parts, current_weights]).astype(np.float32)
        except Exception:
            return None

    def _current_weight(self, symbol: str) -> float:
        pv = self.portfolio.total_value
        if pv <= 0 or symbol not in self.portfolio.positions:
            return 0.0
        pos = self.portfolio.positions[symbol]
        return (pos.qty * pos.avg_cost) / pv


def _build_regime_env(universe_bars: pd.DataFrame, selector: StockSelector,
                      lookback: int, regime: str):
    """Build TradingEnv filtered to trending or ranging periods by ADX."""
    from envs.trading_env import TradingEnv
    from core.indicators import rsi, macd, adx as calc_adx
    if universe_bars.empty:
        return None
    closes_all = universe_bars["close"].unstack(level=0)
    selected = selector.select(closes_all)
    if not selected:
        return None
    closes = closes_all.reindex(columns=selected).dropna()
    if len(closes) < lookback + 10:
        return None
    # Filter rows to the desired regime using ADX of first selected stock
    try:
        bars_sym = universe_bars.xs(selected[0], level=0) if isinstance(universe_bars.index, pd.MultiIndex) else None
        if bars_sym is not None and len(bars_sym) >= 20:
            adx_series = calc_adx(bars_sym["high"], bars_sym["low"], bars_sym["close"]).reindex(closes.index).fillna(22)
            if regime == "trend":
                mask = adx_series >= ADX_TREND_THRESHOLD
            else:
                mask = adx_series <= ADX_RANGE_THRESHOLD
            filtered = closes[mask]
            if len(filtered) >= lookback + 10:
                closes = filtered
    except Exception:
        pass
    ind_cols = {}
    for sym in selected:
        close_s = closes[sym]
        ind_cols[f"{sym}_rsi"] = rsi(close_s).fillna(50) / 100.0
        mdf = macd(close_s)
        ind_cols[f"{sym}_macd"] = (mdf["histogram"] / (close_s.mean() + 1e-8)).fillna(0)
    ind_df = pd.DataFrame(ind_cols, index=closes.index).fillna(0)
    return TradingEnv(closes, ind_df, lookback=lookback)
