"""
Bot 10 — LightGBM Factor Model
Strategy: Supervised gradient-boosted trees trained on momentum + volatility factors
from the stock selector to predict next-day return quintile. Complement to RL bots —
fast, interpretable, no environment simulation needed.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed
from core.stock_selector import StockSelector, DEFAULT_WEIGHTS
from core.universe import UNIVERSE

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bot10_lgbm.pkl"
TOP_N = 5   # take top 5 predicted stocks as buys


class LGBMFactorBot(BaseBot):
    def __init__(self, client: AlpacaClient, feed: DataFeed):
        super().__init__("bot10_lgbm_factor", client, feed)
        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        import pickle
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            return
        self.logger.log("training_start")
        self._train()

    def _train(self):
        try:
            import lightgbm as lgb
            import pickle
            bars = self.feed.universe_bars(lookback=504)
            if bars.empty:
                return
            closes = bars["close"].unstack(level=0).dropna(axis=1, how="all")
            X, y = self._build_features(closes)
            if X is None or len(X) < 50:
                return
            model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                        num_leaves=31, random_state=42, verbose=-1)
            model.fit(X, y)
            MODEL_PATH.parent.mkdir(exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
            self.model = model
            self.logger.log("training_done", n_samples=len(X))
        except Exception as e:
            self.logger.log("training_error", error=str(e))

    def _build_features(self, closes: pd.DataFrame):
        """Build (features, next-day-return-quintile) dataset."""
        try:
            universe_cols = [c for c in closes.columns if c in set(UNIVERSE)]
            closes = closes[universe_cols]
            rows_X, rows_y = [], []
            for t in range(63, len(closes) - 1):
                window = closes.iloc[max(0, t - 252):t + 1]
                last = window.iloc[-1]
                for sym in universe_cols:
                    if sym not in window.columns:
                        continue
                    try:
                        def mom(lag):
                            if len(window) <= lag:
                                return 0.0
                            prev = window.iloc[-(lag + 1)][sym]
                            return float(np.clip((last[sym] - prev) / (abs(prev) + 1e-8), -1, 1))
                        m1 = mom(21); m3 = mom(63)
                        m12 = mom(252) if len(window) > 252 else m3
                        m12m1 = float(np.clip(m12 - m1, -1, 1))
                        rets = window[sym].pct_change().dropna().iloc[-20:]
                        vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.2
                        rows_X.append([m1, m3, m12m1, vol])
                        fwd_ret = float((closes[sym].iloc[t + 1] - last[sym]) / (last[sym] + 1e-8))
                        rows_y.append(fwd_ret)
                    except Exception:
                        continue
            if not rows_X:
                return None, None
            X = np.array(rows_X, dtype=np.float32)
            y_raw = np.array(rows_y, dtype=np.float32)
            # Convert to quintile labels 0-4 (label 4 = top quintile = buy signal)
            y = pd.qcut(y_raw, q=5, labels=False, duplicates="drop")
            mask = ~np.isnan(y.astype(float))
            return X[mask], y[mask].astype(int)
        except Exception:
            return None, None

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        if self.model is None:
            return {}
        closes = bars_df["close"].unstack(level=0) if isinstance(bars_df.index, pd.MultiIndex) else bars_df
        universe_cols = [c for c in closes.columns if c in set(UNIVERSE)]
        if not universe_cols:
            return {}
        features = []
        syms = []
        last = closes.iloc[-1]
        for sym in universe_cols:
            if sym not in closes.columns:
                continue
            try:
                window = closes[sym].dropna()
                def mom(lag):
                    if len(window) <= lag:
                        return 0.0
                    prev = float(window.iloc[-(lag + 1)])
                    return float(np.clip((float(window.iloc[-1]) - prev) / (abs(prev) + 1e-8), -1, 1))
                m1 = mom(21); m3 = mom(63)
                m12 = mom(252) if len(window) > 252 else m3
                m12m1 = float(np.clip(m12 - m1, -1, 1))
                rets = window.pct_change().dropna().iloc[-20:]
                vol = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.2
                features.append([m1, m3, m12m1, vol])
                syms.append(sym)
            except Exception:
                continue
        if not features:
            return {}
        X = np.array(features, dtype=np.float32)
        try:
            proba = self.model.predict_proba(X)
            top_quintile_prob = proba[:, -1]  # prob of being top quintile
        except Exception:
            return {}
        signals = {}
        ranked = sorted(zip(syms, top_quintile_prob), key=lambda x: -x[1])
        for i, (sym, prob) in enumerate(ranked):
            if i < TOP_N and prob > 0.4:
                signals[sym] = float(prob)
                self.logger.log_signal(sym, signals[sym], reason=f"lgbm_top_quintile_prob={prob:.3f}")
            elif sym in self.portfolio.positions:
                # Check if we should exit (fallen to bottom quintile)
                if prob < 0.2:
                    signals[sym] = -prob
        return signals
