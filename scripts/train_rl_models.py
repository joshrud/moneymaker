"""
Standalone RL model trainer. Run once to pre-train Bot 2 (SAC) and Bot 4 (PPO)
before market hours. Re-run periodically to retrain on fresh data.

Usage:
  .venv/bin/python scripts/train_rl_models.py
  .venv/bin/python scripts/train_rl_models.py --bot2-steps 100000 --bot4-steps 50000
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed
from core.logger import TradeLogger


def train_bot2(feed: DataFeed, steps: int = 100_000):
    """
    Trains the two-stage SAC portfolio model and saves to models/bot2_sac.zip.

    Stage 1: StockSelector scores the full 261-stock universe and picks the
             top 20 stocks (at least 1 per GICS sector) from 2 years of data.
    Stage 2: SAC trains on those 20 stocks using TradingEnv.
    """
    from stable_baselines3 import SAC
    from bots.bot2_sac_rl import _build_env, TOP_N, LOOKBACK
    from core.stock_selector import StockSelector

    logger = TradeLogger("train_bot2_sac")
    model_path = Path(__file__).resolve().parent.parent / "models" / "bot2_sac"
    model_path.parent.mkdir(exist_ok=True)

    print("Fetching universe bars for 261 stocks (this may take ~60 s)...")
    bars = feed.universe_bars(lookback=504)

    selector = StockSelector(top_n=TOP_N, min_per_sector=1)
    env = _build_env(bars, selector, LOOKBACK)
    if env is None:
        print("ERROR: could not build training environment — insufficient data.")
        return

    n_assets = env.n_assets
    print(f"Training SAC on {n_assets} selected stocks for {steps:,} steps...")
    model = SAC(
        "MlpPolicy", env, verbose=1, learning_rate=3e-4,
        buffer_size=50_000, batch_size=256, gamma=0.99,
        ent_coef="auto", device="cpu",
    )
    model.learn(total_timesteps=steps)
    model.save(str(model_path))
    print(f"Bot 2 SAC model saved to {model_path}.zip")
    logger.log("training_complete", steps=steps, n_assets=n_assets, path=str(model_path))


def train_bot4(feed: DataFeed, steps: int = 50_000):
    """Trains PPO models for each pair and saves to models/bot4_ppo_*.zip."""
    from stable_baselines3 import PPO
    from envs.trading_env import PairsTradingEnv
    from core.data_feed import PAIRS
    from scipy import stats

    logger = TradeLogger("train_bot4_ppo")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    for sym_a, sym_b in PAIRS:
        key = f"{sym_a}_{sym_b}"
        print(f"Training PPO for pair {key}...")

        bars = feed.daily_bars([sym_a, sym_b], lookback=504)
        closes = bars["close"].unstack(level=0).dropna()
        if sym_a not in closes.columns or sym_b not in closes.columns:
            print(f"  Skipping {key}: missing price data.")
            continue

        slope, intercept, *_ = stats.linregress(closes[sym_b], closes[sym_a])
        spread = closes[sym_a] - slope * closes[sym_b] - intercept

        env = PairsTradingEnv(spread)
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                    n_steps=1024, batch_size=64, gamma=0.99, device="cpu")
        model.learn(total_timesteps=steps)
        model_path = models_dir / f"bot4_ppo_{sym_a}_{sym_b}"
        model.save(str(model_path))
        print(f"  Saved to {model_path}.zip")
        logger.log("training_complete", pair=key, steps=steps, path=str(model_path))


def train_bot7(feed, steps=75_000):
    """Trains TD3 models for each of the 10 expanded pairs."""
    from stable_baselines3 import TD3
    from envs.trading_env import PairsTradingEnv
    from scipy import stats
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    logger = TradeLogger("train_bot7_td3")
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    pairs = [
        ("AAPL", "MSFT"), ("NVDA", "AMD"), ("AMZN", "GOOGL"),
        ("JPM", "BAC"), ("XOM", "CVX"), ("PEP", "KO"),
        ("GS", "MS"), ("V", "MA"), ("HD", "TGT"), ("META", "NFLX"),
    ]
    all_syms = list({s for pair in pairs for s in pair})
    bars = feed.daily_bars(symbols=all_syms, lookback=504)
    closes = bars["close"].unstack(level=0).dropna()
    for sym_a, sym_b in pairs:
        key = f"{sym_a}_{sym_b}"
        if sym_a not in closes.columns or sym_b not in closes.columns:
            print(f"  Skipping {key}: missing data")
            continue
        slope, intercept, *_ = stats.linregress(closes[sym_b], closes[sym_a])
        spread = closes[sym_a] - slope * closes[sym_b] - intercept
        env = PairsTradingEnv(spread)

        class ContinuousPairsEnv(gym.Env):
            """Wraps PairsTradingEnv with a continuous action for TD3."""
            def __init__(self, pairs_env):
                self.env = pairs_env
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                self.observation_space = pairs_env.observation_space
            def reset(self, **kwargs): return self.env.reset(**kwargs)
            def step(self, action):
                a = float(action[0])
                if a > 0.5: discrete = 1
                elif a < -0.5: discrete = 2
                elif abs(a) < 0.1: discrete = 0
                else: discrete = 3
                return self.env.step(discrete)

        cont_env = ContinuousPairsEnv(env)
        print(f"  Training TD3 for pair {key}...")
        model = TD3("MlpPolicy", cont_env, verbose=0, learning_rate=3e-4,
                    buffer_size=50_000, batch_size=64, gamma=0.99, device="cpu")
        model.learn(total_timesteps=steps)
        path = models_dir / f"bot7_td3_{key}"
        model.save(str(path))
        print(f"    Saved {path}.zip")
        logger.log("training_complete", pair=key, steps=steps)


def train_bot8(feed, steps=75_000):
    """Trains SAC with Sortino reward for bot8."""
    from stable_baselines3 import SAC
    from core.stock_selector import StockSelector
    from bots.bot8_sac_sortino import _build_sortino_env
    logger = TradeLogger("train_bot8_sac_sortino")
    model_path = Path(__file__).resolve().parent.parent / "models" / "bot8_sac_sortino"
    model_path.parent.mkdir(exist_ok=True)
    print("Fetching universe bars for bot8...")
    bars = feed.universe_bars(lookback=504)
    selector = StockSelector(top_n=20, min_per_sector=1)
    env = _build_sortino_env(bars, selector, lookback=20)
    if env is None:
        print("ERROR: could not build bot8 Sortino env.")
        return
    print(f"Training bot8 SAC (Sortino) for {steps:,} steps...")
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                buffer_size=50_000, batch_size=256, gamma=0.99,
                ent_coef="auto", device="cpu")
    model.learn(total_timesteps=steps)
    model.save(str(model_path))
    print(f"Bot 8 saved to {model_path}.zip")
    logger.log("training_complete", steps=steps)


def train_bot9(feed, steps=75_000):
    """Trains DQN VWAP bot on each WATCHLIST stock, saves combined model."""
    from stable_baselines3 import DQN
    from core.data_feed import WATCHLIST
    from core.indicators import ema as calc_ema
    from envs.trading_env import VWAPTradingEnv
    import pandas as pd
    import numpy as np
    logger = TradeLogger("train_bot9_dqn_vwap")
    model_path = Path(__file__).resolve().parent.parent / "models" / "bot9_dqn_vwap"
    model_path.parent.mkdir(exist_ok=True)
    print("Fetching bars for bot9 DQN VWAP...")
    bars = feed.daily_bars(symbols=WATCHLIST, lookback=504)
    closes = bars["close"].unstack(level=0).dropna() if not bars.empty else None
    if closes is None or closes.empty:
        print("ERROR: no data for bot9"); return
    sym = next((s for s in WATCHLIST if s in closes.columns and len(closes[s].dropna()) > 50), None)
    if sym is None:
        print("ERROR: no suitable symbol for bot9"); return
    close_s = closes[sym].dropna()
    df_ind = pd.DataFrame({
        "rsi": pd.Series(np.linspace(30, 70, len(close_s)), index=close_s.index),
        "ema20": calc_ema(close_s, window=20),
    }, index=close_s.index).ffill().bfill()
    env = VWAPTradingEnv(close_s.to_frame("close"), df_ind, lookback=20)
    print(f"Training DQN VWAP on {sym} for {steps:,} steps...")
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-4,
                buffer_size=50_000, batch_size=64, gamma=0.99,
                exploration_final_eps=0.05, device="cpu")
    model.learn(total_timesteps=steps)
    model.save(str(model_path))
    print(f"Bot 9 saved to {model_path}.zip")
    logger.log("training_complete", symbol=sym, steps=steps)


def train_bot11(feed, steps=50_000):
    """Trains two SAC models for bot11: one trend, one ranging."""
    from stable_baselines3 import SAC
    from core.stock_selector import StockSelector
    from bots.bot11_regime_sac import _build_regime_env
    logger = TradeLogger("train_bot11_regime_sac")
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    print("Fetching universe bars for bot11...")
    bars = feed.universe_bars(lookback=504)
    selector = StockSelector(top_n=20, min_per_sector=1)
    for regime in ("trend", "range"):
        env = _build_regime_env(bars, selector, lookback=20, regime=regime)
        if env is None:
            print(f"  Skipping bot11 {regime}: insufficient filtered data")
            continue
        path = model_dir / f"bot11_sac_{regime}"
        print(f"  Training bot11 SAC ({regime}) for {steps:,} steps...")
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                    buffer_size=50_000, batch_size=256, gamma=0.99,
                    ent_coef="auto", device="cpu")
        model.learn(total_timesteps=steps)
        model.save(str(path))
        print(f"  Saved {path}.zip")
        logger.log("training_complete", regime=regime, steps=steps)


def main():
    parser = argparse.ArgumentParser(description="Train RL models offline")
    parser.add_argument("--bot2-steps", type=int, default=100_000)
    parser.add_argument("--bot4-steps", type=int, default=50_000)
    parser.add_argument("--bot2-only", action="store_true")
    parser.add_argument("--bot4-only", action="store_true")
    parser.add_argument("--bot7-only", action="store_true")
    parser.add_argument("--bot8-only", action="store_true")
    parser.add_argument("--bot9-only", action="store_true")
    parser.add_argument("--bot11-only", action="store_true")
    parser.add_argument("--bot14-only", action="store_true")
    parser.add_argument("--new-bots", action="store_true",
                        help="Train all new RL bots (7, 8, 9, 11, 14)")
    args = parser.parse_args()

    client = AlpacaClient()
    feed = DataFeed(client)

    only_flags = [args.bot2_only, args.bot4_only, args.bot7_only, args.bot8_only,
                  args.bot9_only, args.bot11_only, args.bot14_only, args.new_bots]
    run_all = not any(only_flags)

    if run_all or args.bot2_only:
        train_bot2(feed, args.bot2_steps)
    if run_all or args.bot4_only:
        train_bot4(feed, args.bot4_steps)
    if args.new_bots or args.bot7_only:
        train_bot7(feed)
    if args.new_bots or args.bot8_only:
        train_bot8(feed)
    if args.new_bots or args.bot9_only:
        train_bot9(feed)
    if args.new_bots or args.bot11_only:
        train_bot11(feed)
    if args.new_bots or args.bot14_only:
        try:
            from scripts.train_rl_models import train_bot14
            train_bot14(feed)
        except ImportError:
            print("train_bot14 not yet defined — skipping")
    print("\nAll training complete.")


if __name__ == "__main__":
    main()
