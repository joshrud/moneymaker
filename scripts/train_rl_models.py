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
    """Trains the SAC portfolio model and saves to models/bot2_sac.zip."""
    from stable_baselines3 import SAC
    from envs.trading_env import TradingEnv
    from core.data_feed import WATCHLIST
    from core.indicators import rsi, macd
    import pandas as pd

    logger = TradeLogger("train_bot2_sac")
    assets = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]
    model_path = Path("models/bot2_sac")
    model_path.parent.mkdir(exist_ok=True)

    print(f"Training SAC (Bot 2) for {steps:,} steps...")
    bars = feed.daily_bars(assets, lookback=504)
    closes = bars["close"].unstack(level=0).reindex(columns=assets).dropna()

    ind_cols = {}
    for sym in assets:
        close_s = closes[sym]
        ind_cols[f"{sym}_rsi"] = rsi(close_s).fillna(50) / 100.0
        mdf = macd(close_s)
        ind_cols[f"{sym}_macd"] = (mdf["histogram"] / closes[sym].mean()).fillna(0)
    ind_df = pd.DataFrame(ind_cols, index=closes.index).fillna(0)

    env = TradingEnv(closes, ind_df, lookback=20)
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                buffer_size=50_000, batch_size=256, gamma=0.99,
                ent_coef="auto", device="cpu")
    model.learn(total_timesteps=steps)
    model.save(str(model_path))
    print(f"Bot 2 SAC model saved to {model_path}.zip")
    logger.log("training_complete", steps=steps, path=str(model_path))


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


def main():
    parser = argparse.ArgumentParser(description="Train RL models offline")
    parser.add_argument("--bot2-steps", type=int, default=100_000)
    parser.add_argument("--bot4-steps", type=int, default=50_000)
    parser.add_argument("--bot2-only", action="store_true")
    parser.add_argument("--bot4-only", action="store_true")
    args = parser.parse_args()

    client = AlpacaClient()
    feed = DataFeed(client)

    if not args.bot4_only:
        train_bot2(feed, args.bot2_steps)
    if not args.bot2_only:
        train_bot4(feed, args.bot4_steps)
    print("\nAll training complete.")


if __name__ == "__main__":
    main()
