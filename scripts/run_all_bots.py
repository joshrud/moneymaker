"""
Main runner: starts all 5 bots, runs them concurrently each market cycle,
and writes the daily report at close.

Usage:
  .venv/bin/python scripts/run_all_bots.py            # run once (good for cron)
  .venv/bin/python scripts/run_all_bots.py --loop     # loop every 5 min during market hours

Cron job for daily open (9:35 ET):
  35 9 * * 1-5 cd /path/to/moneymaker && .venv/bin/python scripts/run_all_bots.py
"""
from __future__ import annotations
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed
from core.reporter import generate_report
from bots import (
    MomentumBot, SACBot, ClaudeSentimentBot, FinBERTPPOBot, EnsembleBot,
    EMABot, TD3PairsBot, SortinoSACBot, DQNVWAPBot, LGBMFactorBot,
    RegimeSACBot, CoveredCallBot, CSPBot, DeepHedgingBot, AggressiveEnsembleBot,
)

NY = ZoneInfo("America/New_York")
CYCLE_SECONDS = 300  # 5-minute cycle during market hours


def build_bots(client: AlpacaClient, feed: DataFeed):
    """Instantiates all 15 bots. RL models auto-train if no model exists."""
    print("Initialising bots (RL models will train if not found)...")
    bot1 = MomentumBot(client, feed)
    bot2 = SACBot(client, feed)
    bot3 = ClaudeSentimentBot(client, feed)
    bot4 = FinBERTPPOBot(client, feed)
    bot5 = EnsembleBot(client, feed, sub_bots=[bot1, bot2, bot3, bot4])
    bot6 = EMABot(client, feed)
    bot7 = TD3PairsBot(client, feed)
    bot8 = SortinoSACBot(client, feed)
    bot9 = DQNVWAPBot(client, feed)
    bot10 = LGBMFactorBot(client, feed)
    bot11 = RegimeSACBot(client, feed)
    bot12 = CoveredCallBot(client, feed)
    bot13 = CSPBot(client, feed)
    bot14 = DeepHedgingBot(client, feed)
    bot15 = AggressiveEnsembleBot(client, feed, sub_bots=[
        bot6, bot7, bot8, bot9, bot10, bot11, bot12, bot13, bot14,
    ])
    return [bot1, bot2, bot3, bot4, bot5, bot6, bot7, bot8, bot9, bot10,
            bot11, bot12, bot13, bot14, bot15]


def run_cycle(bots: list, feed: DataFeed) -> list[dict]:
    """Runs all bots in parallel for one cycle. Returns their summaries."""
    feed.clear_cache()  # refresh data each cycle
    results = []
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(bot.run_once): bot.name for bot in bots}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                pnl = result.get("daily_pnl", 0.0)
                pv = result.get("portfolio_value", 0.0)
                print(f"  {name}: PnL=${pnl:+,.2f}  Value=${pv:,.2f}")
            except Exception as e:
                print(f"  {name}: ERROR — {e}")
    return results


def is_market_hours(client: AlpacaClient) -> bool:
    try:
        return client.is_market_open()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all 15 trading bots")
    parser.add_argument("--loop", action="store_true", help="Loop every 5 min during market hours")
    parser.add_argument("--force", action="store_true", help="Run even if market is closed (testing)")
    args = parser.parse_args()

    client = AlpacaClient()
    feed = DataFeed(client)
    bots = build_bots(client, feed)

    # Reset daily state for all bots
    for bot in bots:
        bot.reset_day()

    print(f"\n{'='*60}")
    print(f"Moneymaker — {datetime.now(NY).strftime('%Y-%m-%d %H:%M ET')}")
    print(f"{'='*60}\n")

    if args.loop:
        print("Loop mode: running every 5 minutes during market hours.\n")
        while True:
            if is_market_hours(client) or args.force:
                print(f"[{datetime.now(NY).strftime('%H:%M:%S')}] Running cycle...")
                run_cycle(bots, feed)
            else:
                print(f"[{datetime.now(NY).strftime('%H:%M:%S')}] Market closed. Waiting...")
                # Generate end-of-day report if near close
                now = datetime.now(NY)
                if now.hour == 16 and now.minute < 10:
                    report_path = generate_report()
                    print(f"Daily report written: {report_path}")
            time.sleep(CYCLE_SECONDS)
    else:
        if not is_market_hours(client) and not args.force:
            print("Market is currently closed. Use --force to run anyway (for testing).")
        run_cycle(bots, feed)
        report_path = generate_report()
        print(f"\nDaily report: {report_path}")


if __name__ == "__main__":
    main()
