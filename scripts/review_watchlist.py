"""
Weekly watchlist reviewer. Re-scores all active symbols in the dynamic watchlist
and deactivates underperformers.

Removal criteria (any one triggers deactivation):
  - Score dropped below 0.35 on this review
  - Price now outside $1–$80 (stock moved out of small-cap range or delisted)
  - Avg daily volume dropped below 50k (gone illiquid)
  - Symbol has been active > 90 days without ever scoring above 0.60

Usage:
  .venv/bin/python scripts/review_watchlist.py
  .venv/bin/python scripts/review_watchlist.py --dry-run
"""
from __future__ import annotations
import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.alpaca_client import AlpacaClient
from core.dynamic_watchlist import active_symbols, update_entry, deactivate
from alpaca.data.timeframe import TimeFrame

NY = ZoneInfo("America/New_York")
BATCH_SIZE = 50
LOOKBACK_CALENDAR_DAYS = 90

MIN_SCORE = 0.35
MIN_PRICE = 1.0
MAX_PRICE = 80.0
MIN_AVG_VOLUME = 50_000
MAX_DAYS_WITHOUT_HIGH_SCORE = 90
HIGH_SCORE_THRESHOLD = 0.60


def _fetch_bars(client: AlpacaClient, symbols: list[str]):
    import pandas as pd

    end = datetime.now(NY).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    frames = []

    for i in range(0, len(symbols), BATCH_SIZE):
        chunk = symbols[i:i + BATCH_SIZE]
        try:
            df = client.get_bars(chunk, TimeFrame.Day, start=start, end=end,
                                 limit=65 * len(chunk))
            if not df.empty:
                frames.append(df)
        except Exception:
            pass

    return pd.concat(frames) if frames else pd.DataFrame()


def _score(sym_bars) -> dict | None:
    sym_bars = sym_bars.sort_index()
    if len(sym_bars) < 25:
        return None

    closes = sym_bars["close"].values[-25:]
    volumes = sym_bars["volume"].values[-25:]
    highs = sym_bars["high"].values[-14:]
    lows = sym_bars["low"].values[-14:]
    close14 = sym_bars["close"].values[-14:]

    price = float(closes[-1])
    if price <= 0:
        return None

    mom_20d = float(closes[-1] / closes[-20] - 1) if closes[-20] > 0 else 0.0
    avg_vol_20 = float(volumes.mean())
    avg_vol_5 = float(volumes[-5:].mean())
    vol_surge = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 0.0

    x = np.arange(20)
    lr = linregress(x, closes[-20:])
    r2 = float(lr.rvalue ** 2)
    trend_r2 = r2 if lr.slope > 0 else -r2

    daily_range_pct = float(((highs - lows) / close14).mean())
    vol_score = float(np.exp(-((daily_range_pct - 0.025) ** 2) / (2 * 0.012 ** 2)))

    mom_norm = 1.0 / (1.0 + np.exp(-15.0 * mom_20d))
    vsurge_norm = min(vol_surge / 3.0, 1.0)
    tr2_norm = (trend_r2 + 1.0) / 2.0
    composite = round(0.30 * mom_norm + 0.25 * vsurge_norm + 0.25 * tr2_norm + 0.20 * vol_score, 4)

    return {
        "price": round(price, 2),
        "avg_volume": int(avg_vol_20),
        "momentum_20d": round(mom_20d, 4),
        "score": composite,
    }


def main():
    parser = argparse.ArgumentParser(description="Weekly dynamic watchlist review")
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate and print only; do not update watchlist file")
    args = parser.parse_args()

    today = date.today()
    print(f"\nWatchlist review — {today.isoformat()}\n")

    symbols = active_symbols()
    if not symbols:
        print("Dynamic watchlist is empty.")
        return

    print(f"{len(symbols)} active symbols to review\n")

    client = AlpacaClient()
    bars = _fetch_bars(client, symbols)

    kept, removed, no_data = [], [], []

    for sym in symbols:
        try:
            sym_bars = bars.loc[sym]
        except KeyError:
            no_data.append(sym)
            continue

        metrics = _score(sym_bars)
        if metrics is None:
            no_data.append(sym)
            continue

        score = metrics["score"]
        price = metrics["price"]
        avg_vol = metrics["avg_volume"]

        # Check removal criteria
        reason = None
        if score < MIN_SCORE:
            reason = f"score_too_low ({score:.3f} < {MIN_SCORE})"
        elif not (MIN_PRICE <= price <= MAX_PRICE):
            reason = f"price_out_of_range (${price:.2f})"
        elif avg_vol < MIN_AVG_VOLUME:
            reason = f"illiquid (vol {avg_vol:,} < {MIN_AVG_VOLUME:,})"
        else:
            # Check if stale: added > 90 days ago and never hit high score
            from core.dynamic_watchlist import all_entries
            entry = all_entries().get(sym, {})
            added_str = entry.get("added", today.isoformat())
            added_date = date.fromisoformat(added_str)
            days_active = (today - added_date).days
            best_score = max(entry.get("score", 0.0), score)
            if days_active > MAX_DAYS_WITHOUT_HIGH_SCORE and best_score < HIGH_SCORE_THRESHOLD:
                reason = f"stale ({days_active}d active, best score {best_score:.3f})"

        if reason:
            removed.append((sym, score, reason))
            if not args.dry_run:
                deactivate(sym, reason=reason)
        else:
            kept.append((sym, score, price, avg_vol))
            if not args.dry_run:
                update_entry(sym, score=score, price=price, avg_volume=avg_vol,
                             last_checked=today.isoformat(), momentum_20d=metrics["momentum_20d"])

    # Print results
    if kept:
        print(f"KEPT ({len(kept)}):")
        print(f"  {'Symbol':<8} {'Score':>7} {'Price':>7} {'AvgVol':>10}")
        print(f"  {'-'*38}")
        for sym, score, price, vol in sorted(kept, key=lambda x: -x[1]):
            print(f"  {sym:<8} {score:>7.4f} ${price:>6.2f} {vol:>10,}")

    if removed:
        print(f"\nREMOVED ({len(removed)}):")
        for sym, score, reason in removed:
            print(f"  {sym:<8} score={score:.4f}  reason: {reason}")

    if no_data:
        print(f"\nNO DATA ({len(no_data)}): {', '.join(no_data)}")
        if not args.dry_run:
            for sym in no_data:
                deactivate(sym, reason="no_bar_data")

    print(f"\nSummary: {len(kept)} kept · {len(removed)} removed · {len(no_data)} no-data")
    if args.dry_run:
        print("(dry-run: no changes written)")


if __name__ == "__main__":
    main()
