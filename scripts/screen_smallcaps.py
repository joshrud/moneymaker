"""
Small-cap screener. Samples random stocks from Alpaca's active US equity universe,
scores them on 4 technical criteria, and promotes top candidates to the dynamic
watchlist so rules-based bots (bot1, bot6) can monitor them.

Scoring criteria:
  momentum_20d  (0.30) — 20-day price return; positive trend preferred
  volume_surge  (0.25) — 5-day avg volume vs 20-day avg; detects accumulation
  trend_r2      (0.25) — OLS R² over 20 days; rewards consistent directional move
  vol_score     (0.20) — daily range %; bell curve peaking at 2.5% (not too wild/flat)

Usage:
  .venv/bin/python scripts/screen_smallcaps.py              # sample 200 stocks
  .venv/bin/python scripts/screen_smallcaps.py --n 400      # larger sample
  .venv/bin/python scripts/screen_smallcaps.py --top 30     # show/promote top 30
  .venv/bin/python scripts/screen_smallcaps.py --dry-run    # score only, no watchlist update
"""
from __future__ import annotations
import argparse
import json
import random
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
from core.dynamic_watchlist import add_symbols
from core.universe import UNIVERSE
from alpaca.data.timeframe import TimeFrame

NY = ZoneInfo("America/New_York")
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
BATCH_SIZE = 50
LOOKBACK_CALENDAR_DAYS = 90   # covers ~60 trading days
PROMOTION_THRESHOLD = 0.55
MIN_PRICE = 2.0
MAX_PRICE = 40.0
MIN_AVG_VOLUME = 100_000


def _get_candidate_symbols(client: AlpacaClient, n_sample: int) -> list[str]:
    """All active tradeable US equities not already in the main universe, random-sampled."""
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    assets = client.trader.get_all_assets(
        GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    )
    known = set(UNIVERSE)
    eligible = [
        a.symbol for a in assets
        if a.tradable
        and a.symbol not in known
        and "/" not in a.symbol
        and "." not in a.symbol
        and len(a.symbol) <= 5
    ]
    random.shuffle(eligible)
    return eligible[:n_sample]


def _fetch_bars(client: AlpacaClient, symbols: list[str]):
    """Daily bars for all symbols in batches. Returns MultiIndex DataFrame."""
    import pandas as pd

    end = datetime.now(NY).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    n_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE
    frames = []

    for i in range(0, len(symbols), BATCH_SIZE):
        chunk = symbols[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{n_batches} ({len(chunk)} symbols)...    ", end="\r")
        try:
            df = client.get_bars(chunk, TimeFrame.Day, start=start, end=end,
                                 limit=65 * len(chunk))
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"\n  Warning: batch {batch_num} failed — {e}")

    print()
    import pandas as pd
    return pd.concat(frames) if frames else pd.DataFrame()


def _score(sym_bars) -> dict | None:
    """
    Returns raw metrics for one symbol, or None if there's not enough history.
    sym_bars is a DataFrame with columns open/high/low/close/volume, sorted by date.
    """
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

    return {
        "price": round(price, 2),
        "avg_volume": int(avg_vol_20),
        "momentum_20d": round(mom_20d, 4),
        "volume_surge": round(vol_surge, 4),
        "trend_r2": round(trend_r2, 4),
        "vol_score": round(vol_score, 4),
    }


def _composite(m: dict) -> float:
    mom_norm = 1.0 / (1.0 + np.exp(-15.0 * m["momentum_20d"]))
    vsurge_norm = min(m["volume_surge"] / 3.0, 1.0)
    tr2_norm = (m["trend_r2"] + 1.0) / 2.0
    return round(
        0.30 * mom_norm +
        0.25 * vsurge_norm +
        0.25 * tr2_norm +
        0.20 * m["vol_score"],
        4,
    )


def main():
    parser = argparse.ArgumentParser(description="Small-cap screener")
    parser.add_argument("--n", type=int, default=200, help="Symbols to sample (default 200)")
    parser.add_argument("--top", type=int, default=20, help="Candidates to show/promote (default 20)")
    parser.add_argument("--min-price", type=float, default=MIN_PRICE, help="Min price (default $2)")
    parser.add_argument("--max-price", type=float, default=MAX_PRICE, help="Max price (default $40)")
    parser.add_argument("--min-volume", type=int, default=MIN_AVG_VOLUME,
                        help="Min avg daily volume (default 100k)")
    parser.add_argument("--threshold", type=float, default=PROMOTION_THRESHOLD,
                        help="Minimum score to promote to watchlist (default 0.55)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score and print only; do not update watchlist")
    args = parser.parse_args()

    print(f"\nSmall-cap screener — {date.today().isoformat()}")
    print(f"Sample {args.n} symbols · show top {args.top} · promote ≥ {args.threshold}\n")

    client = AlpacaClient()

    print("Fetching asset list...")
    symbols = _get_candidate_symbols(client, args.n)
    print(f"  {len(symbols)} symbols after removing known universe\n")

    if not symbols:
        print("No candidates found.")
        return

    print("Fetching daily bars...")
    bars = _fetch_bars(client, symbols)
    if bars.empty:
        print("No bar data returned.")
        return

    available = bars.index.get_level_values(0).unique()
    print(f"  Data for {len(available)} symbols\n")

    # Score
    results = []
    for sym in available:
        try:
            sym_bars = bars.loc[sym]
        except KeyError:
            continue
        metrics = _score(sym_bars)
        if metrics is None:
            continue
        if not (args.min_price <= metrics["price"] <= args.max_price):
            continue
        if metrics["avg_volume"] < args.min_volume:
            continue
        metrics["symbol"] = sym
        metrics["score"] = _composite(metrics)
        results.append(metrics)

    if not results:
        print("No symbols passed price/volume filters.")
        return

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:args.top]
    promotable = [r for r in top if r["score"] >= args.threshold]

    # Print table
    print(f"{'#':<4} {'Symbol':<7} {'Price':>7} {'AvgVol':>10} {'Mom20d':>8} "
          f"{'VolSurge':>9} {'TrendR²':>8} {'Score':>7}")
    print("-" * 68)
    for i, r in enumerate(top, 1):
        flag = " ✓" if r["score"] >= args.threshold else ""
        print(
            f"{i:<4} {r['symbol']:<7} ${r['price']:>6.2f} {r['avg_volume']:>10,} "
            f"{r['momentum_20d']:>+8.1%} {r['volume_surge']:>8.2f}x "
            f"{r['trend_r2']:>+8.3f} {r['score']:>7.4f}{flag}"
        )

    print(f"\n{len(results)} symbols passed filters · "
          f"{len(promotable)} scored ≥ {args.threshold} (marked ✓)")

    # Save JSONL log
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"{date.today().isoformat()}_smallcap_screen.jsonl"
    with open(log_path, "w") as f:
        for r in top:
            f.write(json.dumps({**r, "ts": datetime.utcnow().isoformat()}) + "\n")
    print(f"Screen log: {log_path}")

    if not args.dry_run and promotable:
        add_symbols(promotable)
        print(f"Promoted {len(promotable)} symbols → data/dynamic_watchlist.json")
    elif args.dry_run:
        print("(dry-run: watchlist not updated)")


if __name__ == "__main__":
    main()
