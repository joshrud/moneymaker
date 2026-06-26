"""Tests for StockSelector and the two-stage SAC pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.universe import UNIVERSE, UNIVERSE_BY_SECTOR, SECTOR_MAP
from core.stock_selector import StockSelector


def make_closes(n_rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic (dates × symbols) close price DataFrame for the full universe."""
    np.random.seed(seed)
    prices = np.cumprod(
        1 + np.random.randn(n_rows, len(UNIVERSE)) * 0.01, axis=0
    ) * 100
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="B")
    return pd.DataFrame(prices, index=dates, columns=UNIVERSE)


# ── universe sanity ────────────────────────────────────────────────────────────

def test_universe_no_duplicates():
    assert len(UNIVERSE) == len(set(UNIVERSE))


def test_universe_sector_map_complete():
    for ticker in UNIVERSE:
        assert ticker in SECTOR_MAP, f"{ticker} missing from SECTOR_MAP"


def test_universe_covers_all_sectors():
    assert len(UNIVERSE_BY_SECTOR) == 11


def test_universe_min_size():
    assert len(UNIVERSE) >= 250


# ── StockSelector ──────────────────────────────────────────────────────────────

def test_selector_returns_correct_count():
    closes = make_closes()
    sel = StockSelector(top_n=20)
    picks = sel.select(closes)
    assert len(picks) == 20


def test_selector_all_unique():
    closes = make_closes()
    sel = StockSelector(top_n=20)
    picks = sel.select(closes)
    assert len(picks) == len(set(picks))


def test_selector_sector_coverage():
    """At least one pick per GICS sector (min_per_sector=1)."""
    closes = make_closes()
    sel = StockSelector(top_n=20, min_per_sector=1)
    picks = sel.select(closes)
    covered = {SECTOR_MAP[t] for t in picks}
    assert covered == set(UNIVERSE_BY_SECTOR.keys()), f"Missing sectors: {set(UNIVERSE_BY_SECTOR.keys()) - covered}"


def test_selector_all_picks_in_universe():
    closes = make_closes()
    sel = StockSelector(top_n=20)
    picks = sel.select(closes)
    for t in picks:
        assert t in SECTOR_MAP, f"{t} not in universe"


def test_selector_short_history_fallback():
    """Should not crash when closes has fewer than 252 rows."""
    closes = make_closes(n_rows=30)
    sel = StockSelector(top_n=20)
    picks = sel.select(closes)
    assert len(picks) <= 20


def test_selector_empty_df_returns_empty():
    sel = StockSelector(top_n=20)
    picks = sel.select(pd.DataFrame())
    assert picks == []


def test_selector_custom_weights():
    closes = make_closes()
    weights = {"mom_1m": 0.5, "mom_3m": 0.3, "mom_12m1": 0.1, "inv_vol": 0.1}
    sel = StockSelector(top_n=15, weights=weights)
    picks = sel.select(closes)
    assert len(picks) == 15


def test_selector_top_n_too_small_raises():
    with pytest.raises(ValueError):
        StockSelector(top_n=5, min_per_sector=1)  # 11 sectors > 5


# ── two-stage env construction ─────────────────────────────────────────────────

def test_build_env_returns_valid_env():
    from bots.bot2_sac_rl import _build_env, TOP_N, LOOKBACK

    closes = make_closes(n_rows=300)
    dates = closes.index
    frames = []
    for sym in UNIVERSE:
        idx = pd.MultiIndex.from_product([[sym], dates], names=["symbol", "timestamp"])
        frames.append(pd.DataFrame({"close": closes[sym].values}, index=idx))
    bars = pd.concat(frames)

    sel = StockSelector(top_n=TOP_N)
    env = _build_env(bars, sel, LOOKBACK)
    assert env is not None
    assert env.n_assets == TOP_N
    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]


def test_build_env_insufficient_data_returns_none():
    from bots.bot2_sac_rl import _build_env, LOOKBACK

    # Only 5 rows — not enough
    closes = make_closes(n_rows=5)
    dates = closes.index
    frames = []
    for sym in UNIVERSE[:10]:
        idx = pd.MultiIndex.from_product([[sym], dates], names=["symbol", "timestamp"])
        frames.append(pd.DataFrame({"close": closes[sym].values}, index=idx))
    bars = pd.concat(frames)

    sel = StockSelector(top_n=11)  # minimum valid top_n for 11 sectors
    env = _build_env(bars, sel, LOOKBACK)
    assert env is None
