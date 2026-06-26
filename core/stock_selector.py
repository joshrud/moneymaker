"""
Two-stage stock selection: score all universe stocks on momentum + volatility
factors, then return the top N with at least one pick per GICS sector.

Factors (all rank-normalised to [0, 1]):
  - 1-month momentum  (21-day return)
  - 3-month momentum  (63-day return)
  - 12-minus-1 momentum (252-day minus 21-day, avoids short-term reversal)
  - Inverse volatility (20-day realised vol, lower vol scores higher)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.universe import UNIVERSE_BY_SECTOR, SECTOR_MAP

# Default factor weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "mom_1m":   0.25,
    "mom_3m":   0.35,
    "mom_12m1": 0.30,
    "inv_vol":  0.10,
}


class StockSelector:
    """
    Selects top_n stocks from a close-price DataFrame using multi-factor scoring.
    Guarantees at least min_per_sector stocks from each GICS sector.
    """

    def __init__(
        self,
        top_n: int = 20,
        min_per_sector: int = 1,
        weights: dict | None = None,
    ):
        if top_n < len(UNIVERSE_BY_SECTOR) * min_per_sector:
            raise ValueError(
                f"top_n={top_n} is too small to satisfy min_per_sector={min_per_sector} "
                f"across {len(UNIVERSE_BY_SECTOR)} sectors"
            )
        self.top_n = top_n
        self.min_per_sector = min_per_sector
        self.weights = weights or DEFAULT_WEIGHTS

    def select(self, closes: pd.DataFrame) -> list[str]:
        """
        Returns up to top_n tickers sorted descending by composite factor score.

        closes: (dates × symbols) DataFrame — only columns present in UNIVERSE
                are scored; others are ignored.
        """
        # Restrict to universe tickers that have data
        universe_cols = [c for c in closes.columns if c in SECTOR_MAP]
        closes = closes[universe_cols].dropna(axis=1, how="all")

        if closes.empty:
            return []
        if len(closes) < 22:
            # Too little history — return top_n by alphabetical order as fallback
            return sorted(closes.columns)[: self.top_n]

        scores = self._compute_scores(closes)
        return self._sector_constrained_select(scores)

    # ── internals ─────────────────────────────────────────────────────────────

    def _compute_scores(self, closes: pd.DataFrame) -> pd.Series:
        n = len(closes)
        last = closes.iloc[-1]

        def _mom(lag: int) -> pd.Series:
            if n <= lag:
                return pd.Series(0.0, index=closes.columns)
            prev = closes.iloc[-(lag + 1)]
            return ((last - prev) / (prev.abs() + 1e-8)).clip(-1.0, 1.0)

        mom_1m  = _mom(21)
        mom_3m  = _mom(63)  if n > 63  else pd.Series(0.0, index=closes.columns)
        mom_12m = _mom(252) if n > 252 else mom_3m
        # 12-minus-1: strip the most-recent month to avoid reversal
        mom_12m1 = (mom_12m - mom_1m).clip(-1.0, 1.0)

        # 20-day realised vol (annualised), then invert and min-max normalise
        rets = closes.pct_change().dropna().iloc[-20:]
        vol = rets.std() * np.sqrt(252)
        inv_vol = 1.0 / (vol + 1e-6)
        inv_vol = (inv_vol - inv_vol.min()) / (inv_vol.max() - inv_vol.min() + 1e-8)

        def _rank(s: pd.Series) -> pd.Series:
            return s.rank(pct=True).fillna(0.5)

        composite = (
            self.weights["mom_1m"]   * _rank(mom_1m)
            + self.weights["mom_3m"]   * _rank(mom_3m)
            + self.weights["mom_12m1"] * _rank(mom_12m1)
            + self.weights["inv_vol"]  * inv_vol
        )
        return composite.sort_values(ascending=False)

    def _sector_constrained_select(self, scores: pd.Series) -> list[str]:
        selected: set[str] = set()
        ordered: list[str] = []

        # Pass 1: satisfy sector minimums from best available per sector
        for sector, tickers in UNIVERSE_BY_SECTOR.items():
            sector_scores = scores[[t for t in tickers if t in scores.index]]
            for ticker in sector_scores.index:
                if len(ordered) >= self.top_n:
                    break
                if ticker not in selected and len([t for t in ordered if SECTOR_MAP.get(t) == sector]) < self.min_per_sector:
                    selected.add(ticker)
                    ordered.append(ticker)

        # Pass 2: fill remaining slots from overall top scores
        for ticker in scores.index:
            if len(ordered) >= self.top_n:
                break
            if ticker not in selected:
                selected.add(ticker)
                ordered.append(ticker)

        # Return sorted by descending composite score
        ordered.sort(key=lambda t: -scores.get(t, 0.0))
        return ordered[: self.top_n]
