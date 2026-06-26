"""Fetches and caches OHLCV bars. All bots call this once per run cycle."""
from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.timeframe import TimeFrame

from core.alpaca_client import AlpacaClient

NY = ZoneInfo("America/New_York")

# Default universe: liquid large-caps across sectors + semis for pairs
WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL", "JPM", "V"]

# Pairs for mean-reversion bot (co-integrated pairs)
PAIRS = [("AAPL", "MSFT"), ("NVDA", "AMD"), ("AMZN", "GOOGL")]

_BATCH_SIZE = 50  # max symbols per Alpaca request to stay within rate limits


class DataFeed:
    """
    Fetches historical and latest bars from Alpaca.
    Results are cached per day to minimise API calls across bots.
    """

    def __init__(self, client: AlpacaClient):
        self.client = client
        self._cache: dict[str, pd.DataFrame] = {}  # key = f"{symbols}_{lookback}"

    def daily_bars(self, symbols: list = None, lookback: int = 60) -> pd.DataFrame:
        """
        Returns daily OHLCV bars for the given symbols over `lookback` trading days.
        Result is a DataFrame with MultiIndex (symbol, timestamp).
        """
        symbols = symbols or WATCHLIST
        cache_key = f"{'_'.join(sorted(symbols))}_{lookback}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        end = datetime.now(NY).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=lookback * 2)  # fetch extra to account for weekends
        # limit is total rows across all symbols, so scale by symbol count
        df = self.client.get_bars(symbols, TimeFrame.Day, start=start, end=end,
                                  limit=lookback * len(symbols))
        self._cache[cache_key] = df
        return df

    def universe_bars(self, lookback: int = 252) -> pd.DataFrame:
        """
        Fetches daily bars for the full 261-stock UNIVERSE in batches of 50.
        Returns a (symbol, timestamp) MultiIndex DataFrame of OHLCV data.
        Cached for the session; call clear_cache() to refresh.
        """
        from core.universe import UNIVERSE

        cache_key = f"universe_{lookback}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        end = datetime.now(NY).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=lookback * 2)

        chunks = [
            UNIVERSE[i: i + _BATCH_SIZE]
            for i in range(0, len(UNIVERSE), _BATCH_SIZE)
        ]
        frames = []
        for chunk in chunks:
            df = self.client.get_bars(
                chunk, TimeFrame.Day, start=start, end=end,
                limit=lookback * len(chunk),
            )
            if not df.empty:
                frames.append(df)

        result = pd.concat(frames) if frames else pd.DataFrame()
        self._cache[cache_key] = result
        return result

    def pivot_close(self, symbols: list = None, lookback: int = 60) -> pd.DataFrame:
        """Returns a (date × symbol) DataFrame of closing prices."""
        bars = self.daily_bars(symbols, lookback)
        if bars.empty:
            return pd.DataFrame()
        return bars["close"].unstack(level=0)

    def pivot_returns(self, symbols: list = None, lookback: int = 60) -> pd.DataFrame:
        """Returns a (date × symbol) DataFrame of daily log returns."""
        closes = self.pivot_close(symbols, lookback)
        import numpy as np
        return np.log(closes / closes.shift(1)).dropna()

    def latest_prices(self, symbols: list = None) -> dict:
        """Returns {symbol: latest_ask_price} for quick signal generation."""
        symbols = symbols or WATCHLIST
        quotes = self.client.get_latest_quotes(symbols)
        return {sym: float(q.ask_price) for sym, q in quotes.items()}

    def get_news(self, symbols: list = None, limit: int = 20) -> list:
        symbols = symbols or WATCHLIST
        return self.client.get_news(symbols, limit=limit)

    def clear_cache(self):
        self._cache.clear()
