"""
Bot 5 — Sharpe-Weighted Ensemble Meta-Bot
Strategy: Aggregates signals from Bots 1–4, weighted by each bot's rolling 5-day Sharpe ratio.
When the ensemble reaches high-conviction consensus, it also places REAL Alpaca paper trades.
This is the only bot that submits actual orders to the paper account.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd

from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed, WATCHLIST
from core.logger import TradeLogger, LOGS_DIR
from datetime import date, timedelta


CONSENSUS_THRESHOLD = 0.35   # minimum weighted signal to place a real order
MIN_WEIGHT = 0.05             # floor weight for bots with no history (equal-weight fallback)
REAL_TRADE_MAX_PCT = 0.10     # max 10% of paper account per real trade


class EnsembleBot(BaseBot):
    """
    Combines signals from the other 4 bots using Sharpe-weighted voting.
    Also executes real Alpaca paper trades when consensus is strong enough.
    """

    def __init__(
        self,
        client: AlpacaClient,
        feed: DataFeed,
        sub_bots: list,   # [MomentumBot, SACBot, ClaudeSentimentBot, FinBERTPPOBot]
    ):
        super().__init__("bot5_ensemble", client, feed)
        self.sub_bots = sub_bots
        self.watchlist = WATCHLIST

    # ── Signal aggregation ────────────────────────────────────────────────────

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        """
        Collects signals from each sub-bot, weights by rolling Sharpe, averages.
        """
        weights = self._compute_bot_weights()
        all_signals: list[Dict[str, float]] = []

        for bot in self.sub_bots:
            try:
                sigs = bot.generate_signals(bars_df)
                all_signals.append(sigs)
            except Exception as e:
                self.logger.log("sub_bot_error", bot=bot.name, error=str(e))
                all_signals.append({})

        # Weighted average per symbol
        combined: Dict[str, float] = {}
        for i, sym in enumerate(self.watchlist):
            total_weight = 0.0
            weighted_signal = 0.0
            for bot_idx, sigs in enumerate(all_signals):
                sig = sigs.get(sym, 0.0)
                w = weights[bot_idx]
                weighted_signal += sig * w
                total_weight += w
            combined[sym] = weighted_signal / max(total_weight, 1e-8)
            self.logger.log_signal(sym, combined[sym],
                                   reason=f"ensemble,weights={[round(w,3) for w in weights]}")

        return combined

    def run_once(self) -> dict:
        """
        Extends base run_once: after virtual trades, also places real Alpaca orders
        when the ensemble signal exceeds CONSENSUS_THRESHOLD.
        """
        bars_df = self.feed.daily_bars()
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
            # Virtual trade (tracked in portfolio)
            self._execute_signal(symbol, signal, price, portfolio_val)
            # Real Alpaca paper trade for high-conviction signals
            if abs(signal) >= CONSENSUS_THRESHOLD:
                self._execute_real_trade(symbol, signal, price)

        return self._end_of_cycle(prices)

    # ── Real trade execution ──────────────────────────────────────────────────

    def _execute_real_trade(self, symbol: str, signal: float, price: float):
        """Places a real market order on the Alpaca paper account."""
        try:
            account = self.client.get_account()
            buying_power = float(account.buying_power)
            alloc = buying_power * REAL_TRADE_MAX_PCT
            qty = round(alloc / price, 2)
            if qty < 0.01:
                return

            if signal > 0:
                order = self.client.market_buy(symbol, qty)
                self.logger.log("real_trade", side="buy", symbol=symbol, qty=qty,
                                price=price, order_id=str(order.id),
                                reason=f"ensemble_signal={signal:.3f}")
            elif signal < 0:
                # Only sell if we hold a position
                positions = self.client.get_positions()
                if symbol in positions:
                    order = self.client.close_position(symbol)
                    self.logger.log("real_trade", side="sell", symbol=symbol,
                                    order_id=str(order.id),
                                    reason=f"ensemble_signal={signal:.3f}")
        except Exception as e:
            self.logger.log("real_trade_error", symbol=symbol, error=str(e))

    # ── Bot weight calculation ─────────────────────────────────────────────────

    def _compute_bot_weights(self) -> list[float]:
        """
        Weights each sub-bot by its rolling 5-day Sharpe from log files.
        Falls back to equal weights if insufficient history.
        """
        sharpes = []
        for bot in self.sub_bots:
            sharpe = self._rolling_sharpe(bot.name, days=5)
            sharpes.append(max(sharpe, MIN_WEIGHT))

        total = sum(sharpes)
        return [s / total for s in sharpes]

    def _rolling_sharpe(self, bot_name: str, days: int = 5) -> float:
        """Reads recent log files to compute a bot's recent daily Sharpe."""
        daily_pnls = []
        for i in range(days):
            dt = date.today() - timedelta(days=i + 1)
            path = LOGS_DIR / f"{dt.isoformat()}_{bot_name}.jsonl"
            if not path.exists():
                continue
            import json
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if rec.get("event") == "summary":
                            daily_pnls.append(rec.get("daily_pnl", 0.0))
                            break
                    except json.JSONDecodeError:
                        continue

        if len(daily_pnls) < 2:
            return MIN_WEIGHT
        arr = np.array(daily_pnls)
        std = arr.std()
        return float(arr.mean() / std * np.sqrt(252)) if std > 0 else MIN_WEIGHT
