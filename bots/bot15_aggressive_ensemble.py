"""
Bot 15 — Aggressive Ensemble (Bots 6–14)
Same Sharpe-weighted architecture as bot5 but:
  - Lower threshold: 0.15 (vs bot5's 0.20)
  - Larger position per trade: 15% of buying power (vs 10%)
  - Lower min-weight floor: 0.02 (vs 0.05) — lets strong bots dominate weighting
  - Covers union of all symbols from sub-bots (not just WATCHLIST)
"""
from __future__ import annotations
from typing import Dict
from datetime import date, timedelta
import numpy as np
import pandas as pd
import json
from bots.base_bot import BaseBot
from core.alpaca_client import AlpacaClient
from core.data_feed import DataFeed
from core.logger import TradeLogger, LOGS_DIR

CONSENSUS_THRESHOLD = 0.15
MIN_WEIGHT = 0.02
REAL_TRADE_MAX_PCT = 0.15
MAX_POSITION_PCT = 0.30


class AggressiveEnsembleBot(BaseBot):
    def __init__(self, client: AlpacaClient, feed: DataFeed, sub_bots: list):
        super().__init__("bot15_aggressive_ensemble", client, feed)
        self.sub_bots = sub_bots

    def generate_signals(self, bars_df: pd.DataFrame) -> Dict[str, float]:
        weights = self._compute_bot_weights()
        all_signals: list[Dict[str, float]] = []
        for bot in self.sub_bots:
            try:
                sigs = bot.generate_signals(bars_df)
                all_signals.append(sigs)
            except Exception as e:
                self.logger.log("sub_bot_error", bot=bot.name, error=str(e))
                all_signals.append({})

        # Collect union of all symbols
        all_syms = set()
        for sigs in all_signals:
            all_syms.update(sigs.keys())

        combined: Dict[str, float] = {}
        for sym in all_syms:
            total_weight = 0.0
            weighted_signal = 0.0
            for bot_idx, sigs in enumerate(all_signals):
                sig = sigs.get(sym, 0.0)
                w = weights[bot_idx]
                weighted_signal += sig * w
                total_weight += w
            combined[sym] = weighted_signal / max(total_weight, 1e-8)
            self.logger.log_signal(sym, combined[sym],
                                   reason=f"agg_ensemble,w={[round(w,3) for w in weights]}")
        return combined

    def run_once(self) -> dict:
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
            self._execute_signal(symbol, signal, price, portfolio_val)
            if abs(signal) >= CONSENSUS_THRESHOLD:
                self._execute_real_trade(symbol, signal, price)
        return self._end_of_cycle(prices)

    def _execute_real_trade(self, symbol: str, signal: float, price: float):
        try:
            account = self.client.get_account()
            portfolio_value = float(account.equity)
            buying_power = float(account.buying_power)
            if signal > 0:
                positions = self.client.get_positions()
                if symbol in positions:
                    current_market_val = float(positions[symbol].market_value)
                    if current_market_val / max(portfolio_value, 1) >= MAX_POSITION_PCT:
                        return
                alloc = buying_power * REAL_TRADE_MAX_PCT
                qty = round(alloc / price, 2)
                if qty < 0.01:
                    return
                order = self.client.market_buy(symbol, qty)
                self.logger.log("real_trade", side="buy", symbol=symbol, qty=qty,
                                price=price, order_id=str(order.id),
                                reason=f"agg_signal={signal:.3f}")
            elif signal < 0:
                positions = self.client.get_positions()
                if symbol in positions:
                    order = self.client.close_position(symbol)
                    self.logger.log("real_trade", side="sell", symbol=symbol,
                                    order_id=str(order.id), reason=f"agg_signal={signal:.3f}")
        except Exception as e:
            self.logger.log("real_trade_error", symbol=symbol, error=str(e))

    def _compute_bot_weights(self) -> list[float]:
        sharpes = []
        for bot in self.sub_bots:
            sharpe = self._rolling_sharpe(bot.name, days=5)
            sharpes.append(max(sharpe, MIN_WEIGHT))
        total = sum(sharpes)
        return [s / total for s in sharpes]

    def _rolling_sharpe(self, bot_name: str, days: int = 5) -> float:
        daily_pnls = []
        for i in range(days):
            dt = date.today() - timedelta(days=i + 1)
            path = LOGS_DIR / f"{dt.isoformat()}_{bot_name}.jsonl"
            if not path.exists():
                continue
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
