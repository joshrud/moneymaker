"""Structured JSONL trade logger. One file per bot per day in logs/."""
from __future__ import annotations
import json
import os
from datetime import date, datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class TradeLogger:
    """Appends trade events to logs/YYYY-MM-DD_<bot_name>.jsonl."""

    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        LOGS_DIR.mkdir(exist_ok=True)

    def _log_path(self) -> Path:
        today = date.today().isoformat()
        return LOGS_DIR / f"{today}_{self.bot_name}.jsonl"

    def log(self, event_type: str, **kwargs):
        """Writes one JSON line. event_type examples: 'trade', 'signal', 'error', 'summary'."""
        record = {
            "ts": datetime.utcnow().isoformat(),
            "bot": self.bot_name,
            "event": event_type,
            **kwargs,
        }
        with open(self._log_path(), "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_trade(self, side: str, symbol: str, qty: float, price: float, reason: str = ""):
        self.log("trade", side=side, symbol=symbol, qty=round(qty, 4),
                 price=round(price, 4), reason=reason)

    def log_signal(self, symbol: str, signal: float, reason: str = ""):
        self.log("signal", symbol=symbol, signal=round(signal, 4), reason=reason)

    def log_summary(self, portfolio_value: float, daily_pnl: float, **metrics):
        self.log("summary", portfolio_value=round(portfolio_value, 2),
                 daily_pnl=round(daily_pnl, 2), **metrics)

    def read_today(self) -> list[dict]:
        path = self._log_path()
        if not path.exists():
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def read_date(self, dt: date) -> list[dict]:
        path = LOGS_DIR / f"{dt.isoformat()}_{self.bot_name}.jsonl"
        if not path.exists():
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
