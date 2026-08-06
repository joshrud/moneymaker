"""Manages data/dynamic_watchlist.json — symbols discovered by the small-cap screener."""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "data" / "dynamic_watchlist.json"


def _load() -> dict:
    if not _PATH.exists():
        return {}
    return json.loads(_PATH.read_text())


def _save(entries: dict):
    _PATH.parent.mkdir(exist_ok=True)
    _PATH.write_text(json.dumps(entries, indent=2, sort_keys=True))


def active_symbols() -> list[str]:
    """Returns symbols whose status is 'active'."""
    return [s for s, v in _load().items() if v.get("status") == "active"]


def all_entries() -> dict:
    return _load()


def add_symbols(candidates: list[dict]):
    """Upserts symbols from screener output. Each dict must have 'symbol' and 'score'."""
    entries = _load()
    today = date.today().isoformat()
    for c in candidates:
        sym = c["symbol"]
        entries[sym] = {
            "added": entries.get(sym, {}).get("added", today),
            "last_checked": today,
            "score": c["score"],
            "price": c.get("price"),
            "avg_volume": c.get("avg_volume"),
            "momentum_20d": c.get("momentum_20d"),
            "status": "active",
        }
    _save(entries)


def update_entry(symbol: str, **kwargs):
    """Merges kwargs into an existing entry (e.g. to update score after weekly review)."""
    entries = _load()
    if symbol in entries:
        entries[symbol].update(kwargs)
        _save(entries)


def deactivate(symbol: str, reason: str = ""):
    """Marks a symbol inactive so bots stop monitoring it."""
    entries = _load()
    if symbol in entries:
        entries[symbol]["status"] = "inactive"
        entries[symbol]["deactivated"] = date.today().isoformat()
        if reason:
            entries[symbol]["deactivate_reason"] = reason
        _save(entries)
