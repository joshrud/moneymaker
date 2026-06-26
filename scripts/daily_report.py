"""
Standalone daily report generator. Safe to run at any time; reads only log files.
Cron target at 4:05 PM ET every trading day:
  5 16 * * 1-5 cd /path/to/moneymaker && .venv/bin/python scripts/daily_report.py

Usage:
  .venv/bin/python scripts/daily_report.py
  .venv/bin/python scripts/daily_report.py --date 2026-06-25
"""
from __future__ import annotations
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reporter import generate_report


def main():
    parser = argparse.ArgumentParser(description="Generate daily bot comparison report")
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD format")
    args = parser.parse_args()

    dt = date.fromisoformat(args.date) if args.date else date.today()
    path = generate_report(dt)
    print(f"Report written: {path}")


if __name__ == "__main__":
    main()
