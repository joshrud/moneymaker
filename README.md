# Moneymaker

Algorithmic trading system running 5 competing bots in parallel on Alpaca paper trading. Each bot uses a different strategy drawn from `algo_trading_summary.md` and `rl_algorithms_summary.md`. All bots trade a virtual $20K allocation daily and are ranked each evening by PnL, Sharpe, and drawdown in a generated markdown report.

## Bots

| Bot | Strategy |
|-----|----------|
| **Bot 1 — Momentum** | Classical RSI + MACD + Bollinger Bands rules. No ML. Baseline benchmark. |
| **Bot 2 — SAC RL** | Soft Actor-Critic allocates continuous portfolio weights across 5 assets. Trained offline on 2 years of daily bars; reward = incremental Sharpe minus transaction costs. |
| **Bot 3 — Claude Sentiment** | Fetches pre-market news via Alpaca, sends headlines to Claude Haiku, gets a per-symbol conviction score in `[-1, +1]`, then confirms with RSI/MACD before sizing positions. |
| **Bot 4 — FinBERT + PPO** | Pairs mean-reversion (AAPL/MSFT, NVDA/AMD, AMZN/GOOGL). PPO agent learns optimal spread entry/exit thresholds. FinBERT news filter skips pairs with fundamental divergence. |
| **Bot 5 — Ensemble** | Aggregates signals from Bots 1–4, weighted by each bot's rolling 5-day Sharpe. The only bot that places **real Alpaca paper orders**. |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Add credentials to `.env`:
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ANTHROPIC_API_KEY=...
TRADING_MODE=paper
```

## Usage

```bash
# Pre-train RL models (run once before market open):
.venv/bin/python scripts/train_rl_models.py

# Run once per cycle (cron: 35 9 * * 1-5):
.venv/bin/python scripts/run_all_bots.py

# Loop every 5 min during market hours:
.venv/bin/python scripts/run_all_bots.py --loop

# Generate report manually:
.venv/bin/python scripts/daily_report.py
```

## Tests

```bash
.venv/bin/pytest tests/ -v
```

38 tests covering core utilities (indicators, portfolio, risk, logger), Gym environments, and bot signal logic.

## Structure

```
core/        Shared utilities — Alpaca client, data feed, indicators, risk, portfolio, sentiment, logger, reporter
bots/        5 bot implementations + abstract base class
envs/        Gym environments for RL training (TradingEnv for SAC, PairsTradingEnv for PPO)
models/      Saved RL model checkpoints (.zip)
logs/        YYYY-MM-DD_<bot>.jsonl trade logs written each cycle
reports/     YYYY-MM-DD_report.md daily comparison reports
scripts/     run_all_bots.py · train_rl_models.py · daily_report.py
tests/       test_core.py · test_trading_env.py · test_bots.py
```

## Risk Controls

All bots share the same `RiskManager`: ATR-based position sizing (risk 1% of capital per trade), 20% max single-position cap, and a 3% daily-loss circuit breaker that halts trading for the day.
# deployment pipeline test 2026-09-01T16:57:09Z
