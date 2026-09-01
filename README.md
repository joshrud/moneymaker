# Moneymaker

Algorithmic trading system running 15 competing bots in parallel on Alpaca paper trading. Each bot uses a different strategy drawn from `algo_trading_summary.md` and `rl_algorithms_summary.md`. All bots trade a virtual $20K allocation daily and are ranked each evening by PnL, Sharpe, and drawdown in a generated markdown report.

## Bots

| Bot | Strategy |
|-----|----------|
| **Bot 1 — Momentum** | Classical RSI + MACD + Bollinger Bands rules. No ML. Baseline benchmark. |
| **Bot 2 — SAC RL** | Soft Actor-Critic allocates continuous portfolio weights across 5 assets. Trained offline on 2 years of daily bars; reward = incremental Sharpe minus transaction costs. |
| **Bot 3 — Claude Sentiment** | Fetches pre-market news via Alpaca, sends headlines to Claude Haiku, gets a per-symbol conviction score in `[-1, +1]`, then confirms with RSI/MACD before sizing positions. |
| **Bot 4 — FinBERT + PPO** | Pairs mean-reversion (AAPL/MSFT, NVDA/AMD, AMZN/GOOGL). PPO agent learns optimal spread entry/exit thresholds. FinBERT news filter skips pairs with fundamental divergence. |
| **Bot 5 — Ensemble** | Aggregates signals from Bots 1–4, weighted by each bot's rolling 5-day Sharpe. The only bot that places **real Alpaca paper orders**. |
| **Bot 6 — Triple EMA Trend** | Bull when price > EMA8 > EMA21 > EMA55 AND OBV trending up. More active than Bot 1 — fires on partial EMA alignment without needing deep oversold. |
| **Bot 7 — TD3 Pairs Stat Arb** | TD3 trading spreads across 10 co-integrated pairs (financials, energy, consumer staples). Continuous position sizing via spread z-score. |
| **Bot 8 — SAC Sortino** | Same two-stage architecture as Bot 2 (StockSelector → SAC) but trained with Sortino ratio reward to penalize downside volatility more aggressively. |
| **Bot 9 — DQN VWAP Reversion** | DQN with 5 discrete actions trading when price deviates significantly from rolling VWAP (approximated as 5-day EMA of typical price weighted by volume). |
| **Bot 10 — LightGBM Factor** | Supervised gradient-boosted trees trained on momentum + volatility factors to predict next-day return quintile. Fast, interpretable, no environment simulation. |
| **Bot 11 — Regime-Switching SAC** | Two SAC models: one optimised for trending conditions (ADX > 25), one for ranging markets. A regime classifier selects the active model each cycle. |
| **Bot 12 — Covered Call Writer** | When any bot is long a stock with >1% unrealized gain, sells a 30–45 DTE covered call at the nearest OTM strike to generate premium income. |
| **Bot 13 — Cash-Secured Put Seller** | For stocks >3% below their 20-day MA, sells a 21–30 DTE cash-secured put at a strike ~5% below current price as a value entry with premium cushion. |
| **Bot 14 — Deep Hedging RL** | Simplified Deep Hedging (Bühlmann et al. 2019). SAC agent learns to delta-hedge a synthetic options book, minimizing CVaR of the net P&L distribution. |
| **Bot 15 — Aggressive Ensemble** | Sharpe-weighted ensemble of Bots 6–14, mirroring Bot 5's architecture but drawing on the expanded bot set. Places real Alpaca paper orders. |

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
bots/        15 bot implementations + abstract base class
envs/        Gym environments for RL training (TradingEnv for SAC/TD3/DQN, PairsTradingEnv for PPO)
models/      Saved RL model checkpoints (.zip)
logs/        YYYY-MM-DD_<bot>.jsonl trade logs written each cycle
reports/     YYYY-MM-DD_report.md daily comparison reports (all 15 bots ranked)
scripts/     run_all_bots.py · train_rl_models.py · daily_report.py
deploy/      CodeDeploy lifecycle hooks (stop/install/start/validate) + systemd unit files
tests/       test_core.py · test_trading_env.py · test_bots.py
.github/     GitHub Actions workflow — push to prod triggers test → deploy pipeline
```

## Deployment

Every push to the `prod` branch automatically deploys to the EC2 production server via GitHub Actions → AWS CodeDeploy:

1. **Test** — `pytest tests/ -v` runs in CI; deploy is blocked on failure.
2. **Package** — repo zipped (excluding `.env`, `.venv/`, `models/`, `logs/`, `reports/`) and uploaded to S3.
3. **Deploy** — CodeDeploy lifecycle:
   - `BeforeInstall` (`stop.sh`) — stops services, clears code dirs so CodeDeploy can write cleanly.
   - `AfterInstall` (`install.sh`) — creates `moneymaker` system user if absent, creates `.venv` if absent, sets ownership, `pip install -r requirements.txt`.
   - `ApplicationStart` (`start.sh`) — copies systemd units from repo to `/etc/systemd/system/`, reloads daemon, starts `moneymaker.service` and `moneymaker-train.timer`.
   - `ValidateService` (`validate.sh`) — confirms `moneymaker.service` is active; dumps `journalctl` on failure.

The `.env` file (API keys) is **never committed** and must be manually placed at `/opt/moneymaker/.env` on a fresh instance. RL model checkpoints in `models/` are preserved across deploys; bots retrain automatically if a checkpoint is missing.

AWS credentials use GitHub OIDC → IAM role (no stored static secrets).

## Risk Controls

All bots share the same `RiskManager`: ATR-based position sizing (risk 1% of capital per trade), 20% max single-position cap, and a 3% daily-loss circuit breaker that halts trading for the day.
