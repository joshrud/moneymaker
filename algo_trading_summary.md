# Algorithmic Trading Research Summary
### Foundational Research for Building an Algo Trading Bot

**Research date:** June 2026  
**Scope:** 28 major open-source algorithmic trading repositories + brokerage API analysis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Best APIs / Brokerages Comparison](#2-best-apis--brokerages-comparison)
3. [Top Recommendation](#3-top-recommendation)
4. [Most Successful Features](#4-most-successful-features)
5. [Architecture Patterns](#5-architecture-patterns)
6. [Repo Rankings - Top 10](#6-repo-rankings---top-10)
7. [Technology Stack Recommendations](#7-technology-stack-recommendations)
8. [Key Learnings / Gotchas](#8-key-learnings--gotchas)
9. [Next Steps / Roadmap](#9-next-steps--roadmap)

---

## 1. Executive Summary

After analyzing 28 top algorithmic trading repositories (combined ~500k+ GitHub stars), several clear patterns emerge in what makes successful algo trading bots:

### Common Patterns in Successful Bots

**1. Event-Driven Architecture is Non-Negotiable**  
Every production-grade system (freqtrade, NautilusTrader, backtrader, Lean, vnpy) uses an event-driven pattern. This decouples data ingestion, signal generation, and order execution — enabling accurate backtesting that mirrors live trading.

**2. Same-Code Backtest-to-Live is the Gold Standard**  
The best frameworks (freqtrade, NautilusTrader, lumibot, blankly) allow identical strategy code to run in backtest, paper trading, and live modes. This eliminates a major source of bugs: strategy implementations that differ between test and live.

**3. Backtesting is Table Stakes**  
100% of production-ready frameworks include backtesting. The quality varies dramatically — from simple vectorized backtesting (fast but inaccurate) to event-driven simulation with order book modeling, slippage, and realistic fill simulation.

**4. ML/AI Integration is Increasingly Standard**  
The top repos now integrate some form of ML — from adaptive ML in freqtrade's FreqAI, to deep reinforcement learning in FinRL, to full LLM integration in lumibot and OctoBot. Sentiment analysis via FinBERT or GPT appears in 4–5 major frameworks.

**5. Multi-Exchange Support via CCXT**  
CCXT (42k stars) has become the de-facto standard for crypto exchange connectivity. Most crypto-focused bots (freqtrade, jesse, OctoBot) use it under the hood, giving access to 100+ exchanges with a unified API.

**6. Risk Management is Built-In, Not Bolted On**  
Successful frameworks expose risk controls as first-class citizens: position limits, max drawdown stops, daily loss limits, and circuit breakers appear in every production system.

**7. LLM Integration is Emerging, Not Yet Mature**  
Only ~5 repos use LLMs as a core feature (lumibot, OctoBot, FinGPT, jesse's JesseGPT, qlib's RD-Agent). The pattern is LLMs as *reasoning layer* over market data + news rather than as signal generators directly.

**8. Sentiment Analysis Has Real Signal (When Done Right)**  
Repos using sentiment (FinGPT, ML-for-Trading, OctoBot) focus on news/earnings transcripts via FinBERT or GPT rather than raw social media, which tends to be too noisy.

---

## 2. Best APIs / Brokerages Comparison

### Stocks & Multi-Asset Brokerages

| Broker | Ease of Use | Docs Quality | Paper Trading | Commission | Asset Coverage | Python SDK | Best For |
|--------|-------------|--------------|---------------|------------|----------------|------------|----------|
| **Alpaca** | ★★★★★ | ★★★★★ | ★★★★★ | $0 stocks/ETF, $0.25/contract options, low crypto | Stocks, ETFs, Crypto, Options | `alpaca-py` (excellent, async) | **Best starting point for new builders** |
| **Interactive Brokers** | ★★★ | ★★★★ | ★★★★ | $0.005/share min $1, tiered available | Stocks, Options, Futures, Forex, Bonds, 150+ markets | `ib_insync` / `ib_async` (good) | Professional multi-asset strategies |
| **TD Ameritrade / Schwab** | ★★★★ | ★★★★ | ★★★ | $0 stocks, $0.65/contract options | Stocks, Options, Futures, Forex | `schwab-py` (community) | Options-heavy strategies |
| **Tradier** | ★★★★ | ★★★★ | ★★★★ | $0.35/contract options | Stocks, Options | `tradier` (community) | Options strategies, best option rates |

### Cryptocurrency Exchanges

| Exchange | Ease of Use | Docs Quality | Paper/Testnet | Commission | Python SDK | Volume | Best For |
|----------|-------------|--------------|---------------|------------|------------|--------|----------|
| **Binance** | ★★★★ | ★★★★ | ★★★★★ (testnet) | 0.1% spot, 0.02/0.05% futures | `python-binance` (7.2k stars) | #1 global | Crypto, HFT, futures |
| **Coinbase** | ★★★★ | ★★★★ | ★★★ | 0.4%/0.6% maker/taker | Official Python SDK | #1 US | US-regulated crypto |
| **Kraken** | ★★★★ | ★★★★★ | ★★★ | 0.16/0.26% maker/taker | `python-kraken-sdk` (official) | #3 US | Clean API, stable |
| **Bybit** | ★★★ | ★★★★ | ★★★★ | 0.1% spot, 0.02/0.055% futures | `pybit` | Growing | Derivatives, leverage |
| **OKX** | ★★★ | ★★★ | ★★★★ | 0.08/0.1% maker/taker | `python-okx` (official) | Global top 5 | Diversification |

### Key API Feature Details

**Alpaca (Recommended Entry Point)**
- Commission-free stock/ETF trading
- Free paper trading with up to 3 paper accounts
- Exact same API endpoints for paper vs. live (just change base URL)
- WebSocket market data streaming ~30–50ms latency
- `alpaca-py` SDK: async/await, Pydantic validation, type hints throughout
- Data: 5000+ US stocks, 20+ crypto pairs, options
- No minimum balance requirement
- MCP server available for AI assistant integration
- Free IEX market data included

**Interactive Brokers (Best for Production / Multi-Asset)**
- Access to 150+ markets in 34 countries
- 150+ order types
- Sub-50ms average execution latency
- Requires TWS or IB Gateway desktop app running
- `ib_insync` reduces boilerplate by ~70% vs. native TWS API
- Paper trading is functionally identical to live (same API)
- IBKR Lite: commission-free US stocks/ETFs
- IBKR Pro: tiered pricing, better for high-volume algo trading

**Binance (Best for Crypto)**
- 1,200 req/min rate limit for order endpoints
- Testnet available for spot, futures, options
- Comprehensive WebSocket API for all products
- `python-binance` (7.2k stars) is the community standard
- Also accessible via CCXT for unified multi-exchange code

---

## 3. Top Recommendation

### For Building a New Bot: Start with Alpaca + freqtrade or lumibot

**If building a stock/multi-asset bot:**  
Use **Alpaca** as your brokerage + **lumibot** as your framework.
- Lumibot's "same code for backtest and live" eliminates strategy drift
- Alpaca provides free paper trading, zero commissions, excellent SDK
- Lumibot natively supports Alpaca as its primary broker
- LLM agent runtime built-in for AI-assisted strategy development
- SEC filing and FRED macro data integration out of the box

**If building a crypto bot:**  
Use **freqtrade** + **Binance/Bybit**.
- 51k+ stars, massive community, 5+ years production-proven
- FreqAI adaptive ML built-in — no separate ML pipeline needed
- CCXT gives you access to 100+ exchanges if you want to diversify
- Excellent documentation and active Discord community

**If you want maximum flexibility from day one:**  
Use **NautilusTrader**.
- Supports both stocks (via Interactive Brokers) and crypto (25+ venues)
- Rust core = production-level performance
- Research-to-live parity eliminates rewrite risk
- Steeper learning curve but the most robust architecture

**Verdict: Alpaca is the best starting brokerage for a new builder because:**
1. Zero commissions eliminate a major variable when testing strategy profitability
2. Free paper trading with the exact same API as live
3. Best Python SDK (`alpaca-py`) in terms of developer experience
4. No minimum balance — start testing immediately
5. Stocks + crypto + options in one API

---

## 4. Most Successful Features

### Backtesting Frameworks

| Framework | Speed | Accuracy | Ease of Use | Notes |
|-----------|-------|----------|-------------|-------|
| NautilusTrader | ★★★★★ | ★★★★★ | ★★★ | Nanosecond resolution, L2 order book |
| freqtrade | ★★★★ | ★★★★ | ★★★★★ | Best for crypto, highly configurable |
| backtrader | ★★★ | ★★★★ | ★★★★ | Gold standard for stocks, no longer maintained |
| lumibot | ★★★ | ★★★★ | ★★★★★ | Best for ease of use |
| Lean (QuantConnect) | ★★★★ | ★★★★★ | ★★★ | Most accurate for stocks/options |
| zipline-reloaded | ★★★ | ★★★★ | ★★★★ | Best pandas integration |

### Technical Indicators Most Commonly Used

Based on prevalence across repositories:

**Trend Following (most common):**
- SMA/EMA crossovers (used in nearly every framework)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Ichimoku Cloud

**Mean Reversion:**
- Bollinger Bands (appear in 15+ repos)
- RSI (Relative Strength Index) — most universal indicator
- Stochastic Oscillator
- Z-score of price relative to rolling mean

**Volume-Based:**
- Volume-weighted moving averages
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price) — critical for institutional strategies

**Volatility:**
- ATR (Average True Range) — used for position sizing in most professional systems
- Realized vs. implied volatility spreads

**Most used technical library:** `TA-Lib` (12k stars, C library with Python wrapper) — comprehensive, fast, used by freqtrade, backtrader, Jesse, and most others.

### Risk Management Patterns

The most successful systems implement risk management at **multiple levels**:

```python
# Example layered risk management structure
class RiskManager:
    # Level 1: Per-trade risk
    max_position_size = 0.02      # 2% of portfolio per trade
    stop_loss_pct = 0.02          # 2% stop loss per position
    take_profit_pct = 0.04        # 4% take profit (2:1 R/R minimum)
    
    # Level 2: Strategy-level risk
    max_open_positions = 5        # Max concurrent positions
    max_sector_exposure = 0.20    # 20% max in any sector
    
    # Level 3: Portfolio-level risk
    max_daily_loss = 0.05         # 5% daily loss circuit breaker
    max_drawdown = 0.15           # 15% max drawdown → halt trading
    
    # Level 4: Market regime
    vix_threshold = 30            # Reduce position size when VIX > 30
    correlation_threshold = 0.8   # Don't open correlated positions
```

**Key risk management patterns observed:**
1. **ATR-based position sizing** — size positions by volatility, not fixed $ amounts
2. **Kelly Criterion** (and fractional Kelly) — optimal bet sizing based on edge
3. **Maximum drawdown circuit breaker** — auto-halt at configurable drawdown
4. **Portfolio heat limits** — cap total portfolio risk at 10–15% at any given time
5. **Correlation limits** — avoid opening positions that move together

### Position Sizing Strategies

In order of sophistication:
1. **Fixed fraction** (e.g., 2% per trade) — simplest, widely used
2. **Volatility-adjusted** (position size ∝ 1/ATR) — keeps dollar risk constant
3. **Kelly Criterion** (edge × odds) — theoretically optimal, use fractional (0.25×) in practice
4. **Risk parity** — size so each position contributes equal risk to portfolio

### LLM Integration Patterns

The repos that use LLMs do so in three distinct patterns:

**Pattern 1: LLM as Reasoning Layer (lumibot, OctoBot)**
```
Market data + News → LLM prompt → Trade decision
```
The LLM receives structured context (price, technicals, recent news) and outputs a trade signal with reasoning. Pros: interpretable, catches regime changes. Cons: slow (~1–5s), expensive, inconsistent.

**Pattern 2: LLM for Sentiment Scoring (FinGPT, ML-for-Trading)**
```
Financial news/filings → FinBERT/GPT fine-tuned → Sentiment score → Signal
```
Fine-tune a small financial LLM specifically for sentiment classification. More consistent than general GPT, faster inference, replicable. FinBERT is the gold standard for this.

**Pattern 3: LLM for Strategy Development (Jesse's JesseGPT, qlib RD-Agent)**
```
Trader describes desired strategy → LLM generates code → Backtest → Refine
```
Use LLM as a coding assistant that generates Python strategy code. Best ROI for developers.

**Practical recommendation:** Start with **Pattern 2 (FinBERT sentiment)** if you want sentiment signal. Use **Pattern 3 (strategy generation)** as a development accelerator. Avoid Pattern 1 in production until latency improves.

### Sentiment Analysis Approaches

| Source | Signal Quality | Latency | Cost | Implementation |
|--------|---------------|---------|------|----------------|
| Financial news (Reuters, Bloomberg) | ★★★★★ | Minutes | $$$ | Alpaca News API, Polygon News |
| SEC filings (10-K, 10-Q, 8-K) | ★★★★ | Hours-Days | Free | EDGAR API, lumibot built-in |
| Earnings call transcripts | ★★★★ | Hours | $ | Motley Fool, Seeking Alpha |
| Reddit (r/wallstreetbets) | ★★★ | Real-time | Free | praw (Reddit API) |
| Twitter/X | ★★ | Real-time | $$$$ | Twitter API (expensive post-2023) |
| Google Trends | ★★★ | Daily | Free | pytrends |

**Best ROI:** Alpaca News API (free with account) + FinBERT sentiment scoring.

### Real-Time vs. End-of-Day Trading

| Strategy Type | Infrastructure Needed | Capital Efficiency | Complexity |
|--------------|----------------------|-------------------|------------|
| End-of-day (EOD) | Simple scheduler, daily data | Moderate | Low |
| Intraday (hourly/15min) | WebSocket feed, cloud server | Higher | Medium |
| High-frequency (<1min) | Co-location, Rust/C++, L2 data | Very high | Very High |
| Event-driven (earnings, macro) | News feed + NLP pipeline | High | High |

**Recommendation for new builder:** Start with **end-of-day or daily bars**. The edge from intraday adds complexity without proportional return for retail traders. Most successful retail algos operate on 1-day to 1-hour timeframes.

---

## 5. Architecture Patterns

### Recommended Production Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Strategy Layer                         │
│  Signal Generator → Risk Manager → Portfolio Optimizer   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Execution Layer                          │
│  Order Router → Broker API → Fill Tracker → P&L Logger  │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    Data Layer                             │
│  Market Data (WebSocket) + News Feed + Alternative Data  │
└─────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                 Persistence Layer                         │
│  Time-series DB (InfluxDB/TimescaleDB) + SQLite trades   │
└─────────────────────────────────────────────────────────┘
```

### Event-Driven Architecture

Every production framework uses an event bus pattern:

```python
# Core event types in most trading systems
class EventType(Enum):
    BAR = "BAR"              # OHLCV bar completed
    TICK = "TICK"            # Real-time price tick
    SIGNAL = "SIGNAL"        # Strategy generated signal
    ORDER = "ORDER"          # Order submitted
    FILL = "FILL"            # Order filled
    PORTFOLIO = "PORTFOLIO"  # Portfolio state update
    RISK = "RISK"            # Risk limit triggered
```

**Why event-driven?**
- Enables exact replay of historical events for accurate backtesting
- Decouples components — strategy doesn't know about broker
- Makes the system testable at the unit level
- Supports concurrent strategies without race conditions

### Microservices vs. Monolith

For a first bot: **start monolithic, plan for microservices**.

**Monolithic (recommended for v1):**
- Single Python process with all components
- SQLite for persistence
- Works well for daily/hourly strategies
- Much simpler to debug and deploy

**Microservices (for production scale):**
- Data service → Kafka/Redis streams → Strategy service → Order service
- Required for HFT or running 20+ strategies simultaneously
- Adds significant operational overhead

### State Management

```python
# Critical state to persist across restarts
state = {
    "positions": {},           # Current open positions
    "orders": {},              # Pending orders
    "equity_curve": [],        # Historical equity
    "daily_pnl": 0.0,          # Today's P&L (for circuit breaker)
    "drawdown_peak": 0.0,      # Peak equity (for max drawdown calc)
    "strategy_params": {},     # Current strategy parameters
}
```

Always persist state to disk — process restarts happen.

---

## 6. Repo Rankings - Top 10

### 1. freqtrade/freqtrade (51,800 stars)
**Why it's #1:** The most production-proven open-source crypto trading framework. Active community (Discord, GitHub), regularly updated, excellent documentation. FreqAI provides built-in adaptive ML without needing to write ML code. If building a crypto bot, start here — years of production experience are baked in.  
**Best for:** Crypto trading, ML-based strategies, teams wanting a mature foundation.

### 2. microsoft/qlib (45,200 stars)
**Why it's top-tier:** Microsoft-backed, research-grade, covers the entire quantitative investment pipeline. The RD-Agent LLM integration for automated factor discovery is genuinely novel. Best for stocks-focused, ML-heavy strategies.  
**Best for:** Quantitative research, US/China equities, ML pipeline development.

### 3. ccxt/ccxt (42,000 stars)
**Why it's top-tier:** Not a bot but essential infrastructure. If you're building any crypto trading system, you will use CCXT. It abstracts away 100+ exchange differences into one clean API.  
**Best for:** Crypto exchange connectivity layer, used as dependency in most crypto bots.

### 4. vnpy/vnpy (42,100 stars)
**Why it's top-tier:** The dominant platform for Chinese market trading (A-shares, domestic futures) but also supports international markets. Battle-tested at institutional scale.  
**Best for:** Chinese markets, institutional-grade requirements, futures/options.

### 5. nautechsystems/nautilus_trader (24,200 stars)
**Why it's elite:** The technically superior architecture — Rust core provides performance that Python frameworks cannot match. The "research-to-live parity" design eliminates the most common source of live trading bugs. Growing rapidly.  
**Best for:** Professional developers, HFT approaches, multi-asset production systems.

### 6. mementum/backtrader (22,100 stars)
**Why it's important:** The benchmark for Python backtesting. Enormous community, extensive documentation, and vast tutorial ecosystem. Note: no longer actively maintained since ~2022, but stable.  
**Best for:** Learning backtesting, stocks/forex strategies, Interactive Brokers live trading.

### 7. QuantConnect/Lean (20,200 stars)
**Why it's important:** Professional-grade engine used by QuantConnect's cloud platform. Best for multi-asset strategies including options, futures, and complex instruments. C# core with Python API.  
**Best for:** Multi-asset strategies, options/futures, institutional-grade requirements.

### 8. AI4Finance-Foundation/FinGPT (20,700 stars)
**Why it's notable:** The leading open-source financial LLM framework. Demonstrates that fine-tuned LLMs can beat GPT-4 on financial tasks at 1% of the cost. Essential reference for anyone adding NLP/sentiment to their bot.  
**Best for:** Financial NLP research, sentiment analysis, LLM-enhanced trading.

### 9. stefan-jansen/machine-learning-for-trading (19,300 stars)
**Why it's valuable:** A complete curriculum for ML-based trading. Not a framework but a comprehensive codebase covering everything from data acquisition to production deployment. The book's third edition adds LLM/RAG approaches.  
**Best for:** Learning ML for trading, reference implementation, strategy ideation.

### 10. hummingbot/hummingbot (19,000 stars)
**Why it's notable:** Dominant open-source framework for crypto market making. The $34B+ in trading volume speaks to production-scale usage. DEX support is unique — few other frameworks support on-chain trading.  
**Best for:** Market making, arbitrage, DeFi/DEX trading, HFT on crypto.

---

## 7. Technology Stack Recommendations

### Recommended Stack for a New Algo Trading Bot

**Language:** Python 3.11+  
**Framework:** lumibot (stocks/multi-asset) or freqtrade (crypto)  
**Broker:** Alpaca (stocks/crypto) or Binance via CCXT (crypto)

```
Component               Recommended Tool                    Alternative
─────────────────────────────────────────────────────────────────────
Framework               lumibot / freqtrade                 NautilusTrader (advanced)
Brokerage API           alpaca-py                           ib_async (IB)
Crypto Exchange         ccxt + python-binance               alpaca-py crypto
Data (market)           alpaca-py / Polygon.io              Yahoo Finance (free)
Data (news)             Alpaca News API                     Polygon News, NewsAPI
Technical Indicators    ta-lib / pandas-ta                  ta (simpler API)
ML / AI                 scikit-learn, LightGBM, PyTorch     qlib (heavier)
Sentiment Analysis      FinBERT (transformers)              OpenAI API
Backtesting             lumibot / backtrader                zipline-reloaded
Database (trades)       SQLite + SQLAlchemy                 PostgreSQL
Database (timeseries)   pandas + parquet files              InfluxDB / TimescaleDB
Task scheduling         APScheduler / cron                  Prefect / Airflow (heavy)
Notifications           python-telegram-bot                 Slack webhooks
Containerization        Docker + docker-compose             -
Monitoring              Grafana + InfluxDB                  simple logging to file
```

### Minimal Working Stack (v1 — get trading in a week)

```python
# requirements.txt for a minimal Alpaca + lumibot stock bot
lumibot==3.*
alpaca-py>=0.20
pandas>=2.0
pandas-ta>=0.3
scikit-learn>=1.3
python-dotenv>=1.0
```

### Production Stack (v2 — after you have a working strategy)

```python
# Additional dependencies for production
nautilus_trader>=1.200    # or stay with lumibot
influxdb-client>=3.0      # time-series metrics
grafana                   # dashboard (Docker)
redis>=5.0                # state caching
celery>=5.0               # async task queue
sentry-sdk                # error tracking
```

### Data Sources by Cost

| Provider | Cost | Data Quality | Coverage | Notes |
|----------|------|-------------|----------|-------|
| Alpaca (free tier) | Free | Good | US stocks, crypto | Requires account |
| Yahoo Finance (yfinance) | Free | Adequate | US/global stocks | EOD only, unreliable |
| Polygon.io | Free tier / $29+/mo | Excellent | US stocks, options, forex, crypto | Best value |
| Quandl / NASDAQ Data Link | $0–$500/mo | Excellent | Global, alternative data | Best for fundamentals |
| Refinitiv / Bloomberg | $1,500+/mo | Institutional | Everything | Overkill for retail |
| ThetaData | $20–$100/mo | Excellent | US options | Best for options backtesting |

---

## 8. Key Learnings / Gotchas

### The Most Dangerous Backtesting Pitfalls

**1. Look-Ahead Bias (most insidious)**  
Using data that wasn't available at decision time. Example: using end-of-day closing price to make a decision "at" the open. Can inflate Sharpe ratio from 0.8 to 1.5+ artificially.  
*Fix:* Always use point-in-time data. Use `shifted()` on pandas series. Never use `row[-1]` for the current bar — use `row[-2]`.

**2. Overfitting**  
A strategy with 15 parameters optimized on 3 years of data will look incredible in-sample and fail out-of-sample.  
*Fix:* Walk-forward optimization (rolling windows). Out-of-sample test set must never be touched until final validation. Maximum 1–2 parameters per year of data.

**3. Survivorship Bias**  
Backtesting on S&P 500 "current constituents" ignores all the companies that got delisted or went bankrupt. Creates artificially high returns.  
*Fix:* Use point-in-time index constituents. Providers: Norgate Data, CRSP. Polygon.io includes delisted stocks.

**4. Unrealistic Transaction Costs**  
Even "commission-free" trading has costs: bid-ask spread (0.01–0.10%), market impact (1–5bps on large orders), and slippage. A strategy returning 20% pre-costs may return 8% after realistic costs.  
*Fix:* Model spread as 0.05–0.1% per trade in backtest. Add 1–3bps market impact for positions > 1% of ADV.

**5. Regime Changes**  
A strategy that worked perfectly in 2019–2021 (low volatility, trending market) often fails catastrophically in 2022 (high volatility, mean-reverting). Most backtests cover too short a window.  
*Fix:* Backtest across at least 10 years including different market regimes. Include 2008, 2020, 2022 in your test set.

### Common Implementation Mistakes

**6. Timezone Hell**  
Market data timestamps often mix UTC, Eastern, and exchange-local times. A single timezone inconsistency can cause massive look-ahead bias.  
*Fix:* Store everything in UTC. Convert at the very last moment for display.

**7. Not Accounting for Corporate Actions**  
Stock splits (e.g., AAPL 4:1 in 2020) create apparent 75% crashes in raw price data. Dividends cause apparent gaps.  
*Fix:* Always use adjusted close prices for backtesting. Most data providers offer this.

**8. Integer vs. Fractional Shares**  
Alpaca supports fractional shares but Interactive Brokers does not. Position sizing that requires 2.3 shares will behave differently in backtest vs. live.  
*Fix:* Always floor to integer shares in your position sizing unless you've confirmed fractional share support.

**9. Capital Allocation at Open**  
If multiple strategy signals fire simultaneously, naive implementations try to deploy 100% of capital into each one. Portfolio-level position sizing must run before individual orders.  
*Fix:* Implement a portfolio allocator that runs after all signals are collected, not within each signal handler.

**10. API Rate Limits**  
Alpaca allows 200 requests per minute. Binance allows 1,200/min. Breaching these causes temporary bans. During market open, rate limits can be hit instantly when checking many positions.  
*Fix:* Implement exponential backoff. Batch order status checks. Cache market data locally.

### Things That Sound Good But Don't Work Well

- **Neural networks on raw OHLCV data:** Random-walk-like price data is extremely hard for NNs to learn from. Start with engineered features (returns, volatility, indicators) rather than raw prices.
- **Twitter/X sentiment for retail trading:** Too noisy, and the API is now extremely expensive. News-based sentiment (FinBERT on Reuters/AP articles) has much better signal-to-noise.
- **Minute-bar backtesting for daily strategies:** Adds computation without accuracy benefit. Daily-bar backtest for daily-bar strategies.
- **Paper trading as a perfect proxy for live:** Paper trading fills are often at mid-price. Live fills are at ask (buy) or bid (sell). Spreads can be 0.05–0.5% in illiquid names.
- **Optimizing to minimize Sharpe ratio — maximize returns:** Maximizing raw returns selects for strategies with hidden leverage or tail risk. Always optimize Sharpe or Sortino.

### What Actually Works

Based on the repos studied, strategies with the most documented success:
1. **Trend following on futures** (momentum) — works across all time periods
2. **Mean reversion on pairs / stat arb** — pairs that are cointegrated
3. **Factor investing (value + momentum)** — small-cap, value, momentum factors
4. **Earnings surprise plays** — trading the gap between analyst estimates and actual
5. **Volatility selling** (careful risk management required) — selling options premium
6. **Market making** (HFT) — extremely capital-efficient but requires co-location

---

## 9. Next Steps / Roadmap

### Phase 1: Foundation (Week 1–2)
- [ ] Create Alpaca paper trading account at alpaca.markets
- [ ] Install `lumibot` and run the example strategies
- [ ] Implement your first strategy: a simple EMA crossover on SPY
- [ ] Verify backtest results match intuition (positive Sharpe on trend period)
- [ ] Deploy to paper trading and run for 1–2 weeks

**Verify:** Does the live paper P&L roughly match backtest? If not, find the discrepancy before continuing.

### Phase 2: Strategy Development (Week 3–6)
- [ ] Add `pandas-ta` for technical indicators
- [ ] Implement proper position sizing (ATR-based)
- [ ] Add stop-loss and take-profit logic
- [ ] Run walk-forward validation (not just single backtest)
- [ ] Research: read first 3 chapters of "Machine Learning for Algorithmic Trading" (Stefan Jansen)
- [ ] Build a simple equity curve dashboard

**Goal:** Have a strategy with Sharpe > 1.0 on out-of-sample data across multiple market regimes.

### Phase 3: ML Enhancement (Week 7–12)
- [ ] Add `FinBERT` sentiment scoring on Alpaca news feed
- [ ] Train a simple LightGBM model on engineered features (returns, volume, indicators)
- [ ] Add the ML signal as a filter (only trade when ML confidence > threshold)
- [ ] Compare ML-enhanced vs. rule-based performance on same backtest period

### Phase 4: Risk & Operations (Week 13–16)
- [ ] Implement multi-level risk management (per-trade, portfolio, daily loss)
- [ ] Add Telegram notifications for fills, errors, and daily P&L
- [ ] Deploy to a VPS (DigitalOcean, AWS t3.small is sufficient for daily/hourly strategies)
- [ ] Set up basic monitoring: log all trades to SQLite, alert on errors
- [ ] Implement graceful shutdown and state persistence

### Phase 5: Live Trading (Week 17+)
- [ ] Run paper trading for minimum 30 days before going live
- [ ] Start with very small position sizes ($100–500 per trade)
- [ ] Track live vs. backtest divergence weekly
- [ ] Gradually increase position size as confidence grows
- [ ] Implement a circuit breaker: auto-halt if daily loss > 2% of account

### Capital Sizing Guide

| Account Size | Strategies | Max Position | Notes |
|-------------|------------|-------------|-------|
| $1,000–5,000 | 1 strategy | $500/position | Learning mode, expect losses |
| $5,000–25,000 | 1–3 strategies | $1,000–2,500/position | Pattern Day Trader rules apply |
| $25,000+ | 3–10 strategies | 2–5% per position | Full PDT-compliant, can day trade |

> **Note:** The SEC Pattern Day Trader (PDT) rule requires $25,000 minimum for making 4+ day trades per week in US markets. Swing trading (hold > 1 day) avoids this restriction.

### Key Resources

- **Documentation:** [Alpaca API Docs](https://docs.alpaca.markets) | [lumibot Docs](https://lumibot.lumiwealth.com)
- **Book:** "Machine Learning for Algorithmic Trading" by Stefan Jansen (3rd ed.)
- **Book:** "Advances in Financial Machine Learning" by Marcos Lopez de Prado
- **Community:** freqtrade Discord (most active algo trading community)
- **Backtesting reference:** [QuantConnect LEAN Docs](https://www.lean.io/docs) for methodology
- **Data:** [Polygon.io](https://polygon.io) for affordable quality data
- **Research:** [SSRN Quantitative Finance papers](https://ssrn.com/en/index.cfm/quant-finance/) for strategy ideas

---

*This document was generated on 2026-06-25 based on research into 28 major algorithmic trading GitHub repositories. Star counts are approximate as of June 2026.*
