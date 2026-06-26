# Reinforcement Learning for Finance & Trading Bots
### A Comprehensive Technical Reference

**Purpose:** Foundational reading before implementing RL in a trading bot. Covers theory, algorithms, finance-specific formulations, implementation patterns, and the top papers you actually need to read.

**Last updated:** June 2026

---

## Table of Contents

1. [Introduction & Why RL for Finance](#1-introduction--why-rl-for-finance)
2. [RL Fundamentals Refresher (Finance-Focused)](#2-rl-fundamentals-refresher-finance-focused)
3. [Core Algorithms: Deep Dives](#3-core-algorithms-deep-dives)
4. [Finance-Specific RL Formulations](#4-finance-specific-rl-formulations)
5. [Reward Function Design](#5-reward-function-design-critical-section)
6. [State Space Design](#6-state-space-design)
7. [Action Space Design](#7-action-space-design)
8. [Training Infrastructure](#8-training-infrastructure)
9. [Best Libraries & Frameworks](#9-best-libraries--frameworks)
10. [Top Papers Ranked & Annotated](#10-top-papers-ranked--annotated)
11. [Recommended Implementation Path](#11-recommended-implementation-path)
12. [Common Pitfalls & How to Avoid Them](#12-common-pitfalls--how-to-avoid-them)
13. [Cutting Edge (2023–2025)](#13-cutting-edge-20232025)

---

## 1. Introduction & Why RL for Finance

### Why RL is Well-Suited to Trading

Trading is a sequential decision problem under uncertainty — the exact setting RL is designed for. Unlike supervised learning, which fits a static model to labeled data, RL explicitly optimises a policy for long-term outcomes in an environment that responds to the agent's actions.

Key structural reasons RL fits trading:

| Trading Property | RL Counterpart |
|---|---|
| Continuous stream of market data | Markov Decision Process (MDP) with time steps |
| Buy/sell/size decisions at each step | Action space |
| Profit, Sharpe ratio, drawdown | Reward function |
| Positions, prices, portfolio value | State space |
| Market impact (your orders move prices) | Environment transition dynamics |
| Transaction costs | Negative reward / cost term |
| Unknown future price path | Stochastic environment |

### Key Differences From Supervised Learning for Trading

**Supervised learning (e.g., predicting next-day returns):**
- Predicts a fixed target at each step independently
- No feedback loop between prediction and market outcomes
- Cannot model transaction costs, position constraints, or sequential dependencies
- Optimal by MSE/cross-entropy, not necessarily profitable

**Reinforcement learning:**
- Optimises a cumulative, risk-adjusted objective directly (e.g., Sharpe ratio)
- Models the full trajectory: each trade affects future state
- Transaction costs are part of the reward signal — agent learns to minimize unnecessary trading
- Can learn contrarian strategies that lose on individual predictions but profit overall
- Naturally handles position limits, leverage constraints, and multi-asset correlations

### Major Challenges

1. **Non-stationarity:** Markets change regime constantly. A policy trained on 2018 bull market fails in 2020 crash. Unlike Atari, financial MDPs are non-stationary.

2. **Sparse rewards:** Daily return signal is noisy and delayed. A good trade today might show profit only in 3 days after all fills complete.

3. **Transaction costs:** Every action has a cost. Without explicit cost modelling, RL agents overtrade dramatically, generating huge gross profit but negative net profit.

4. **Market impact:** Large orders move the market against you. This creates a non-linear cost function the agent must learn.

5. **Partial observability:** True market state (order flow, institutional positioning, latent liquidity) is not observable. You only see prices and volume.

6. **Data scarcity:** 20 years of daily data = ~5,000 data points. Compare to Atari where you can generate millions of frames. Financial RL is severely data-constrained.

7. **Overfitting to historical data:** Historical market data has survivorship bias, look-ahead bias, and regime changes that don't repeat. A policy that perfectly optimises the training period will fail live.

8. **Delayed reward attribution:** Was a profitable trade due to the entry signal or the exit timing? Credit assignment across time is hard in finance.

---

## 2. RL Fundamentals Refresher (Finance-Focused)

### MDP Formulation for Trading

A trading problem maps to a Markov Decision Process (MDP) defined by the tuple `(S, A, R, P, γ)`:

**State space S** — what the agent observes at each timestep:
```
s_t = [price_history, volume, technical_indicators, current_position, portfolio_value, ...]
```

**Action space A** — what the agent decides:
```
# Discrete:   a_t ∈ {0=hold, 1=buy_small, 2=buy_large, 3=sell_small, 4=sell_large}
# Continuous: a_t ∈ [-1, +1]  (negative = short, positive = long, magnitude = size)
# Portfolio:  a_t ∈ Δ^N       (portfolio weight vector, sums to 1, N assets)
```

**Reward function R** — the signal the agent optimises:
```
r_t = PnL_t - transaction_cost_t - risk_penalty_t
```

**Transition dynamics P** — how state evolves (the market, mostly uncontrollable):
```
s_{t+1} ~ P(s_{t+1} | s_t, a_t)   # market moves after your action
```

**Discount factor γ** — how much the agent values future rewards:
```
γ ∈ [0.99, 1.0] for finance (long-horizon tasks)
```

**Objective:** Find policy `π*(s)` that maximises expected discounted return:
```
J(π) = E_π [ Σ_t γ^t r_t ]
```

### On-Policy vs Off-Policy

This distinction critically affects data efficiency and training stability in finance:

**On-policy (PPO, A2C):**
- Learns from experience generated by the *current* policy
- Cannot reuse old experiences — must generate fresh data
- More stable, less sample-efficient
- Better when your trading simulator is fast and cheap to run
- PPO's clipping prevents catastrophic policy changes

**Off-policy (DQN, DDPG, SAC, TD3):**
- Learns from a replay buffer of *past* experiences from any policy
- Can reuse expensive historical data multiple times
- More sample-efficient but potentially less stable
- **Preferred in finance** because real trading data is expensive and limited

**Why this matters for trading:** If you're training on real historical data (limited), off-policy methods let you repeatedly reuse the same data via experience replay. If you have a fast simulator, on-policy methods like PPO work well.

### Model-Free vs Model-Based

| Aspect | Model-Free | Model-Based |
|---|---|---|
| Learns | Policy or value function directly | A model of environment dynamics |
| Sample efficiency | Lower | Higher (can generate synthetic data) |
| Compute | Less | More |
| Risk | Overfit to environment | Model error propagates |
| Finance use | Standard approach | Emerging — market simulators |
| Examples | DQN, PPO, SAC | MBPO, Dreamer, EIIE world model |

**Finance verdict:** Most production RL trading systems use model-free methods (PPO, SAC) due to simplicity. Model-based RL is promising for data augmentation — learn a market simulator, generate synthetic trajectories, train model-free agent on them.

### Algorithm Comparison Table

| Algorithm | Type | Action Space | Sample Efficiency | Stability | Finance Suitability |
|---|---|---|---|---|---|
| DQN | Value-based | Discrete | Low | Medium | ★★★☆☆ |
| DDPG | Actor-Critic (off-policy) | Continuous | High | Low | ★★★★☆ |
| TD3 | Actor-Critic (off-policy) | Continuous | High | Medium | ★★★★☆ |
| SAC | Actor-Critic (off-policy, max-entropy) | Continuous | High | High | ★★★★★ |
| PPO | Policy Gradient (on-policy) | Both | Medium | High | ★★★★☆ |
| A2C | Actor-Critic (on-policy) | Both | Low | Medium | ★★★☆☆ |
| DRQN | Value-based (recurrent) | Discrete | Low | Medium | ★★★★☆ |
| C51 | Value-based (distributional) | Discrete | Low | Medium | ★★★★☆ |
| Decision Transformer | Offline RL | Both | N/A (offline) | High | ★★★★☆ |
| CQL | Offline RL | Both | N/A (offline) | High | ★★★★☆ |

---

## 3. Core Algorithms: Deep Dives

### 3.1 DQN — Deep Q-Network

**Core idea:** Approximate the Q-function `Q(s, a)` — the expected future return from taking action `a` in state `s` — with a neural network. Use experience replay to decorrelate training samples and a target network to stabilise the Bellman backup.

**Key equations:**
```
Loss = E[(r + γ max_{a'} Q_target(s', a') - Q(s, a))^2]

Q_target: frozen copy of Q updated every N steps
Replay buffer D: store (s, a, r, s') tuples; sample random minibatches
```

**Strengths for finance:**
- Simple to implement and debug
- Discrete actions map naturally to buy/hold/sell signals
- Prioritised experience replay handles rare market events (crashes)

**Weaknesses / pitfalls:**
- Only discrete actions — cannot directly output portfolio weights
- Tends to overestimate Q-values (use Double DQN to fix)
- Epsilon-greedy exploration is naive for financial data

**Best finance applications:**
- Directional trading (long/short signals)
- Discrete position sizing (3-5 levels)
- Order type selection (market/limit)

**Recommended hyperparameters for trading:**
```python
learning_rate = 1e-4          # Conservative; financial signals are noisy
buffer_size = 100_000         # Large replay buffer
batch_size = 64
target_update_freq = 1000     # Frequent updates for non-stationary markets
gamma = 0.99                  # Long horizon; don't discount too heavily
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 50_000 steps  # Slow decay; markets are hard to explore
```

**Python library:** `stable-baselines3` (DQN), `FinRL`

---

### 3.2 DDPG — Deep Deterministic Policy Gradient

**Core idea:** Extend DQN to continuous action spaces using a deterministic policy `μ(s)` as the actor and a Q-function `Q(s, a)` as the critic. The actor is updated by backpropagating through the critic.

**Key equations:**
```
Actor update:  ∇_θ J ≈ E[∇_a Q(s, a)|_{a=μ(s)} · ∇_θ μ(s)]
Critic update: L = E[(r + γ Q_target(s', μ_target(s')) - Q(s, a))^2]
Exploration:   a_t = μ(s_t) + N_t   (Ornstein-Uhlenbeck noise)
```

**Strengths for finance:**
- Continuous actions — directly outputs portfolio weights or position sizes
- Off-policy: reuses experience replay
- Established baseline for portfolio management (EIIE, FinRL)

**Weaknesses / pitfalls:**
- Systematic Q-value overestimation → use TD3 instead
- Sensitive to hyperparameters; often unstable
- OU noise for exploration is poorly calibrated for financial data
- Struggles with highly correlated assets

**Best finance applications:**
- Portfolio weight optimization (multi-asset)
- Continuous order sizing
- Execution rate scheduling

**Recommended hyperparameters:**
```python
actor_lr = 1e-4
critic_lr = 1e-3
tau = 0.005               # Soft target update; keep small for stability
batch_size = 128
buffer_size = 200_000
noise_sigma = 0.1         # OU noise std; tune for exploration needed
```

**Python library:** `stable-baselines3` (DDPG), `FinRL`

---

### 3.3 TD3 — Twin Delayed Deep Deterministic Policy Gradient

**Core idea:** Fixed DDPG's overestimation bias with three key changes: (1) twin critics take the minimum; (2) delayed actor updates; (3) target policy smoothing noise.

**Key equations:**
```
# Twin critics
y = r + γ min_{i=1,2} Q_{i,target}(s', μ_target(s') + ε)
  where ε ~ clip(N(0, σ), -c, c)   # smoothing noise

# Actor updated every 2 critic updates
∇_θ J = E[∇_a Q_1(s, a)|_{a=μ(s)} · ∇_θ μ(s)]
```

**Strengths for finance:**
- More stable than DDPG — reliable training without collapse
- Lower variance Q-estimates → better policy
- Target policy smoothing acts as implicit regularization

**Weaknesses / pitfalls:**
- Still deterministic — can overfit to specific entry/exit patterns
- Requires careful tuning of smoothing noise `σ` and `c`
- Less entropy-driven exploration than SAC

**Best finance applications:**
- Portfolio rebalancing (continuous weights)
- Options delta hedging
- Market making (continuous bid-ask spread)

**Recommended hyperparameters:**
```python
actor_lr = 3e-4
critic_lr = 3e-4
policy_delay = 2           # Update actor every 2 critic updates
noise_std = 0.2            # Target policy smoothing noise
noise_clip = 0.5           # Clip smoothing noise
tau = 0.005
```

**Python library:** `stable-baselines3` (TD3), `sfujim/TD3` (reference implementation)

---

### 3.4 SAC — Soft Actor-Critic

**Core idea:** Maximise a trade-off between expected reward and policy entropy, encouraging the agent to be as random as possible while still being rewarded. Results in naturally stochastic policies, better exploration, and resistance to overfitting.

**Key equations:**
```
Objective: J(π) = Σ_t E_{s_t~π}[r(s_t, a_t) + α H(π(·|s_t))]

α: temperature parameter (auto-tuned in SAC v2)
Auto-tuning: ∇_α L(α) = E[-α log π(a|s) - α H̄]  where H̄ is target entropy

Twin soft Q-functions:
V_soft(s) = E_a[min(Q_1, Q_2)(s,a) - α log π(a|s)]

Actor loss: E[-min(Q_1, Q_2)(s, ã) + α log π(ã|s)]
  where ã ~ π(·|s) reparameterized
```

**Strengths for finance:**
- Maximum entropy → naturally diversified portfolio weights
- Auto-tuning of temperature `α` eliminates a key hyperparameter
- Off-policy + continuous → best of both worlds
- Stochastic policy reduces overfitting to historical patterns
- State-of-the-art on MuJoCo; consistently strong on financial benchmarks

**Weaknesses / pitfalls:**
- More complex than PPO; harder to debug
- Requires symmetric action bounds (use softmax wrapper for portfolio weights)
- Target entropy H̄ = -dim(A) may not be optimal for finance

**Best finance applications:**
- **Best overall choice for portfolio weight optimization**
- Continuous position sizing
- Risk-adjusted trading (entropy ≈ diversification)

**Recommended hyperparameters:**
```python
learning_rate = 3e-4
buffer_size = 1_000_000
batch_size = 256
tau = 0.005
gamma = 0.99
target_entropy = "auto"    # -dim(action_space)
ent_coef = "auto"          # auto-tune temperature
```

**Python library:** `stable-baselines3` (SAC), `haarnoja/sac` (original)

---

### 3.5 PPO — Proximal Policy Optimization

**Core idea:** Policy gradient method that clips the probability ratio between new and old policies to prevent destructively large updates. Allows multiple epochs of minibatch SGD on the same data, unlike vanilla policy gradients.

**Key equations:**
```
Clipped surrogate objective:
L^CLIP(θ) = E_t[min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]

where:
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)   # probability ratio
Â_t = GAE advantage estimate
ε = 0.2 (clip range)

Total loss = L^CLIP - c1 * L^VF + c2 * S[π_θ]
  where S is entropy bonus for exploration
```

**Strengths for finance:**
- Excellent stability — won't catastrophically collapse
- Simple implementation; well-documented
- Works for both discrete and continuous actions
- Parallel environments trivially supported
- Most widely used algorithm in FinRL literature

**Weaknesses / pitfalls:**
- On-policy: can't reuse old experience → needs continuous data generation
- Lower sample efficiency than SAC/TD3 on data-limited finance tasks
- Clip range `ε` needs tuning; too small → slow learning; too large → instability

**Best finance applications:**
- General-purpose trading strategy
- Portfolio management with gym-style environments
- Best default choice when you're starting out

**Recommended hyperparameters:**
```python
learning_rate = 3e-4
n_steps = 2048             # Steps per rollout per environment
batch_size = 64
n_epochs = 10              # Gradient update passes per rollout
gamma = 0.99
gae_lambda = 0.95          # GAE smoothing
clip_range = 0.2
ent_coef = 0.01            # Entropy bonus; increase if underfitting
```

**Python library:** `stable-baselines3` (PPO), `FinRL`

---

### 3.6 A2C / A3C — Advantage Actor-Critic

**Core idea:** Simultaneously train an actor (policy) and critic (value function). The advantage `A(s, a) = Q(s, a) - V(s)` reduces variance of policy gradient estimates. A3C uses asynchronous parallel actors; A2C uses synchronous parallel actors.

**Key equations:**
```
Policy gradient: ∇_θ J ≈ E[∇_θ log π_θ(a|s) · Â(s, a)]
Advantage:       Â(s, a) = r + γV(s') - V(s)   (1-step TD)
             or  Â = Σ_{k=0}^{T} (γλ)^k δ_{t+k}  (GAE, k-step)
Critic loss:     L^V = E[(r + γV(s') - V(s))^2]
```

**Strengths for finance:**
- A2C with vectorized envs (N parallel envs) is very efficient on CPU
- Lower memory than replay buffer methods
- Included in FinRL as a baseline

**Weaknesses / pitfalls:**
- Higher variance than PPO
- Less sample-efficient than off-policy methods
- A3C's asynchronous updates can be unstable

**Best finance applications:**
- Rapid prototyping; good baseline
- Parallel training on multiple stocks simultaneously
- When you need a simple on-policy baseline

**Python library:** `stable-baselines3` (A2C), `FinRL`

---

### 3.7 Decision Transformer

**Core idea:** Cast RL as a sequence modelling problem. Given a context of (return-to-go, state, action) triples, use a GPT-style transformer to autoregressively predict the next action. Condition on a desired future return at test time.

**Key equations:**
```
Context: [(R̂_1, s_1, a_1), (R̂_2, s_2, a_2), ..., (R̂_t, s_t)]
Output:  â_t = DT(R̂_1, s_1, a_1, ..., R̂_t, s_t)

where R̂_t = Σ_{t'=t}^{T} r_{t'} is the return-to-go

Training loss: E[(a_t - DT(context_t))^2]   (or cross-entropy for discrete)
At test time: set R̂_1 to your desired total return (e.g., target Sharpe ratio)
```

**Strengths for finance:**
- Works entirely in offline mode — no environment interaction needed
- Can specify desired performance at inference (e.g., "achieve 2.0 Sharpe ratio")
- Transformer attention naturally handles long-term dependencies
- No reward hacking; no value function overestimation

**Weaknesses / pitfalls:**
- Offline only — cannot adapt to new market regimes without retraining
- Return-to-go conditioning requires knowing what "good" looks like upfront
- Requires large offline datasets; limited for short price history

**Best finance applications:**
- Learning from historical trade logs
- Imitating known profitable strategies
- Conditioning on desired risk-adjusted returns

**Python library:** Official: `kzl/decision-transformer`; integrations in `d3rlpy`

---

## 4. Finance-Specific RL Formulations

### 4a. Portfolio Management

Portfolio management is the most studied RL finance application. The goal: allocate wealth across N assets to maximise long-term risk-adjusted returns.

#### State Representation

```python
# Recommended state for portfolio RL
state = {
    # Price features (shape: N_assets × T_window)
    "price_history": np.array,        # Normalised OHLCV, last 30-60 bars
    "returns": np.array,              # Log returns, last 30-60 bars
    
    # Technical indicators (computed from OHLCV)
    "rsi": np.array,                  # RSI(14) per asset
    "macd": np.array,                 # MACD signal per asset
    "bollinger": np.array,            # BB %B per asset
    "volume_ratio": np.array,         # Volume vs 20-day average
    
    # Portfolio state
    "current_weights": np.array,      # Shape: (N_assets,), sums to 1
    "portfolio_value": float,         # Normalised by initial value
    "cash_position": float,           # Fraction held in cash
    
    # Macro features (optional but valuable)
    "vix": float,                     # Market volatility index
    "market_return": float,           # Broad market return today
}
```

#### Action Space for Portfolio

```python
# Softmax output ensures weights sum to 1 and are positive
def portfolio_action(raw_output):
    # raw_output: (N_assets,) from actor network
    weights = softmax(raw_output)     # Sum to 1, all positive
    return weights

# For long-short portfolios:
def long_short_action(raw_output):
    # Allow negative weights (short positions)
    # Constraint: |weights| sum to leverage (e.g., 1.0 for 1x leverage)
    normalized = raw_output / np.abs(raw_output).sum()
    return normalized
```

#### Reward Functions

```python
# Option 1: Log portfolio return (most common)
r_t = np.log(portfolio_value_t / portfolio_value_{t-1}) - transaction_costs

# Option 2: Sharpe ratio (window-based)
r_t = mean_return(window) / std_return(window) * sqrt(252)

# Option 3: Risk-adjusted return with drawdown penalty
r_t = daily_return - lambda_1 * transaction_cost - lambda_2 * max_drawdown_penalty

# Option 4: Sortino-style (only penalise downside volatility)
downside_vol = std(returns[returns < 0])
r_t = mean_return / (downside_vol + 1e-8)
```

#### Best Algorithms for Portfolio Management

1. **SAC** — Best for continuous portfolio weights; entropy encourages diversification
2. **PPO** — Most stable; good default choice
3. **DDPG / TD3** — Work but less stable than SAC
4. **Decision Transformer** — Excellent for offline learning from historical data

#### EIIE Architecture (Jiang et al. 2017)

The Ensemble of Identical Independent Evaluators (EIIE) is a purpose-built architecture for portfolio RL:

```
Input: price_history (N_assets × T_window × features)
       ↓
For each asset i independently:
    CNN/LSTM_i(price_history[i]) → score_i   # identical network weights
       ↓
[score_1, score_2, ..., score_N, cash_bias]
       ↓
Softmax → portfolio_weights               # sums to 1

Key: same network weights for all assets → permutation invariant
Portfolio Vector Memory (PVM): prev weights fed back as input to each sub-network
```

The PVM encodes the "cost to change portfolio" — if you hold 40% in AAPL and want to go to 0%, that costs money. The network learns to account for this.

---

### 4b. Market Making

Market making is the business of providing liquidity by continuously quoting bid and ask prices, profiting from the spread while managing inventory risk.

#### Avellaneda-Stoikov Model Overview

The classical model (2008) derives optimal bid/ask quotes:

```
Reservation price: q̄(s, q, t) = s - q·γ·σ²·(T-t)
  where:
    s = mid-price
    q = current inventory
    γ = risk aversion parameter
    σ = price volatility
    T-t = time remaining

Optimal spread: δ* = γ·σ²·(T-t) + (2/γ) · log(1 + γ/κ)
  where κ = order arrival rate sensitivity

Bid:  b* = q̄ - δ*/2
Ask:  a* = q̄ + δ*/2
```

The intuition: skew quotes away from current inventory. Long inventory? Lower your ask to sell, raise your bid to buy more carefully.

#### RL Formulation for Market Making

```python
# State
s_t = [
    mid_price,
    spread,
    order_book_imbalance,   # (best_bid_qty - best_ask_qty) / total_qty
    current_inventory,       # how many units long/short
    time_of_day,
    recent_volatility,
    last_fill_size,
]

# Action (continuous)
a_t = [bid_offset, ask_offset]   # offset from mid price
# or (discrete)
a_t ∈ {narrow, medium, wide} × {neutral, skew_bid, skew_ask}

# Reward
r_t = (ask_price - bid_price) * fills_t           # spread capture
    - inventory_penalty * inventory_t^2            # quadratic inventory cost
    - transaction_cost * |trades_t|
```

**Best algorithms:** DQN with discretised spreads, DDPG for continuous bid/ask offsets.

---

### 4c. Optimal Execution / Order Execution

The problem: liquidate (or acquire) a large position over T periods with minimal market impact and implementation shortfall.

#### Almgren-Chriss Framework

The classical model (2001) assumes:
```
Price impact: linear in trade rate — g(v) = η·v
Permanent impact: f(v) = γ·v (shifts fair value permanently)
Temporary impact: g(v) = η·v (reversion after fill)

Optimal execution: trade n_t units at each period t
Minimize: E[cost] + λ · Var[cost]

Solution: n_t* = X/T + ... (arithmetic TWAP is special case λ→0)
```

#### RL as Improvement over Almgren-Chriss

RL can dynamically adapt execution based on real-time signals:

```python
# State for execution RL
s_t = [
    remaining_shares_to_execute,
    time_remaining,
    current_spread,
    volume_at_best_bid_ask,
    recent_price_trend,      # momentum — adapt urgency
    intraday_volume_pattern, # known U-shape for equities
]

# Action
a_t = fraction_to_execute_now ∈ [0, 1]

# Reward
r_t = -(execution_price - arrival_price) * shares_executed  # implementation shortfall
    + bonus if executed_early when momentum favorable
```

**VWAP/TWAP vs RL execution:**
- TWAP: execute evenly — ignores all signals
- VWAP: execute proportional to expected volume profile — static schedule
- RL: adapts dynamically to spread, momentum, and volume signals — learns to accelerate in favorable conditions

**Best algorithms:** DQN with discretised rates, DDPG for continuous rate.

---

### 4d. Options Hedging / Deep Hedging

#### Deep Hedging (Bühlеr et al. 2019) — Key Innovation

Traditional Black-Scholes delta hedging assumes:
- Frictionless markets (no transaction costs)
- Continuous trading
- Volatility is known and constant

Deep Hedging replaces this with a neural network that learns the optimal hedging strategy under **realistic conditions**:

```
Problem: Hedge derivative payoff V_T with trading strategy (δ_t) ∈ [-1, 1]

Terminal PnL: Z = -V_T + Σ_t δ_t · ΔS_t - Σ_t c_t(|δ_t - δ_{t-1}|)
  where c_t is the transaction cost function

Objective: min_π E[ρ(-Z)]
  where ρ is a convex risk measure (CVaR, variance, expected shortfall)

Policy: δ_t = π(market_features_t, δ_{t-1})   # neural network
```

The key insight: **transaction costs make Black-Scholes suboptimal**. Deep Hedging can handle any cost structure, any risk measure, and market incompleteness natively.

#### How Deep Hedging Works

```python
# Deep Hedging neural network
class DeepHedger(nn.Module):
    def __init__(self, n_features, n_hedging_instruments):
        self.lstm = nn.LSTM(n_features, 64)
        self.head = nn.Linear(64, n_hedging_instruments)
        self.prev_delta = nn.Parameter(torch.zeros(n_hedging_instruments))
    
    def forward(self, market_features, prev_delta):
        # market_features: [S_t, σ_t, t_remaining, moneyness, prev_delta]
        h, _ = self.lstm(market_features)
        raw_delta = self.head(h)
        delta = torch.tanh(raw_delta)  # constrain to [-1, 1]
        return delta

# Training: minimise CVaR of terminal hedging loss
def risk_measure_CVaR(losses, alpha=0.95):
    sorted_losses = torch.sort(losses)[0]
    cutoff = int(alpha * len(sorted_losses))
    return sorted_losses[cutoff:].mean()
```

**Extension: Gamma/Vega Hedging (Cao et al. 2022)**
Uses D4PG (distributional DDPG) to hedge second-order Greeks, accounting for CVaR and VaR objectives simultaneously.

---

### 4e. Cryptocurrency / High-Frequency Trading

#### Unique Challenges for Crypto

| Challenge | Description | RL Response |
|---|---|---|
| 24/7 markets | No overnight gaps; continuous learning | Persistent agent, online adaptation |
| High volatility | 10x+ daily moves possible | Higher position limits; volatility-scaled rewards |
| Thin order books | Large slippage on even moderate orders | LOB-aware state; slippage in reward |
| Multiple exchanges | Price discrepancies | Multi-venue action space |
| Stablecoins | Can hold cash as an asset | Include cash in portfolio vector |
| No circuit breakers | Flash crashes happen | Drawdown-aware reward shaping |

#### Best Algorithms for Crypto

- **Portfolio management:** SAC with stochastic continuous weights
- **Market making:** DDPG for continuous quote offsets; PPO for discrete spread levels
- **HFT:** PPO on LOB features (see Ardon et al. 2021)
- **Crypto-specific:** EIIE architecture (Jiang et al. 2017) validated on crypto

#### Critical: Address Backtest Overfitting

Per Gort et al. (2022), use **walk-forward validation**:

```
Training set:  2019 Jan - 2021 Dec
Validation 1:  2022 Jan - Jun
Validation 2:  2022 Jul - Dec
Test:          2023 Jan - present

Train → tune hyperparams on Val 1 → retrain on train+val1 → eval Val 2 → final test
DO NOT peek at test set until fully committed to architecture
```

---

## 5. Reward Function Design (Critical Section)

Reward function design is arguably **the most impactful decision** in your trading RL system. A wrong reward function produces a "clever" agent that maximises the wrong thing.

### Why Reward Function Design is the Hardest Part

1. **Proxy reward problem:** You can't directly reward "be a good trader." You reward some proxy (daily return, Sharpe ratio) that correlates with being a good trader but can be gamed.
2. **Reward hacking:** An agent maximising Sharpe by driving denominator (volatility) to zero will refuse to trade — technically infinite Sharpe.
3. **Sparse rewards:** Daily return is 0.0001% on a good day. The agent struggles to distinguish signal from noise.
4. **Temporal mismatch:** Trade entry occurs at t=0 but payoff may not materialise until t=5. Which timestep gets the reward?
5. **Multi-objective tension:** High return usually means high risk. Your reward function must encode the right tradeoff.

### Common Reward Functions

#### 1. Raw PnL (Simplest)

```python
r_t = (portfolio_value_t - portfolio_value_{t-1}) / portfolio_value_{t-1}
     - transaction_cost_rate * abs(position_change_t)
```
- **Pro:** Simple, directly measurable
- **Con:** High variance; doesn't penalise risk; agent takes maximum leverage

#### 2. Sharpe Ratio Reward (Window-Based)

```python
def sharpe_reward(returns_window, risk_free=0.0):
    """Compute rolling Sharpe over recent window."""
    excess = returns_window - risk_free / 252
    if np.std(excess) < 1e-8:
        return 0.0
    return np.mean(excess) / np.std(excess) * np.sqrt(252)

# Use a 20-60 bar window; update at each timestep
r_t = sharpe_reward(recent_returns[-30:])
```
- **Pro:** Risk-adjusted; industry-standard metric
- **Con:** Non-differentiable at σ=0; window length is a hyperparameter

#### 3. Differential Sharpe Ratio (Moody & Saffell 1998)

The original Moody-Saffell reward — a differentiable incremental Sharpe estimate:

```python
def differential_sharpe(returns, A, B, eta=0.01):
    """
    A: running sum of returns
    B: running sum of squared returns
    eta: learning rate for A, B updates
    """
    A_new = A + eta * (returns - A)
    B_new = B + eta * (returns**2 - B)
    
    D_t = (B*returns - 0.5*A*returns**2) / (B - A**2)**1.5
    
    A = A_new
    B = B_new
    return D_t  # differential Sharpe increment
```
- **Pro:** Differentiable; can be backpropped; no window needed
- **Con:** Nonstationary estimates early in training; needs warm-up period

#### 4. Risk-Adjusted Return with Drawdown Penalty

```python
def compound_reward(portfolio_value, prev_value, max_portfolio_value,
                    transaction_cost, lambda_dd=0.1, lambda_tc=1.0):
    daily_return = (portfolio_value - prev_value) / prev_value
    
    # Drawdown penalty
    current_drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value
    
    # Transaction cost penalty
    tc_penalty = lambda_tc * transaction_cost
    
    return daily_return - lambda_dd * current_drawdown - tc_penalty
```
- **Pro:** Multi-objective; directly penalises what you care about
- **Con:** Hyperparameter sensitivity; lambda tuning required

#### 5. Sortino Ratio Reward (Downside-Only Volatility)

```python
def sortino_reward(returns_window):
    downside_returns = returns_window[returns_window < 0]
    if len(downside_returns) < 2:
        return 0.0
    downside_std = np.std(downside_returns)
    if downside_std < 1e-8:
        return 0.0
    return np.mean(returns_window) / downside_std * np.sqrt(252)
```
- **Pro:** Only penalises bad volatility; better for asymmetric strategies
- **Con:** Noisier than Sharpe; less literature validation

### Reward Shaping Techniques

```python
# 1. Normalise by volatility to stabilise training
normalized_reward = raw_reward / rolling_volatility(last_100_returns)

# 2. Clip extreme rewards (prevents gradient explosions from flash crashes)
clipped_reward = np.clip(raw_reward, -0.05, 0.05)  # max 5% daily loss/gain

# 3. Add holding cost to encourage action over passivity
holding_cost = -0.0001 * abs(current_position)  # small cost to hold open positions

# 4. Exploration bonus for novel states
novelty_bonus = 0.001 * (1 - similarity(s_t, past_states))
```

### Sparse vs Dense Rewards

| Reward Type | Frequency | Example | Risk |
|---|---|---|---|
| Dense | Every timestep | Daily log return | Noisy; can obscure long-term patterns |
| Semi-dense | Per trade | Trade PnL on close | Better credit assignment |
| Sparse | End of episode | Total episode Sharpe | Very hard to learn; needs reward shaping |

**Recommendation:** Use dense daily return rewards during early training, then switch to a Sharpe-based reward as the policy improves.

### Transaction Cost Penalties

```python
# Proportional cost (most common)
tc = commission_rate * abs(trade_value)  # e.g., 0.001 = 10bps

# Market impact (for large positions)
def market_impact_cost(trade_size, avg_daily_volume, impact_coeff=0.1):
    # Almgren-Chriss style: cost ∝ (trade / ADV)^2
    participation_rate = trade_size / avg_daily_volume
    return impact_coeff * participation_rate**2 * trade_size

# Combined
def realistic_cost(trade_size, adv, commission_rate=0.001):
    return (commission_rate * abs(trade_size) + 
            market_impact_cost(abs(trade_size), adv))
```

---

## 6. State Space Design

### What Features to Include

```python
class TradingState:
    """Recommended state for a multi-asset portfolio RL agent."""
    
    # --- Price/Volume Features (per asset) ---
    price_returns: np.ndarray        # Log returns, shape (N, T_window)
    normalized_volume: np.ndarray    # Volume / 20d avg, shape (N, T_window)
    
    # --- Technical Indicators (per asset) ---
    rsi_14: np.ndarray              # RSI, normalised to [0, 1]
    macd_signal: np.ndarray         # MACD histogram normalised
    bb_pct: np.ndarray              # Bollinger %B ∈ [0, 1]
    atr_14: np.ndarray              # ATR normalised by price
    
    # --- Portfolio State ---
    current_weights: np.ndarray     # Current portfolio weights, shape (N,)
    portfolio_return: float         # Running portfolio return
    drawdown: float                 # Current drawdown from peak
    
    # --- Market Context (optional but valuable) ---
    vix: float                      # Market fear gauge
    market_return: float            # Broad market index return
    yield_curve: float              # 10y - 2y spread (recession indicator)
    
    # --- Sentiment (if using LLM) ---
    sentiment_score: np.ndarray     # LLM-extracted sentiment per asset
```

### Feature Normalisation for RL

**Critical:** RL neural networks are sensitive to input scale. Unnormalized financial data (raw prices in hundreds) will cause gradient explosions.

```python
# Option 1: Z-score normalisation (rolling)
def zscore_normalise(x, window=252):
    mean = x[-window:].mean()
    std = x[-window:].std() + 1e-8
    return (x[-1] - mean) / std

# Option 2: Min-max normalisation  
def minmax_normalise(x, window=252):
    min_val = x[-window:].min()
    max_val = x[-window:].max()
    return (x[-1] - min_val) / (max_val - min_val + 1e-8)

# Option 3: Use log returns instead of prices (naturally stationary)
log_returns = np.log(prices[1:] / prices[:-1])

# Option 4: Running statistics (for online learning)
class RunningNormalizer:
    def __init__(self):
        self.mean = 0.0; self.var = 1.0; self.count = 0
    def update(self, x):
        self.count += 1
        self.mean += (x - self.mean) / self.count
        self.var += ((x - self.mean)**2 - self.var) / self.count
    def normalise(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)
```

### Observation Window Length Tradeoffs

| Window Length | Pros | Cons |
|---|---|---|
| Short (5–10 bars) | Less computation; responds quickly | Misses medium-term trends |
| Medium (30–60 bars) | Captures momentum and mean reversion | Reasonable computation |
| Long (120–252 bars) | Captures regime changes | High computation; stale features |

**Recommendation:** Use 30–60 bar window as default. For transformer-based models (Decision Transformer), longer windows (128–256) are manageable.

### Handling Multiple Assets

```python
# Option 1: Concatenation (simple, ignores cross-asset structure)
state = np.concatenate([asset_features[i] for i in range(N_assets)])
# Shape: (N_assets * T_window * features,) — flattened vector

# Option 2: Asset matrix (preserves cross-asset structure)
state = np.stack([asset_features[i] for i in range(N_assets)])
# Shape: (N_assets, T_window, features) — use Conv1D or Transformer

# Option 3: Graph representation (captures correlations explicitly)
# Build correlation graph; use GNN to encode cross-asset dependencies
adjacency = compute_rolling_correlation(price_returns)
state = gnn_encode(asset_features, adjacency)

# Recommendation: Option 2 (matrix) + attention for 5-50 assets
# Option 1 (concatenation) for ≤ 10 assets as baseline
```

---

## 7. Action Space Design

### Discrete vs Continuous

#### Discrete Actions

```python
# Simple 3-level action space
actions = {0: "hold", 1: "buy_max", 2: "sell_max"}

# More granular
actions = {
    0: "hold",
    1: "buy_25pct",   2: "buy_50pct",   3: "buy_100pct",
    4: "sell_25pct",  5: "sell_50pct",  6: "sell_100pct",
}

# Use DQN, Double DQN, or Rainbow for discrete actions
```
- **Pro:** Simple; stable training with DQN; easy to enforce constraints
- **Con:** Cannot express fine-grained portfolio weights; curse of dimensionality with many assets

#### Continuous Actions (Recommended for Portfolio)

```python
# Single asset: position fraction
action = np.clip(actor_output, -1, 1)  # -1=max short, +1=max long

# Portfolio weights (softmax ensures sum-to-1)
raw_weights = actor_network(state)          # shape: (N+1,) including cash
weights = F.softmax(raw_weights, dim=-1)   # sum to 1, all positive

# Long-short portfolio
raw = actor_network(state)                 # shape: (N,)
weights = raw / (np.abs(raw).sum() + 1e-8) # sum of abs = 1 (unit leverage)
```

### Position Sizing Within Action Space

```python
# Fixed fractional Kelly sizing
def kelly_size(win_prob, win_return, loss_return):
    q = 1 - win_prob
    b = win_return / abs(loss_return)
    kelly_fraction = win_prob - q / b
    return np.clip(kelly_fraction, 0, 0.25)  # Never bet more than 25% Kelly

# RL with Kelly constraint: multiply raw action by Kelly fraction
action = actor_output * kelly_fraction
```

### Transaction Cost-Aware Actions

```python
# Dead zone: don't trade if rebalancing cost > expected gain
def apply_dead_zone(target_weights, current_weights, tc_rate=0.001):
    delta = target_weights - current_weights
    cost = tc_rate * np.abs(delta).sum()
    expected_gain = expected_return_improvement(target_weights, current_weights)
    
    if cost > expected_gain:
        return current_weights  # Stay put — not worth trading
    return target_weights
```

### Multi-Asset Action Space Design

```python
# Naive approach (doesn't scale):
# For 50 assets with 3 actions each: 3^50 states — impossible

# Correct approach: factored/independent per-asset actions
# Each asset gets independent continuous weight ∈ [0, 1]
# Then normalise: weights = softmax(raw_weights)

# For 100+ assets: hierarchical action decomposition
# Level 1: Select sector/asset class weights  (10 buckets)
# Level 2: Select individual asset weights within each bucket  (10 each)
```

---

## 8. Training Infrastructure

### Vectorised Environments (Critical for Speed)

```python
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

def make_env(stock_data, rank):
    def _init():
        env = StockTradingEnv(stock_data, seed=rank)
        return env
    return _init

# 8 parallel environments = 8x throughput
n_envs = 8
env = SubprocVecEnv([make_env(data, i) for i in range(n_envs)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2_000_000)
```

### Experience Replay for Financial Data

```python
# Standard replay buffer (uniform sampling)
from collections import deque
import random

replay_buffer = deque(maxlen=100_000)
replay_buffer.append((state, action, reward, next_state, done))

# Prioritised Experience Replay (PER) — more important for finance
# Rare market events (crashes) should be sampled more often
class PrioritisedReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.priorities = np.zeros(capacity)
        self.alpha = alpha  # Prioritisation exponent
        self.beta = beta    # Importance sampling correction
    
    def add(self, experience, td_error):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        # Store with high priority for surprising experiences
        ...
```

### Data Leakage / Temporal Ordering

**The most common catastrophic mistake in financial RL:**

```python
# WRONG: Random shuffle splits data (uses future to train on past)
X_train, X_test = train_test_split(data, test_size=0.2, shuffle=True)

# CORRECT: Strict temporal split
n = len(data)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
X_train = data[:train_end]      # e.g., 2015-2021
X_val   = data[train_end:val_end]  # e.g., 2021-2022
X_test  = data[val_end:]            # e.g., 2022-present

# ALSO WRONG: Feature computation that leaks future
# RSI computed on ALL data including test period? → Leak!
# Normalisation using global mean/std? → Leak!

# CORRECT: Compute features using only past data at each point
# Use rolling/expanding windows for normalization
```

### Walk-Forward Validation for RL

```
Traditional CV:  [train|val|test] (fixed splits)
Walk-Forward RL: [train₁|val₁] → [train₂|val₂] → [test]

Example:
  Fold 1: Train 2015-2018, Validate 2019
  Fold 2: Train 2015-2019, Validate 2020
  Fold 3: Train 2015-2020, Validate 2021
  Final:  Train 2015-2021, Test 2022-present

Report average validation Sharpe across folds.
Never retrain on validation; never peek at test.
```

### Hyperparameter Tuning with Optuna

```python
import optuna
from stable_baselines3 import SAC

def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    buffer_size = trial.suggest_int("buffer_size", 50_000, 500_000)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    gamma = trial.suggest_uniform("gamma", 0.95, 0.999)
    
    env = StockTradingEnv(train_data)
    model = SAC("MlpPolicy", env, 
                learning_rate=lr,
                buffer_size=buffer_size,
                batch_size=batch_size,
                gamma=gamma)
    
    model.learn(200_000)
    
    # Evaluate on VALIDATION set (not test!)
    val_env = StockTradingEnv(val_data)
    sharpe = evaluate_sharpe(model, val_env, n_episodes=10)
    return sharpe

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, n_jobs=4)
```

### Curriculum Learning for Trading

Start with easy environments, gradually increase complexity:

```python
# Stage 1: Single asset, no transaction costs, trending market
env_1 = StockEnv(n_assets=1, tc_rate=0.0, market="trending")

# Stage 2: Single asset, with transaction costs
env_2 = StockEnv(n_assets=1, tc_rate=0.001, market="mixed")

# Stage 3: 5 assets, realistic costs
env_3 = StockEnv(n_assets=5, tc_rate=0.001, market="realistic")

# Stage 4: 30 assets, full complexity including macro
env_4 = StockEnv(n_assets=30, tc_rate=0.001, market="full", add_macro=True)

# Transfer learn each stage: start from previous checkpoint
model = SAC(...)
for env in [env_1, env_2, env_3, env_4]:
    model.set_env(env)
    model.learn(total_timesteps=500_000)
```

---

## 9. Best Libraries & Frameworks

| Library | Type | Algorithms | Pros | Cons | Best Use Case |
|---|---|---|---|---|---|
| **stable-baselines3** | RL algorithms | DQN, A2C, PPO, SAC, TD3, DDPG | Excellent docs; clean API; production quality | No built-in financial envs | General RL training with custom gym envs |
| **FinRL** | RL for finance | All SB3 + ElegantRL | Ready-made stock envs; data pipelines; backtesting | Abstraction can hide bugs; less flexible | Quick stock trading prototyping |
| **RLlib (Ray)** | Distributed RL | 30+ algorithms | Scales to cluster; async training; multi-agent | Complex API; heavy dependency | Large-scale training; multi-agent market simulation |
| **TensorTrade** | RL trading framework | SB3 compatible | Modular exchange/action/reward components; crypto support | Smaller community; less maintained | Crypto trading; exchange simulation |
| **d3rlpy** | Offline RL | CQL, IQL, Decision Transformer, BC | Best offline RL library; clean API | Online RL less supported | Learning from historical trade data without live trading |
| **ElegantRL** | High-performance RL | All major + GPU-vectorised | Fastest GPU training; population-based | Less documented; smaller community | Research requiring GPU-accelerated training |
| **Custom PyTorch** | Custom | Anything | Full control; no abstractions | High implementation cost | Novel architectures; research |

### Quick Setup: SB3 + Custom Trading Env

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

class StockPortfolioEnv(gym.Env):
    def __init__(self, prices, tc_rate=0.001, window=30):
        self.prices = prices          # shape: (T, N_assets)
        self.tc_rate = tc_rate
        self.window = window
        self.N = prices.shape[1]
        
        # Portfolio weights ∈ [0, 1]^N, sum to 1
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.N,), dtype=np.float32
        )
        # State: price returns + current weights
        obs_dim = self.N * window + self.N
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
    
    def reset(self, seed=None):
        self.t = self.window
        self.weights = np.ones(self.N) / self.N
        self.portfolio_value = 1.0
        return self._get_obs(), {}
    
    def step(self, action):
        # Normalise action to valid portfolio weights
        new_weights = action / (action.sum() + 1e-8)
        
        # Transaction costs
        tc = self.tc_rate * np.abs(new_weights - self.weights).sum()
        
        # Portfolio return
        returns = self.prices[self.t] / self.prices[self.t-1] - 1
        portfolio_return = (new_weights * returns).sum() - tc
        
        self.weights = new_weights
        self.portfolio_value *= (1 + portfolio_return)
        self.t += 1
        
        reward = portfolio_return  # or Sharpe-based
        done = self.t >= len(self.prices) - 1
        return self._get_obs(), reward, done, False, {}
    
    def _get_obs(self):
        returns = np.log(self.prices[self.t-self.window:self.t] / 
                         self.prices[self.t-self.window-1:self.t-1])
        return np.concatenate([returns.flatten(), self.weights])

# Train
env = StockPortfolioEnv(train_prices)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(500_000)
```

---

## 10. Top Papers Ranked & Annotated

### Must-Read Papers (Ordered by Importance for Building a Trading Bot)

| Rank | Paper | Why It Matters | What to Implement | Difficulty |
|---|---|---|---|---|
| 1 | **FinRL** (Liu et al. 2020, arXiv:2011.09607) | The complete trading RL toolkit | Use directly; understand its env design | Easy |
| 2 | **SAC** (Haarnoja et al. 2018, arXiv:1801.01290) | Best algorithm for continuous portfolio | Implement or use SB3's SAC | Medium |
| 3 | **PPO** (Schulman et al. 2017, arXiv:1707.06347) | Most stable; widely validated in finance | Use SB3's PPO as baseline | Easy |
| 4 | **Deep Hedging** (Bühlеr et al. 2019, arXiv:1802.03042) | Defines RL hedging paradigm | Implement hedging env with CVaR reward | Hard |
| 5 | **DQN Nature** (Mnih et al. 2015) | Foundation of all deep RL | Understand experience replay + target net | Easy |
| 6 | **EIIE** (Jiang et al. 2017, arXiv:1706.10059) | Purpose-built portfolio architecture | Implement EIIE with portfolio vector memory | Medium |
| 7 | **Decision Transformer** (Chen et al. 2021, arXiv:2106.01345) | Offline RL from trade logs | Use d3rlpy; condition on desired Sharpe | Medium |
| 8 | **TD3** (Fujimoto et al. 2018, arXiv:1802.09477) | More stable than DDPG | SB3's TD3; compare to SAC | Easy |
| 9 | **CQL** (Kumar et al. 2020, arXiv:2006.04779) | Safe offline RL from historical data | Use d3rlpy's CQL | Medium |
| 10 | **Market Making RL** (Spooner et al. 2018, arXiv:1804.04216) | Defines MM RL formulation | Build LOB sim + TD reward | Hard |
| 11 | **A3C** (Mnih et al. 2016, arXiv:1602.01783) | Parallel training architecture | Use SB3's A2C | Easy |
| 12 | **Deep Hedging: Gamma/Vega** (Cao et al. 2022, arXiv:2205.05614) | Extends deep hedging to Greeks | D4PG with CVaR objective | Very Hard |
| 13 | **AlphaPortfolio** (Cong et al. 2021, SSRN:3554486) | Cross-asset attention for portfolios | Implement multi-head attention over assets | Medium |
| 14 | **RL Survey: Evolution in Quant Finance** (Hambly et al. 2023) | Best comprehensive survey | Read before implementing anything | Easy |
| 15 | **MBPO** (Janner et al. 2019, arXiv:1906.08253) | 20-40x sample efficiency via market model | Learn price dynamics model + train SAC on it | Hard |

---

## 11. Recommended Implementation Path

### Phase 1: Basic DQN Trading Environment (Week 1–2)

**Goal:** Working DQN agent that trades a single stock and beats buy-and-hold.

```python
# Step 1: Create gym environment
class SingleStockEnv(gym.Env):
    """Minimal single-stock DQN environment."""
    action_space = gym.spaces.Discrete(3)     # hold, buy, sell
    observation_space = gym.spaces.Box(...)   # price features
    
# Step 2: Train DQN with SB3
from stable_baselines3 import DQN
model = DQN("MlpPolicy", env, verbose=1)
model.learn(100_000)

# Step 3: Backtest with realistic transaction costs (0.1%)
# Verify: does agent outperform buy-and-hold on validation set?
```

**Verify:** Agent achieves positive Sharpe ratio on UNSEEN validation data.

---

### Phase 2: SAC for Continuous Portfolio Weights (Week 3–4)

**Goal:** SAC agent managing 5–10 asset portfolio with continuous weights.

```python
# Step 1: Build multi-asset portfolio environment
# - Action: portfolio weight vector (softmax normalised)
# - Reward: daily log return - transaction costs
# - State: 30-bar return history per asset + current weights

# Step 2: Implement walk-forward validation
# Train: 2015-2020, Validate: 2020-2021, Test: 2022-present

# Step 3: Train SAC
from stable_baselines3 import SAC
model = SAC("MlpPolicy", env, verbose=1, 
            buffer_size=500_000, batch_size=256)
model.learn(1_000_000)

# Step 4: Compare to equal-weight and min-variance baselines
```

**Verify:** Out-of-sample Sharpe > 1.0; max drawdown < 20%.

---

### Phase 3: Realistic Market Simulation (Week 5–6)

**Goal:** Add realistic frictions that make the problem harder and the solution more deployable.

```python
# Additions to environment:
class RealisticPortfolioEnv(gym.Env):
    def step(self, action):
        # 1. Slippage model
        slippage = compute_slippage(action, avg_daily_volume, 
                                    impact_coeff=0.1)
        
        # 2. Bid-ask spread cost
        spread_cost = bid_ask_spread * trade_size / 2
        
        # 3. Market impact (permanent)
        price_impact = permanent_impact(trade_size, avg_daily_volume)
        
        # 4. Overnight gap risk (for multi-day positions)
        gap_return = compute_overnight_gap(position)
        
        total_cost = slippage + spread_cost + market_impact
        ...

# Add: turbulence detection (from FinRL)
def compute_turbulence(returns, historical_cov):
    """Mahalanobis distance — reduce exposure in high turbulence."""
    diff = returns - historical_mean
    turbulence = diff @ np.linalg.inv(historical_cov) @ diff.T
    return float(turbulence)
```

**Verify:** Strategy still profitable after all realistic costs; doesn't overtrade.

---

### Phase 4: Add LLM / Sentiment Signals as State Features (Week 7–8)

**Goal:** Incorporate text-based sentiment signals to improve state representation.

```python
# Option A: FinGPT sentiment scores
from fingpt import FinGPT  # or use pre-computed signals

def get_sentiment_features(ticker, date):
    """Get LLM sentiment score for ticker on date."""
    news = fetch_financial_news(ticker, date, lookback_days=3)
    sentiment = fingpt_model.analyse(news)
    return {
        "sentiment_score": sentiment.score,     # -1 to +1
        "sentiment_confidence": sentiment.conf,  # 0 to 1
        "event_detected": sentiment.has_event,  # earnings, merger, etc.
    }

# Augment state
state = np.concatenate([
    price_features,        # Original features
    sentiment_features,    # LLM-derived features
    macro_features,        # VIX, yield curve
    current_weights,       # Portfolio state
])

# Option B: Lighter approach — pre-compute sentiment index
# Run FinBERT/FinGPT on all historical news → build feature dataset
# Use as additional columns in your data pipeline
```

**Verify:** Sentiment features improve Sharpe by at least 0.2 vs price-only on validation set.

---

### Phase 5: Ensemble and Live Deployment (Week 9–12)

**Goal:** Ensemble multiple trained agents; deploy with safeguards.

```python
# Ensemble approach (FinRL ensemble strategy)
class EnsembleAgent:
    def __init__(self, agents, weights=None):
        self.agents = agents  # [PPO, SAC, TD3, ...]
        self.weights = weights or [1/len(agents)] * len(agents)
    
    def act(self, state):
        actions = [agent.predict(state)[0] for agent in self.agents]
        # Weighted average of portfolio weight vectors
        ensemble_action = sum(w * a for w, a in zip(self.weights, actions))
        return ensemble_action / ensemble_action.sum()  # renormalise
    
    def update_weights(self, recent_performance):
        """Reweight agents based on recent performance."""
        perfs = np.array([agent.recent_sharpe() for agent in self.agents])
        perfs = np.clip(perfs - perfs.min() + 1e-6, 0, None)
        self.weights = perfs / perfs.sum()

# Live deployment safeguards
class LiveTradingWrapper:
    MAX_POSITION_SIZE = 0.25      # Max 25% in any single asset
    MAX_DAILY_TURNOVER = 0.20    # Max 20% portfolio change per day
    TURBULENCE_THRESHOLD = 200   # Switch to cash if market turbulent
    
    def safe_act(self, state, turbulence):
        if turbulence > self.TURBULENCE_THRESHOLD:
            return all_cash_portfolio()  # Protect capital
        
        raw_action = self.agent.act(state)
        clipped = np.clip(raw_action, 0, self.MAX_POSITION_SIZE)
        clipped /= clipped.sum()
        
        current = self.current_weights()
        delta = clipped - current
        
        # Limit daily turnover
        if np.abs(delta).sum() > self.MAX_DAILY_TURNOVER:
            scale = self.MAX_DAILY_TURNOVER / np.abs(delta).sum()
            clipped = current + delta * scale
        
        return clipped
```

---

## 12. Common Pitfalls & How to Avoid Them

### Pitfall 1: Training on the Full Dataset

**Problem:** Accidentally using test-period data during training (feature normalization, indicator calculation, model selection).

**Solution:** Implement strict temporal splits. All feature engineering must be "point-in-time" — use only data available at that timestamp.

```python
# Always use ExpandingWindow or RollingWindow transforms
scaler = StandardScaler()
for t in range(window, len(data)):
    scaler.fit(data[:t])  # Only past data
    features[t] = scaler.transform(data[t:t+1])
```

### Pitfall 2: Forgetting Transaction Costs

**Problem:** Agent achieves incredible Sharpe ratio in backtesting but loses money live because it trades thousands of times per day.

**Solution:** Model transaction costs explicitly. Use at minimum 10bps round-trip (buy + sell). Add 5bps slippage. Monitor trade frequency: > 2 trades/day suggests overtrading.

```python
# Always include in reward
reward = portfolio_return - tc_rate * abs(action - prev_action).sum()
# Monitor: log average trades per day during evaluation
```

### Pitfall 3: Reward Hacking / Objective Mismatch

**Problem:** Agent maximises your reward proxy rather than actual objective. Example: maximises Sharpe by refusing to trade (near-zero variance → huge Sharpe on any tiny positive return).

**Solution:** Test your reward function against trivial policies. If a "hold everything" policy scores well, your reward is broken. Add a minimum activity bonus or use raw return rewards.

```python
# Test: evaluate reward of always-hold policy
# If Sharpe(hold) > 1.5, your reward function has a bug
```

### Pitfall 4: Non-Stationarity Blindness

**Problem:** Policy trained on 2015–2020 fails completely in 2022 because market regime changed (rising rates, inflation).

**Solution:**
- Use shorter training windows with frequent retraining (rolling 1–2 year windows)
- Add market regime features to state (VIX, yield curve slope)
- Consider online adaptation (train on recent data, fine-tune weekly)
- Monitor live performance vs backtest performance gap — if >30% drop, retrain

### Pitfall 5: Ignoring the Exploration/Exploitation Tradeoff

**Problem:** Agent learns a conservative policy early and never explores better strategies; gets stuck in local optima.

**Solution:**
- SAC's entropy bonus naturally addresses this
- For DQN: use NoisyNet instead of epsilon-greedy
- For PPO: increase entropy coefficient `ent_coef` during early training
- Use curriculum learning: start with high exploration, reduce over time

### Pitfall 6: Single Environment Overfitting

**Problem:** Agent overfits to quirks of one stock or one time period.

**Solution:**
- Train on multiple assets simultaneously
- Use different random seeds and report mean ± std across seeds
- Evaluate on assets NOT seen during training
- Augment data: add noise, shuffle assets, use bootstrap sampling

### Pitfall 7: Wrong Gamma (Discount Factor)

**Problem:** Low gamma (0.9) makes agent myopic — optimises for next-day returns, ignores multi-day trends. High gamma (1.0) may make training unstable.

**Solution:** For daily trading, use γ = 0.99–0.999. For intraday trading (HFT), γ = 0.9–0.99. Test sensitivity to gamma in hyperparameter search.

### Pitfall 8: Survivorship Bias in Training Data

**Problem:** Training on S&P 500 *current* constituents uses only stocks that survived and grew — companies that went bankrupt are not included. This inflates backtest performance.

**Solution:** Use point-in-time index membership data. Include delisted stocks in training universe. Use services like Compustat or Bloomberg PIT data.

### Pitfall 9: Lookahead Bias in Feature Engineering

**Problem:** Computing a feature like "annual volatility" using the full year's data gives the agent future information. Computing RSI using next week's prices is an obvious bug; using future data for normalization is a subtle one.

**Solution:** Audit every feature for temporal consistency. Use only data available at time `t` when computing `feature_t`. Add a test: inject garbage into future data — performance should drop.

### Pitfall 10: Insufficient Baseline Comparisons

**Problem:** Your RL agent "beats buy-and-hold" but the comparison is meaningless if buy-and-hold itself has a 2.5 Sharpe during your test period (2010–2021 bull market).

**Solution:** Always compare against:
1. Buy-and-hold (each individual asset)
2. Equal-weight portfolio (rebalanced daily/monthly)
3. Minimum-variance portfolio (Markowitz)
4. A moving average strategy (60-day SMA crossover)
5. A random trading policy

If your RL agent doesn't beat ALL of these, it probably hasn't learned anything meaningful.

---

## 13. Cutting Edge (2023–2025)

### LLM + RL Hybrid Approaches

The frontier is combining LLM reasoning capabilities with RL's sequential decision-making:

**Architecture 1: LLM as State Feature Extractor**
```
News/Filings → LLM → Sentiment Embeddings → RL State → RL Agent → Trade
```
Use FinGPT or FinBERT to convert unstructured text into dense sentiment vectors that augment the RL state space. Validated in Cao et al. (2024), arXiv:2510.10526.

**Architecture 2: LLM as High-Level Planner**
```
Market Regime Context → LLM → Strategy Guidelines → RL Agent (low-level executor)
```
LLM analyzes macro context and outputs natural language trading guidelines (e.g., "market is in risk-off; reduce equity exposure"). RL agent translates these into specific positions. See arXiv:2508.02366.

**Architecture 3: RL Fine-Tuning LLMs (RLHF for Finance)**
```
Base LLM → RLHF with financial expert feedback → FinGPT → Trading signals
```
FinGPT (arXiv:2306.06031) uses RLHF to align LLMs with financial expert preferences for sentiment analysis and trading recommendation tasks.

**Current limitations:**
- LLM inference latency (200ms–2s) is too slow for HFT but fine for daily/weekly rebalancing
- LLMs hallucinate financial facts; require fact-checking layer
- LLM output calibration for trading is still an open problem

---

### Offline RL / Conservative Q-Learning for Finance

Offline RL is rapidly becoming practical for finance because it enables learning from historical data **without any live trading risk**:

```
Traditional RL:  Need to explore live → risky for capital
Offline RL:      Learn from historical data only → safe

CQL (arXiv:2006.04779):
- Penalises Q-values for actions not seen in historical data
- Conservative Q̂(s,a) = Q(s,a) - α·penalty(s,a,π_behavior)
- Prevents the agent from taking overconfident actions in novel situations

Decision Transformer (arXiv:2106.01345):
- Even simpler: just supervised learning on (state, action, return_to_go) triplets
- No Q-values; no reward hacking; very stable
```

**Finance-specific application pipeline:**
```
1. Collect historical trades from existing quant system
2. Label each trade with return-to-go (actual future PnL)
3. Train CQL or Decision Transformer offline
4. Deploy with conservative position limits
5. Gradually allow online fine-tuning as confidence grows
```

---

### Multi-Agent RL for Market Simulation

Markets are inherently multi-agent systems. MARL is increasingly used for:

1. **Market simulation:** Generate realistic synthetic market data by simulating multiple competing agents (market makers, HFTs, institutional investors, retail traders)

2. **Strategy stress-testing:** Test your RL strategy against adversarial agents that try to front-run or exploit it

3. **Order book reconstruction:** Learn to generate realistic LOB dynamics for training execution algorithms

Key papers/tools:
- ABIDES simulator (from JP Morgan) — open-source market simulation with multiple agent types
- Karpe et al. 2020 (arXiv:2006.05574) — MARL in realistic LOB with ABIDES
- JaxMARL-HFT (arXiv:2511.02136) — GPU-accelerated MARL for HFT at massive scale

---

### World Models for Financial Markets

Learning a generative model of market dynamics enables:
1. Synthetic data generation for data-limited training
2. Planning ahead (imagine potential market scenarios)
3. Counterfactual reasoning (what if I had held?)

```
Market World Model:
    Input:  (prices_t, volume_t, action_t)
    Output: (prices_{t+1}, volume_{t+1}, reward_t)

Training: learn P(s_{t+1} | s_t, a_t) from historical data
Inference: sample synthetic trajectories → train RL agent in simulation
```

Applications emerging in 2024–2025:
- DreamerV3-style latent world models for financial market simulation
- GAN/diffusion-based price path generators for RL training data augmentation
- Transformer world models conditioned on macro economic variables

---

## Quick Reference: Algorithm Selection Guide

```
Is your action space discrete (buy/sell signals)?
  → DQN (simple) or Rainbow DQN (best discrete)
  → DRQN if temporal dependencies are critical

Is your action space continuous (portfolio weights)?
  → SAC (best overall, especially if sample efficiency matters)
  → PPO (most stable, best default)
  → TD3 (if SAC is too slow to converge)

Do you have only historical data (no live trading)?
  → Decision Transformer (simplest offline RL)
  → CQL (strongest offline RL)

Do you need to hedge derivatives?
  → Deep Hedging (Bühlеr et al.) with CVaR reward + FFNN/LSTM actor

Do you need market making?
  → DDPG with continuous bid/ask offsets
  → TD learning with Avellaneda-Stoikov as warm-start

Do you have limited compute?
  → PPO with vectorized environments
  
Do you have high compute?
  → SAC with large replay buffer (1M+)
  → Try MBPO for 20-40x sample efficiency

Do you want to incorporate news/sentiment?
  → Any algorithm + FinGPT/FinBERT features in state space
```

---

*For the complete bibliography and Excel reference, see `rl_finance_papers.xlsx` in this directory.*
