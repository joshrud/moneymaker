"""
Generates a daily markdown report comparing all 5 bots.
Called by scripts/daily_report.py (cron) and by the main runner at market close.
"""
from __future__ import annotations
from datetime import date, datetime
from pathlib import Path

from core.logger import TradeLogger, LOGS_DIR

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
BOT_NAMES = ["bot1_momentum", "bot2_sac_rl", "bot3_sentiment_claude",
             "bot4_finbert_ppo", "bot5_ensemble"]


def _summarise_bot(bot_name: str, dt: date) -> dict:
    """Extracts stats from a bot's JSONL log for the given date."""
    logger = TradeLogger(bot_name)
    records = logger.read_date(dt)

    trades = [r for r in records if r.get("event") == "trade"]
    summaries = [r for r in records if r.get("event") == "summary"]

    buys = [t for t in trades if t.get("side") == "buy"]
    sells = [t for t in trades if t.get("side") == "sell"]

    # Use last summary record for final portfolio state
    last_summary = summaries[-1] if summaries else {}

    portfolio_value = last_summary.get("portfolio_value", 20_000.0)
    daily_pnl = last_summary.get("daily_pnl", 0.0)
    sharpe = last_summary.get("sharpe", 0.0)
    max_dd = last_summary.get("max_drawdown", 0.0)

    return {
        "bot": bot_name,
        "trades": len(trades),
        "buys": len(buys),
        "sells": len(sells),
        "portfolio_value": portfolio_value,
        "daily_pnl": daily_pnl,
        "daily_pnl_pct": (daily_pnl / 20_000.0) * 100,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def generate_report(dt: date = None) -> str:
    """Builds and saves the daily markdown report. Returns the file path."""
    dt = dt or date.today()
    REPORTS_DIR.mkdir(exist_ok=True)

    stats = [_summarise_bot(name, dt) for name in BOT_NAMES]
    stats_sorted = sorted(stats, key=lambda x: x["daily_pnl"], reverse=True)

    lines = [
        f"# Daily Trading Report — {dt.isoformat()}",
        f"*Generated at {datetime.utcnow().strftime('%H:%M:%S UTC')}*",
        "",
        "---",
        "",
        "## Bot Performance Summary",
        "",
        "| Rank | Bot | Trades | Daily PnL | Daily PnL % | Portfolio Value | Sharpe | Max Drawdown |",
        "|------|-----|--------|-----------|-------------|-----------------|--------|--------------|",
    ]

    medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, s in enumerate(stats_sorted):
        pnl_sign = "+" if s["daily_pnl"] >= 0 else ""
        lines.append(
            f"| {medal[i]} | **{s['bot']}** | {s['trades']} "
            f"| {pnl_sign}${s['daily_pnl']:,.2f} "
            f"| {pnl_sign}{s['daily_pnl_pct']:.2f}% "
            f"| ${s['portfolio_value']:,.2f} "
            f"| {s['sharpe']:.3f} "
            f"| {s['max_drawdown']:.2f}% |"
        )

    winner = stats_sorted[0]
    lines += [
        "",
        "---",
        "",
        f"## Today's Winner: **{winner['bot']}**",
        "",
        f"- Daily PnL: **${winner['daily_pnl']:+,.2f}** ({winner['daily_pnl_pct']:+.2f}%)",
        f"- Portfolio value: **${winner['portfolio_value']:,.2f}**",
        f"- Total trades executed: **{winner['trades']}**",
        "",
        "---",
        "",
        "## Per-Bot Detail",
        "",
    ]

    for s in stats_sorted:
        lines += [
            f"### {s['bot']}",
            f"- **Trades:** {s['trades']} ({s['buys']} buys / {s['sells']} sells)",
            f"- **Daily PnL:** ${s['daily_pnl']:+,.2f} ({s['daily_pnl_pct']:+.2f}%)",
            f"- **Portfolio value:** ${s['portfolio_value']:,.2f}",
            f"- **Sharpe (daily):** {s['sharpe']:.3f}",
            f"- **Max drawdown:** {s['max_drawdown']:.2f}%",
            "",
        ]

    report_text = "\n".join(lines)
    out_path = REPORTS_DIR / f"{dt.isoformat()}_report.md"
    out_path.write_text(report_text)
    return str(out_path)
