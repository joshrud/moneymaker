"""
Sentiment scoring: FinBERT (local, free) and Claude API (reasoning-grade).
Both return a float in [-1.0, +1.0]: -1 = very bearish, +1 = very bullish.
"""
from __future__ import annotations
import os
from typing import Optional


# ── FinBERT ───────────────────────────────────────────────────────────────────

_finbert_pipeline = None  # lazy-loaded on first call to avoid startup cost
_finbert_lock = __import__("threading").Lock()


def _load_finbert():
    global _finbert_pipeline
    with _finbert_lock:
        if _finbert_pipeline is None:
            from transformers import pipeline
            # ProsusAI/finbert: 3-class (positive/negative/neutral) financial sentiment
            _finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,
                truncation=True,
                max_length=512,
            )
    return _finbert_pipeline


def finbert_score(texts: list[str]) -> float:
    """
    Scores a list of news headlines with FinBERT.
    Returns average sentiment in [-1, +1].
    Positive label → +score, Negative → -score, Neutral → 0.
    """
    if not texts:
        return 0.0
    pipe = _load_finbert()
    results = pipe(texts)
    scores = []
    for result in results:
        label_scores = {r["label"].lower(): r["score"] for r in result}
        score = label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0)
        scores.append(score)
    return float(sum(scores) / len(scores))


# ── Claude API ────────────────────────────────────────────────────────────────

def claude_sentiment_score(
    headlines: list[str], symbol: str, api_key: Optional[str] = None
) -> tuple[float, str]:
    """
    Asks Claude to reason over news headlines for a given symbol.
    Returns (score in [-1,+1], explanation string).
    Falls back to FinBERT if ANTHROPIC_API_KEY is not set.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return finbert_score(headlines), "FinBERT fallback (no ANTHROPIC_API_KEY)"

    import anthropic
    client = anthropic.Anthropic(api_key=key)

    headline_text = "\n".join(f"- {h}" for h in headlines[:15])
    prompt = f"""You are a quantitative analyst scoring news sentiment for {symbol}.

Headlines:
{headline_text}

Respond with ONLY valid JSON in this exact format:
{{"score": <float between -1.0 and 1.0>, "conviction": <"high"|"medium"|"low">, "reason": "<one sentence>"}}

Where score: -1.0 = strongly bearish, 0 = neutral, +1.0 = strongly bullish.
Consider earnings impact, macro relevance, and novelty vs noise."""

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",  # fast + cheap for signal generation
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    import json, re
    try:
        raw = msg.content[0].text.strip()
        # Strip markdown code fences if Claude wraps the JSON
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        reason = data.get("reason", "")
        return max(-1.0, min(1.0, score)), reason
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0, "parse error"


def score_watchlist_sentiment(
    news_items: list, watchlist: list[str], use_claude: bool = False
) -> dict[str, float]:
    """
    Groups news by symbol and scores each.
    Returns {symbol: sentiment_score}.
    news_items: list of Alpaca NewsV2 objects (have .symbols and .headline attributes).
    """
    from collections import defaultdict
    symbol_headlines: dict[str, list[str]] = defaultdict(list)

    for item in news_items:
        # Items arrive as dicts from the Alpaca NewsClient
        if isinstance(item, dict):
            headline = item.get("headline", "")
            symbols = item.get("symbols", watchlist)
        else:
            headline = getattr(item, "headline", str(item))
            symbols = getattr(item, "symbols", watchlist)
        for sym in symbols:
            if sym in watchlist:
                symbol_headlines[sym].append(headline)

    scores = {}
    for sym in watchlist:
        headlines = symbol_headlines.get(sym, [])
        if not headlines:
            scores[sym] = 0.0
            continue
        if use_claude:
            score, _ = claude_sentiment_score(headlines, sym)
        else:
            score = finbert_score(headlines)
        scores[sym] = score
    return scores
