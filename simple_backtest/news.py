"""Historical news retrieval and point-in-time feature construction.

GDELT DOC 2.0 is used as an external news source rather than treating the
market-data provider as a news database.  Its ``seendate`` is the timestamp at
which GDELT observed an article, which is the availability timestamp used by
this module.  That is deliberately more conservative than using a later
retrieval time or an article's webpage metadata.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import sqrt
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


@dataclass(frozen=True)
class NewsArticle:
    """A news article with its historical availability timestamp."""

    title: str
    available_at: pd.Timestamp
    url: str = ""
    source_country: str = ""


class GDELTNewsProvider:
    """Small dependency-free client for GDELT DOC article-list queries."""

    def __init__(self, base_url: str = "http://api.gdeltproject.org/api/v2/doc/doc") -> None:
        self.base_url = base_url

    def fetch(
        self,
        query: str,
        start: datetime,
        end: datetime,
        max_records: int = 250,
        timeout: int = 30,
    ) -> list[NewsArticle]:
        """Fetch articles observed in ``[start, end]`` with availability timestamps."""
        if not query.strip():
            raise ValueError("query must not be blank")
        if max_records < 1 or max_records > 250:
            raise ValueError("max_records must be between 1 and 250")
        start_utc = _as_utc(start)
        end_utc = _as_utc(end)
        if start_utc >= end_utc:
            raise ValueError("start must be before end")
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": max_records,
            "startdatetime": start_utc.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end_utc.strftime("%Y%m%d%H%M%S"),
            "sort": "datedesc",
        }
        request = Request(
            f"{self.base_url}?{urlencode(params)}",
            headers={"User-Agent": "simple-backtest-historical-news/1.0"},
        )
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed provider URL.
            raw_payload = response.read().decode("utf-8")
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "GDELT returned a non-JSON response; the endpoint may be rate-limited or unavailable"
            ) from exc
        articles = []
        for item in payload.get("articles", []):
            seen = item.get("seendate")
            title = str(item.get("title", "")).strip()
            if not seen or not title:
                continue
            try:
                available_at = pd.to_datetime(seen, utc=True)
            except (TypeError, ValueError):
                continue
            articles.append(
                NewsArticle(
                    title=title,
                    available_at=available_at,
                    url=str(item.get("url", "")),
                    source_country=str(item.get("sourcecountry", "")),
                )
            )
        return articles


def fetch_historical_news(
    query: str,
    start: str | datetime,
    end: str | datetime,
    *,
    window_days: int = 31,
    max_records_per_window: int = 250,
    pause_seconds: float = 0.25,
    provider: GDELTNewsProvider | None = None,
) -> list[NewsArticle]:
    """Fetch a bounded history in windows and deduplicate by URL/timestamp/title."""
    if window_days < 1:
        raise ValueError("window_days must be positive")
    begin = _as_utc(pd.Timestamp(start).to_pydatetime())
    finish = _as_utc(pd.Timestamp(end).to_pydatetime())
    if begin >= finish:
        raise ValueError("start must be before end")
    provider = provider or GDELTNewsProvider()
    all_articles: list[NewsArticle] = []
    cursor = begin
    while cursor < finish:
        window_end = min(cursor + timedelta(days=window_days), finish)
        all_articles.extend(
            provider.fetch(
                query,
                cursor,
                window_end,
                max_records=max_records_per_window,
            )
        )
        cursor = window_end
        if cursor < finish and pause_seconds:
            time.sleep(pause_seconds)
    unique: dict[tuple[str, pd.Timestamp, str], NewsArticle] = {}
    for article in all_articles:
        key = (article.url, article.available_at, article.title)
        unique[key] = article
    return sorted(unique.values(), key=lambda article: article.available_at)


_POSITIVE_WORDS = {
    "beat", "beats", "bullish", "buy", "growth", "gain", "gains", "improve",
    "improved", "launch", "record", "strong", "surge", "upgrade", "upside",
    "profit", "profits", "positive", "success", "successful",
}
_NEGATIVE_WORDS = {
    "bearish", "cut", "cuts", "decline", "declines", "downgrade", "drop", "falls",
    "fraud", "investigation", "loss", "losses", "lawsuit", "miss", "misses", "negative",
    "probe", "recall", "risk", "risks", "scandal", "slump", "weak", "warning",
}


def score_title(title: str) -> float:
    """Return a transparent lexicon sentiment score in [-1, 1]."""
    words = {word.strip(".,:;!?()[]{}\"'").lower() for word in title.split()}
    positive = len(words & _POSITIVE_WORDS)
    negative = len(words & _NEGATIVE_WORDS)
    total = positive + negative
    if total == 0:
        return 0.0
    return float(max(-1.0, min(1.0, (positive - negative) / max(1, sqrt(total)))))


def build_daily_news_signal(articles: Iterable[NewsArticle]) -> pd.DataFrame:
    """Build an event-level signal stream keyed by GDELT availability timestamps.

    The resulting event stream preserves each article's exact GDELT
    availability timestamp. ``NewsAwareRSIStrategy`` applies a recent-event
    window at the true decision timestamp, so an article seen at noon cannot
    affect a morning decision on the same date.
    """
    rows = []
    for article in articles:
        available_at = pd.Timestamp(article.available_at)
        if available_at.tzinfo is None:
            available_at = available_at.tz_localize("UTC")
        else:
            available_at = available_at.tz_convert("UTC")
        rows.append(
            {
                "available_at": available_at,
                "available_date": available_at.normalize(),
                "article_score": score_title(article.title),
                "title": article.title,
                "url": article.url,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["news_signal", "article_count", "available_date"])
    frame = pd.DataFrame(rows).set_index("available_at").sort_index()
    frame["article_count"] = 1
    return frame[["article_score", "article_count", "available_date"]].rename(
        columns={"article_score": "news_signal"}
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "GDELTNewsProvider",
    "NewsArticle",
    "build_daily_news_signal",
    "fetch_historical_news",
    "score_title",
]
