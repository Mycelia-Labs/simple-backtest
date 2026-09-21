"""Tests for the RSI/news signal integration."""

from datetime import datetime, timezone
from io import BytesIO
from urllib.error import HTTPError

import pandas as pd

from simple_backtest import Backtest, BacktestConfig
from simple_backtest.news import (
    GDELTNewsProvider,
    NewsArticle,
    build_daily_news_signal,
    score_title,
)
from simple_backtest.strategy.rsi_news import NewsAwareRSIStrategy, RSIStrategy, calculate_rsi


def _state(shares: float = 0.0) -> dict:
    return {
        "cash": 10_000.0,
        "total_shares": shares,
        "portfolio_value": 10_000.0,
        "positions": {},
        "current_price": 100.0,
        "is_last_day": False,
    }


def test_rsi_uses_point_in_time_series_and_has_expected_range():
    prices = pd.Series([100, 99, 98, 99, 100, 101, 100, 99, 100, 101, 102, 101, 100, 99, 98])
    result = calculate_rsi(prices, period=5)
    assert result.dropna().between(0, 100).all()
    assert result.index.equals(prices.index)


def test_original_rsi_baseline_buys_on_oversold():
    strategy = RSIStrategy(period=3, oversold=40, overbought=60, shares=10)
    strategy._portfolio_state = _state()
    prices = pd.Series([100, 99, 98, 97, 96], index=pd.date_range("2020-01-01", periods=5, tz="UTC"))
    data = pd.DataFrame({"Close": prices})
    prediction = strategy.predict(data, [])
    assert prediction["signal"] == "buy"
    assert prediction["size"] == 10


def test_news_signal_does_not_use_future_articles():
    features = build_daily_news_signal(
        [
            NewsArticle(
                title="Apple faces investigation and weak outlook",
                available_at=pd.Timestamp("2020-01-10 12:00", tz="UTC"),
            )
        ]
    )
    strategy = NewsAwareRSIStrategy(
        features,
        period=3,
        oversold=40,
        overbought=60,
        shares=10,
        min_news_for_entry=-0.1,
    )
    signal_before_morning_article = strategy._latest_news_signal(
        pd.Timestamp("2020-01-10 09:00", tz="UTC")
    )
    signal_after_article = strategy._latest_news_signal(
        pd.Timestamp("2020-01-10 13:00", tz="UTC")
    )
    signal_before_article_date = strategy._latest_news_signal(
        pd.Timestamp("2020-01-09 23:59", tz="UTC")
    )
    assert signal_before_morning_article == 0.0
    assert signal_after_article < 0.0
    assert signal_before_article_date == 0.0


def test_news_gate_blocks_negative_rsi_entry():
    features = pd.DataFrame(
        {"news_signal": [-1.0], "article_count": [1]},
        index=pd.DatetimeIndex(["2020-01-10 12:00"], tz="UTC"),
    )
    strategy = NewsAwareRSIStrategy(
        features,
        period=3,
        oversold=40,
        overbought=60,
        shares=10,
        min_news_for_entry=-0.1,
    )
    strategy._portfolio_state = _state()
    dates = pd.date_range("2020-01-07", periods=5, tz="UTC")
    data = pd.DataFrame({"Close": [100, 99, 98, 97, 96]}, index=dates)
    # The article was available at noon on 2020-01-10 and the decision is
    # made using the window ending on 2020-01-11; it is therefore usable.
    prediction = strategy.predict(data, [])
    assert prediction["signal"] == "hold"


def test_backtest_uses_current_bar_timestamp_for_news_cutoff():
    dates = pd.date_range("2020-01-01", periods=12, freq="D", tz="UTC")
    prices = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000.0,
        },
        index=dates,
    )
    decision_date = dates[10]
    features = pd.DataFrame(
        {"news_signal": [-1.0, 1.0], "article_count": [1, 1]},
        index=pd.DatetimeIndex(
            [decision_date - pd.Timedelta(hours=12), decision_date], tz="UTC"
        ),
    )
    strategy = NewsAwareRSIStrategy(
        features,
        period=3,
        oversold=40,
        overbought=60,
        shares=1,
        min_news_for_entry=-0.1,
    )
    observed_cutoffs = []
    original = strategy._latest_news_signal

    def capture(as_of):
        observed_cutoffs.append(pd.Timestamp(as_of))
        return original(as_of)

    strategy._latest_news_signal = capture
    config = BacktestConfig(
        initial_capital=10_000,
        lookback_period=5,
        commission_type="percentage",
        commission_value=0.001,
        execution_price="open",
        final_liquidation=True,
        trading_start_date=dates[6].to_pydatetime(),
        trading_end_date=dates[10].to_pydatetime(),
        periods_per_year=252,
        parallel_execution=False,
    )
    Backtest(prices, config).run([strategy])
    assert decision_date in observed_cutoffs
    # The prior-day noon article is available at the decision open; the article
    # timestamped exactly at the current open is not.
    assert original(decision_date) == -1.0
    assert original(decision_date + pd.Timedelta(nanoseconds=1)) == 0.0


def test_gdelt_provider_retries_rate_limit(monkeypatch):
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"articles": []}'

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                BytesIO(),
            )
        return Response()

    monkeypatch.setattr("simple_backtest.news.urlopen", fake_urlopen)
    provider = GDELTNewsProvider(max_retries=1, backoff_seconds=0.001)
    articles = provider.fetch(
        "VOO OR Vanguard",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert articles == []
    assert calls == 2


def test_score_title_is_bounded_and_transparent():
    assert score_title("strong growth and record profits") > 0
    assert score_title("fraud investigation and weak outlook") < 0
    assert -1 <= score_title("bullish upgrade") <= 1
    assert score_title("market commentary") == 0
