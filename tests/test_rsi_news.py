"""Tests for the RSI/news and Itoflow signal integration."""

from datetime import datetime, timezone
from io import BytesIO
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from ito_quant.alpha import (
    build_quality_signal,
    build_value_signal,
    calculate_dip_score,
    calculate_volatility_scaled_position_size,
    mean_reversion_signal,
)
from ito_quant.market_data.indicators import calculate_atr, calculate_supertrend

from simple_backtest import Backtest, BacktestConfig
from simple_backtest.fundamental_signals import _build_cross_section_scores, _latest_metric_as_of
from simple_backtest.news import (
    GDELTNewsProvider,
    NewsArticle,
    build_daily_news_signal,
    score_title,
)
from simple_backtest.strategy.itoflow_signals import (
    FullyInvestedBuyAndHoldStrategy,
    ItoflowSignalStrategy,
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
        "timestamp": pd.Timestamp("2020-01-10 09:30", tz="UTC"),
    }


def _ohlc(rows: int = 320) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC")
    close = 100 + np.linspace(0, 8, rows) + 2 * np.sin(np.arange(rows) / 11)
    return pd.DataFrame(
        {
            "Open": close + 0.1,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 100_000,
        },
        index=dates,
    )


def test_fundamental_runner_emits_independent_gate_statuses():
    from scripts.run_itoflow_signal_suite import run_fundamental_variants

    scores = pd.DataFrame(
        {"value_score": [float("nan"), float("nan")], "quality_score": [0.8, 0.7]},
        index=pd.to_datetime(["2024-01-01", "2024-02-01"], utc=True),
    )
    rows = run_fundamental_variants(
        "MSFT.US",
        _ohlc(320),
        _ohlc(320)["Close"],
        scores,
        "2020-11-01",
    )
    by_name = {row["strategy"]: row for row in rows}
    assert by_name["MSFT.US:value_gate"]["status"] == "unavailable"
    assert by_name["MSFT.US:value_quality_gate"]["status"] == "unavailable"
    assert by_name["MSFT.US:quality_gate"]["status"] in {"available", "unavailable"}


def test_fundamental_history_uses_available_date_without_backfill():
    panel = pd.DataFrame(
        {
            "symbol": ["AAPL.US", "AAPL.US"],
            "canonical_metric": ["total_equity", "total_equity"],
            "available_date": pd.to_datetime(["2024-02-01", "2024-04-01"], utc=True),
            "value": [100.0, 120.0],
        }
    )
    assert _latest_metric_as_of(panel, "AAPL.US", "total_equity", pd.Timestamp("2024-03-01", tz="UTC")) == 100.0
    assert _latest_metric_as_of(panel, "AAPL.US", "total_equity", pd.Timestamp("2024-01-15", tz="UTC")) is None


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


def test_value_and_quality_eligibility_are_independent():
    frame = pd.DataFrame(
        {
            "pe_ratio": [float("nan")] * 20,
            "roe": np.linspace(0.05, 0.25, 20),
            "profit_margin": np.linspace(0.05, 0.20, 20),
            "roa": np.linspace(0.02, 0.15, 20),
            "operating_margin": np.linspace(0.04, 0.25, 20),
        },
        index=[f"STOCK{i}.US" for i in range(20)],
    )
    value, quality = _build_cross_section_scores(frame)
    assert value.isna().all()
    assert quality.notna().sum() == 20


    symbols = [f"STOCK{i}.US" for i in range(20)]
    frame = pd.DataFrame(
        {
            "pe_ratio": np.linspace(10, 30, 20),
            "roe": np.linspace(0.05, 0.25, 20),
            "profit_margin": np.linspace(0.05, 0.20, 20),
            "roa": np.linspace(0.02, 0.15, 20),
            "operating_margin": np.linspace(0.04, 0.25, 20),
        },
        index=symbols,
    )
    value = build_value_signal(frame, metric="pe_ratio")
    quality = build_quality_signal(frame, metrics=["roe", "profit_margin", "roa", "operating_margin"])
    assert value.loc[symbols[0]] > value.loc[symbols[-1]]
    assert quality.loc[symbols[-1]] > quality.loc[symbols[0]]


def test_requested_itoflow_signal_helpers_are_called_with_warmup():
    data = _ohlc()
    lower = data.rename(columns=str.lower)
    market = pd.Series(lower["close"].values * 1.01, index=lower.index)
    residual = mean_reversion_signal(pd.DataFrame({"AAPL.US": lower["close"]}), market, lookback=60, method="residual")
    supertrend, trend = calculate_supertrend(lower, period=10, multiplier=3.0)
    dip = calculate_dip_score(lower, rsi_period=14, atr_period=14)
    atr = calculate_atr(lower, period=14)
    size = calculate_volatility_scaled_position_size(0.10, float(atr.dropna().iloc[-1]), float(atr.dropna().tail(60).median()))
    assert len(residual) == len(data)
    assert len(supertrend) == len(data)
    assert len(trend) == len(data)
    assert len(dip) == len(data)
    assert dip["composite_score"].first_valid_index() is not None
    assert 0.02 <= size <= 0.25


def test_new_strategy_warmup_and_volatility_sizing_are_separate_from_direction():
    data = _ohlc()
    market = data["Close"] * 1.01
    strategy = ItoflowSignalStrategy("vol_scaled", "AAPL.US", market, fixed_shares=10)
    assert strategy.required_history >= 252
    strategy._portfolio_state = _state()
    prediction = strategy.predict(data.iloc[: strategy.required_history], [])
    assert prediction["signal"] in {"buy", "hold", "sell"}
    # Direct library sizing is bounded and differs from the fixed 10-share rule.
    sized = calculate_volatility_scaled_position_size(0.10, 2.0, 1.0, min_size=0.02, max_size=0.25)
    assert sized == 0.05


def test_buy_hold_benchmark_is_fully_invested_and_accounted():
    data = _ohlc(40)
    config = BacktestConfig(
        initial_capital=10_000,
        lookback_period=5,
        commission_type="percentage",
        commission_value=0.001,
        execution_price="open",
        final_liquidation=True,
        trading_start_date=data.index[10].to_pydatetime(),
        trading_end_date=data.index[-1].to_pydatetime(),
        periods_per_year=252,
        parallel_execution=False,
    )
    result = Backtest(data, config).run([FullyInvestedBuyAndHoldStrategy()]).get_strategy("voo_buy_hold")
    buys = [trade for trade in result.trade_history if trade["signal"] == "buy"]
    assert len(buys) == 1
    assert buys[0]["shares"] > 90
    assert result.metrics["total_trades"] == 2


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
    assert strategy._latest_news_signal(pd.Timestamp("2020-01-10 09:00", tz="UTC")) == 0.0
    assert strategy._latest_news_signal(pd.Timestamp("2020-01-10 13:00", tz="UTC")) < 0.0
    assert strategy._latest_news_signal(pd.Timestamp("2020-01-09 23:59", tz="UTC")) == 0.0


def test_backtest_uses_current_bar_timestamp_for_news_cutoff():
    dates = pd.date_range("2020-01-01", periods=12, freq="D", tz="UTC")
    prices = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000.0}, index=dates)
    decision_date = dates[10]
    features = pd.DataFrame(
        {"news_signal": [-1.0, 1.0], "article_count": [1, 1]},
        index=pd.DatetimeIndex([decision_date - pd.Timedelta(hours=12), decision_date], tz="UTC"),
    )
    strategy = NewsAwareRSIStrategy(features, period=3, oversold=40, overbought=60, shares=1, min_news_for_entry=-0.1)
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
            raise HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "0"}, BytesIO())
        return Response()

    monkeypatch.setattr("simple_backtest.news.urlopen", fake_urlopen)
    provider = GDELTNewsProvider(max_retries=1, backoff_seconds=0.001)
    articles = provider.fetch("VOO OR Vanguard", datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc))
    assert articles == []
    assert calls == 2


def test_score_title_is_bounded_and_transparent():
    assert score_title("strong growth and record profits") > 0
    assert score_title("fraud investigation and weak outlook") < 0
    assert -1 <= score_title("bullish upgrade") <= 1
    assert score_title("market commentary") == 0
