#!/usr/bin/env python3
"""Run a held-out RSI baseline versus RSI + historical-news gate.

Market data is loaded through Itoflow's provider-routed ``get_daily_ohlcv``.
Historical news is loaded separately from GDELT DOC 2.0; GDELT ``seendate`` is
used as the article availability timestamp.  The strategy only consumes news
from dates before the bar on which it trades.

Example:
    python scripts/run_rsi_news_comparison.py \\
        --symbol AAPL.US --start 2020-01-01 --end 2025-01-01 \\
        --holdout-start 2023-01-01 --output-dir research_outputs/aapl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from simple_backtest import Backtest, BacktestConfig
from simple_backtest.news import fetch_historical_news, build_daily_news_signal
from simple_backtest.strategy import NewsAwareRSIStrategy, RSIStrategy


def load_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load and normalize Itoflow OHLCV data for the simple-backtest engine."""
    try:
        from ito_quant.market_data import get_daily_ohlcv
    except ImportError as exc:  # pragma: no cover - depends on runtime installation.
        raise RuntimeError(
            "This research runner requires the Itoflow quant library for provider-routed prices."
        ) from exc
    data = get_daily_ohlcv(
        symbol,
        start_date=start,
        end_date=end,
        fill_method=None,
        allow_partial_coverage=True,
    )
    if data is None or data.empty:
        raise RuntimeError(f"No market data returned for {symbol} in {start} to {end}.")
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(symbol, axis=1, level=0, drop_level=True)
    data = data.rename(columns={column: str(column).title() for column in data.columns})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required) - set(data.columns)
    if missing:
        raise RuntimeError(f"Market data is missing required OHLCV fields: {sorted(missing)}")
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.loc[:, required].sort_index().dropna()
    if len(data) < 100:
        raise RuntimeError("The requested price history is too short for a held-out RSI test.")
    return data


def run_comparison(
    prices: pd.DataFrame,
    news_features: pd.DataFrame,
    holdout_start: str,
    initial_capital: float = 10_000.0,
    commission: float = 0.001,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run both strategies with identical dates, capital, and costs."""
    holdout = pd.Timestamp(holdout_start, tz="UTC")
    if holdout <= prices.index[0] or holdout >= prices.index[-1]:
        raise ValueError("holdout_start must fall inside the supplied price history")
    config = BacktestConfig(
        initial_capital=initial_capital,
        lookback_period=50,
        commission_type="percentage",
        commission_value=commission,
        execution_price="open",
        final_liquidation=True,
        trading_start_date=holdout.to_pydatetime(),
        trading_end_date=prices.index[-1].to_pydatetime(),
        periods_per_year=252,
        parallel_execution=False,
        risk_free_rate=0.0,
    )
    baseline = RSIStrategy(period=14, oversold=30, overbought=70, shares=10, name="rsi_baseline")
    news_strategy = NewsAwareRSIStrategy(
        news_features=news_features,
        period=14,
        oversold=30,
        overbought=70,
        shares=10,
        min_news_for_entry=-0.25,
        news_exit_threshold=-0.60,
        max_news_age_days=3,
        name="rsi_news",
    )
    results = Backtest(prices, config).run([baseline, news_strategy])
    rows = []
    for name in [baseline.get_name(), news_strategy.get_name()]:
        metrics = results.get_strategy(name).metrics
        rows.append(
            {
                "strategy": name,
                "period": f"{holdout.date()} to {prices.index[-1].date()}",
                "total_return_pct": metrics["total_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown_pct": metrics["max_drawdown"],
                "trade_count": int(metrics["total_trades"]),
                "final_value": metrics["final_value"],
            }
        )
    comparison = pd.DataFrame(rows)
    baseline_row = comparison.loc[comparison["strategy"] == "rsi_baseline"].iloc[0]
    news_row = comparison.loc[comparison["strategy"] == "rsi_news"].iloc[0]
    summary = {
        "holdout_start": holdout.isoformat(),
        "holdout_end": prices.index[-1].isoformat(),
        "commission_rate": commission,
        "execution_price": "open",
        "lookback_period": 50,
        "news_availability_field": "GDELT seendate",
        "return_delta_pct_points": float(news_row["total_return_pct"] - baseline_row["total_return_pct"]),
        "sharpe_delta": float(news_row["sharpe_ratio"] - baseline_row["sharpe_ratio"]),
        "drawdown_delta_pct_points": float(news_row["max_drawdown_pct"] - baseline_row["max_drawdown_pct"]),
        "trade_count_delta": int(news_row["trade_count"] - baseline_row["trade_count"]),
    }
    return comparison, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AAPL.US")
    parser.add_argument("--news-query", default="AAPL OR Apple")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--holdout-start", default="2023-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("research_outputs/rsi_news"))
    parser.add_argument("--news-window-days", type=int, default=31)
    parser.add_argument("--news-max-records", type=int, default=250)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prices = load_prices(args.symbol, args.start, args.end)
    prices.to_csv(args.output_dir / "prices.csv")

    news_status: dict[str, Any] = {
        "source": "GDELT DOC 2.0",
        "availability_timestamp": "seendate",
        "query": args.news_query,
        "status": "available",
    }
    try:
        articles = fetch_historical_news(
            args.news_query,
            args.start,
            args.end,
            window_days=args.news_window_days,
            max_records_per_window=args.news_max_records,
        )
        news_features = build_daily_news_signal(articles)
        if news_features.empty:
            raise RuntimeError("GDELT returned no usable articles for this query and period.")
        pd.DataFrame(
            [
                {
                    "title": article.title,
                    "available_at": article.available_at.isoformat(),
                    "url": article.url,
                    "source_country": article.source_country,
                }
                for article in articles
            ]
        ).to_csv(args.output_dir / "news_articles.csv", index=False)
        news_features.to_csv(args.output_dir / "daily_news_signal.csv")
    except (OSError, ValueError, RuntimeError) as exc:  # Keep baseline usable when news is unavailable.
        news_features = pd.DataFrame(columns=["news_signal", "article_count"])
        news_status.update({"status": "unavailable", "reason": str(exc)})

    if news_status["status"] == "available":
        comparison, summary = run_comparison(prices, news_features, args.holdout_start)
        comparison.to_csv(args.output_dir / "comparison.csv", index=False)
        summary["news_status"] = news_status
    else:
        # The baseline remains a valid, reproducible result; do not fabricate a
        # news result by substituting an empty signal for missing historical data.
        baseline = RSIStrategy(period=14, oversold=30, overbought=70, shares=10, name="rsi_baseline")
        holdout = pd.Timestamp(args.holdout_start, tz="UTC")
        config = BacktestConfig(
            initial_capital=10_000,
            lookback_period=50,
            commission_type="percentage",
            commission_value=0.001,
            execution_price="open",
            final_liquidation=True,
            trading_start_date=holdout.to_pydatetime(),
            trading_end_date=prices.index[-1].to_pydatetime(),
            periods_per_year=252,
            parallel_execution=False,
        )
        result = Backtest(prices, config).run([baseline]).get_strategy("rsi_baseline")
        comparison = pd.DataFrame(
            [{
                "strategy": "rsi_baseline",
                "period": f"{holdout.date()} to {prices.index[-1].date()}",
                "total_return_pct": result.metrics["total_return"],
                "sharpe_ratio": result.metrics["sharpe_ratio"],
                "max_drawdown_pct": result.metrics["max_drawdown"],
                "trade_count": int(result.metrics["total_trades"]),
                "final_value": result.metrics["final_value"],
            }]
        )
        comparison.to_csv(args.output_dir / "comparison.csv", index=False)
        summary = {"comparison_status": "news_unavailable", "news_status": news_status}

    # Itoflow's IC helper is a diagnostic, not a tuning step.  It is computed
    # only when news is present and uses prior-day availability versus current-day
    # close-to-close returns to preserve the information boundary.
    if news_status["status"] == "available":
        try:
            from ito_quant.alpha import calculate_ic

            signal = news_features["news_signal"].resample("1D").mean()
            signal = signal.reindex(prices.index.normalize()).fillna(0.0)
            forward_returns = prices["Close"].pct_change()
            ic = calculate_ic(signal.shift(1), forward_returns, method="spearman")
            summary["news_ic_prior_day_vs_return"] = float(ic.ic)
            summary["news_ic_observations"] = int(ic.n_observations)
        except Exception as exc:
            summary["news_ic_status"] = f"unavailable: {exc}"
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
