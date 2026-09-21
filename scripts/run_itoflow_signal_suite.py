#!/usr/bin/env python3
"""Run deterministic Itoflow signal comparisons on AAPL.US and VOO.US.

The runner uses Itoflow public signal/data helpers directly. It does not call a
strategy agent at historical steps. All signal calculations consume only the
lookback window ending before the current bar's open, as supplied by
``simple_backtest``.

The fundamental variants use the documented stock universe in
``simple_backtest.fundamental_signals`` and Itoflow's dated
``load_fundamentals_history`` path. ETF fundamentals are never invented for
VOO; value/quality variants are evaluated only on AAPL as a member of the stock
universe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ito_quant.market_data import get_daily_ohlcv, get_daily_prices

from simple_backtest import Backtest, BacktestConfig
from simple_backtest.fundamental_signals import (
    DEFAULT_STOCK_UNIVERSE,
    build_point_in_time_fundamental_signals,
)
from simple_backtest.strategy import RSIStrategy
from simple_backtest.strategy.itoflow_signals import (
    FullyInvestedBuyAndHoldStrategy,
    ItoflowSignalStrategy,
)

PRICE_SYMBOLS = ("AAPL.US", "VOO.US")
PRICE_VARIANTS = (
    "itoflow_rsi",
    "mean_reversion",
    "supertrend",
    "dip_score",
    "vol_scaled",
    "combined",
)
FUNDAMENTAL_VARIANTS = ("value_gate", "quality_gate", "value_quality_gate")


def load_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load adjusted/provider close OHLCV through Itoflow."""
    frame = get_daily_ohlcv(
        symbol,
        start_date=start,
        end_date=end,
        fill_method=None,
        allow_partial_coverage=True,
        include_raw_close=True,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"No Itoflow OHLCV data for {symbol} in {start} to {end}.")
    frame = frame.rename(columns={column: str(column).title() for column in frame.columns})
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Itoflow OHLCV missing fields for {symbol}: {sorted(missing)}")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index().dropna(subset=list(required))


def make_config(holdout_start: str, end_timestamp: pd.Timestamp) -> BacktestConfig:
    """Use one shared cost/accounting configuration for every strategy."""
    holdout = pd.Timestamp(holdout_start, tz="UTC")
    return BacktestConfig(
        initial_capital=10_000.0,
        lookback_period=300,
        commission_type="percentage",
        commission_value=0.001,
        execution_price="open",
        final_liquidation=True,
        trading_start_date=holdout.to_pydatetime(),
        trading_end_date=end_timestamp.to_pydatetime(),
        periods_per_year=252,
        parallel_execution=False,
        risk_free_rate=0.0,
        error_policy="raise",
    )


def summarize_result(
    result: Any,
    prices: pd.DataFrame,
    initial_capital: float = 10_000.0,
) -> dict[str, Any]:
    """Add average exposure/cash and turnover to simple-backtest metrics."""
    metrics = dict(result.metrics)
    values = result.portfolio_values
    trades = sorted(result.trade_history, key=lambda trade: trade["timestamp"])
    shares = 0.0
    trade_index = 0
    exposures = []
    cash_weights = []
    for timestamp, value in values.items():
        while trade_index < len(trades) and pd.Timestamp(trades[trade_index]["timestamp"]) <= timestamp:
            trade = trades[trade_index]
            if trade["signal"] == "buy":
                shares += float(trade["shares"])
            elif trade["signal"] == "sell":
                shares -= float(trade["shares"])
            trade_index += 1
        close = float(prices.loc[timestamp, "Close"])
        invested = max(0.0, shares * close)
        portfolio_value = float(value)
        exposure = invested / portfolio_value if portfolio_value > 0 else 0.0
        exposures.append(exposure)
        cash_weights.append(max(0.0, 1.0 - exposure))
    gross_turnover = sum(abs(float(t["shares"]) * float(t["price"])) for t in trades)
    metrics.update(
        {
            "average_exposure_pct": 100.0 * float(pd.Series(exposures).mean()),
            "average_cash_pct": 100.0 * float(pd.Series(cash_weights).mean()),
            "gross_turnover_x": gross_turnover / initial_capital,
        }
    )
    return metrics


def run_strategy(
    name: str,
    strategy: Any,
    prices: pd.DataFrame,
    holdout_start: str,
) -> dict[str, Any]:
    config = make_config(holdout_start, prices.index[-1])
    result = Backtest(prices, config).run([strategy]).get_strategy(name)
    metrics = summarize_result(result, prices, config.initial_capital)
    metrics.update({"strategy": name, "status": "available"})
    return metrics


def run_price_variants(
    symbol: str,
    prices: pd.DataFrame,
    market_prices: pd.Series,
    holdout_start: str,
) -> list[dict[str, Any]]:
    rows = []
    baseline = RSIStrategy(
        period=14,
        oversold=30,
        overbought=70,
        shares=10,
        name=f"{symbol}:rsi_baseline",
    )
    rows.append(run_strategy(baseline.get_name(), baseline, prices, holdout_start))
    for variant in PRICE_VARIANTS:
        strategy = ItoflowSignalStrategy(
            variant=variant,
            symbol=symbol,
            market_prices=market_prices,
            fixed_shares=10,
            name=f"{symbol}:{variant}",
        )
        rows.append(run_strategy(strategy.get_name(), strategy, prices, holdout_start))
    return rows


def run_fundamental_variants(
    symbol: str,
    prices: pd.DataFrame,
    market_prices: pd.Series,
    scores: pd.DataFrame | None,
    holdout_start: str,
) -> list[dict[str, Any]]:
    rows = []
    if scores is None or scores.empty:
        reason = "Point-in-time value/quality scores unavailable for this stock universe."
        for variant in FUNDAMENTAL_VARIANTS:
            rows.append({"strategy": f"{symbol}:{variant}", "status": "unavailable", "unavailable_reason": reason})
        return rows
    for variant in FUNDAMENTAL_VARIANTS:
        strategy = ItoflowSignalStrategy(
            variant=variant,
            symbol=symbol,
            market_prices=market_prices,
            fundamental_scores=scores,
            fixed_shares=10,
            name=f"{symbol}:{variant}",
        )
        rows.append(run_strategy(strategy.get_name(), strategy, prices, holdout_start))
    return rows


def add_deltas(rows: list[dict[str, Any]], benchmark_name: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    baseline = frame.loc[frame["strategy"].str.endswith(":rsi_baseline")]
    if not baseline.empty:
        base = baseline.iloc[0]
        frame["return_delta_vs_rsi_pct_points"] = frame.get("total_return", pd.Series(index=frame.index)) - base.get("total_return", float("nan"))
        frame["sharpe_delta_vs_rsi"] = frame.get("sharpe_ratio", pd.Series(index=frame.index)) - base.get("sharpe_ratio", float("nan"))
        frame["drawdown_delta_vs_rsi_pct_points"] = frame.get("max_drawdown", pd.Series(index=frame.index)) - base.get("max_drawdown", float("nan"))
    if benchmark_name and benchmark_name in set(frame["strategy"]):
        bench = frame.loc[frame["strategy"] == benchmark_name].iloc[0]
        frame["return_delta_vs_benchmark_pct_points"] = frame.get("total_return", pd.Series(index=frame.index)) - bench.get("total_return", float("nan"))
        frame["sharpe_delta_vs_benchmark"] = frame.get("sharpe_ratio", pd.Series(index=frame.index)) - bench.get("sharpe_ratio", float("nan"))
        frame["drawdown_delta_vs_benchmark_pct_points"] = frame.get("max_drawdown", pd.Series(index=frame.index)) - bench.get("max_drawdown", float("nan"))
    return frame


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--holdout-start", default="2024-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("research_outputs/itoflow_signal_suite"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw: dict[str, Any] = {"settings": {"start": args.start, "end": args.end, "holdout_start": args.holdout_start}}

    data = {symbol: load_ohlcv(symbol, args.start, args.end) for symbol in PRICE_SYMBOLS}
    market_source = "SPY.US"
    try:
        market_prices = load_ohlcv("SPY.US", args.start, args.end)["Close"]
    except Exception as exc:
        # VOO is itself a broad-market ETF and is already loaded for the
        # explicit benchmark; record the fallback rather than hiding a failed
        # separately requested SPY route.
        market_source = f"VOO.US fallback after SPY.US unavailable: {type(exc).__name__}: {exc}"
        market_prices = data["VOO.US"]["Close"]
    price_rows: list[dict[str, Any]] = []
    for symbol in PRICE_SYMBOLS:
        price_rows.extend(run_price_variants(symbol, data[symbol], market_prices, args.holdout_start))

    # Explicit VOO buy-and-hold benchmark, using the same adjusted provider
    # close treatment and cost configuration as the strategies.
    benchmark = FullyInvestedBuyAndHoldStrategy(name="voo_buy_hold")
    benchmark_row = run_strategy(benchmark.get_name(), benchmark, data["VOO.US"], args.holdout_start)
    price_rows.append(benchmark_row)

    for symbol, frame in data.items():
        frame.to_csv(args.output_dir / f"{symbol.replace('.', '_')}_ohlcv.csv")
    price_comparison = add_deltas(price_rows, benchmark_name="voo_buy_hold")
    price_comparison.to_csv(args.output_dir / "price_comparison.csv", index=False)

    fundamental_rows: list[dict[str, Any]] = []
    decision_dates = pd.DatetimeIndex(
        pd.Series(data["AAPL.US"].index).groupby(data["AAPL.US"].index.to_period("M")).min().values
    ).tz_localize("UTC")
    try:
        stock_price_frame = get_daily_prices(
            list(DEFAULT_STOCK_UNIVERSE),
            start_date=args.start,
            end_date=args.end,
            fill_method=None,
            allow_partial=True,
        )
        stock_prices: dict[str, pd.Series] = {}
        missing_price_inputs: dict[str, str] = {}
        for symbol in DEFAULT_STOCK_UNIVERSE:
            if symbol not in stock_price_frame.columns:
                missing_price_inputs[symbol] = "Itoflow get_daily_prices returned no column"
            else:
                series = pd.to_numeric(stock_price_frame[symbol], errors="coerce").dropna()
                if series.empty:
                    missing_price_inputs[symbol] = "Itoflow get_daily_prices returned no valid observations"
                else:
                    stock_prices[symbol] = series
    except Exception as exc:
        stock_prices = {}
        missing_price_inputs = {
            symbol: f"Itoflow get_daily_prices failed for the stock universe: {type(exc).__name__}: {exc}"
            for symbol in DEFAULT_STOCK_UNIVERSE
        }
    if missing_price_inputs:
        fundamental_result = None
        fundamental_status = "unavailable"
        fundamental_reason = f"Required point-in-time price inputs unavailable: {missing_price_inputs}"
        for variant in FUNDAMENTAL_VARIANTS:
            fundamental_rows.append(
                {
                    "strategy": f"AAPL.US:{variant}",
                    "status": "unavailable",
                    "unavailable_reason": fundamental_reason,
                }
            )
        pd.DataFrame(
            [{"symbol": symbol, "reason": reason} for symbol, reason in missing_price_inputs.items()]
        ).to_csv(args.output_dir / "fundamental_missing_inputs.csv", index=False)
    else:
        try:
            fundamental_result = build_point_in_time_fundamental_signals(
                DEFAULT_STOCK_UNIVERSE,
                stock_prices,
                decision_dates,
                exchange="US",
            )
            fundamental_status = fundamental_result.status
            fundamental_reason = fundamental_result.unavailable_reason
            fundamental_result.panel.to_csv(args.output_dir / "fundamentals_history_panel.csv", index=False)
            fundamental_result.diagnostics.to_csv(args.output_dir / "fundamentals_history_diagnostics.csv", index=False)
            score_frames = []
            for symbol, scores in fundamental_result.scores_by_symbol.items():
                if not scores.empty:
                    score_out = scores.copy()
                    score_out.insert(0, "symbol", symbol)
                    score_frames.append(score_out.reset_index())
            if score_frames:
                pd.concat(score_frames, ignore_index=True).to_csv(args.output_dir / "point_in_time_value_quality_scores.csv", index=False)
            fundamental_rows.extend(
                run_fundamental_variants(
                    "AAPL.US",
                    data["AAPL.US"],
                    market_prices,
                    fundamental_result.scores_by_symbol.get("AAPL.US"),
                    args.holdout_start,
                )
            )
        except Exception as exc:
            fundamental_result = None
            fundamental_status = "unavailable"
            fundamental_reason = f"Itoflow dated fundamentals failed: {type(exc).__name__}: {exc}"
            for variant in FUNDAMENTAL_VARIANTS:
                fundamental_rows.append(
                    {
                        "strategy": f"AAPL.US:{variant}",
                        "status": "unavailable",
                        "unavailable_reason": fundamental_reason,
                    }
                )
    fundamental_comparison = add_deltas(fundamental_rows, benchmark_name=None)
    aapl_rsi = price_comparison.loc[price_comparison["strategy"] == "AAPL.US:rsi_baseline"].iloc[0]
    voo_benchmark = price_comparison.loc[price_comparison["strategy"] == "voo_buy_hold"].iloc[0]
    if not fundamental_comparison.empty:
        fundamental_comparison["return_delta_vs_aapl_rsi_pct_points"] = fundamental_comparison["total_return"] - aapl_rsi["total_return"]
        fundamental_comparison["sharpe_delta_vs_aapl_rsi"] = fundamental_comparison["sharpe_ratio"] - aapl_rsi["sharpe_ratio"]
        fundamental_comparison["drawdown_delta_vs_aapl_rsi_pct_points"] = fundamental_comparison["max_drawdown"] - aapl_rsi["max_drawdown"]
        fundamental_comparison["return_delta_vs_voo_benchmark_pct_points"] = fundamental_comparison["total_return"] - voo_benchmark["total_return"]
        fundamental_comparison["sharpe_delta_vs_voo_benchmark"] = fundamental_comparison["sharpe_ratio"] - voo_benchmark["sharpe_ratio"]
        fundamental_comparison["drawdown_delta_vs_voo_benchmark_pct_points"] = fundamental_comparison["max_drawdown"] - voo_benchmark["max_drawdown"]
    fundamental_comparison.to_csv(args.output_dir / "fundamental_comparison.csv", index=False)

    raw["fundamentals"] = {
        "status": fundamental_status,
        "universe": list(DEFAULT_STOCK_UNIVERSE),
        "input_semantics": "Itoflow load_fundamentals_history available_date with publication_lag_days=1; derived PE/ROE/margins; no current screener snapshot backfill",
        "unavailable_reason": fundamental_reason,
    }
    raw["benchmark"] = "voo_buy_hold uses Itoflow provider Close (adjusted series; raw_close retained in OHLCV artifacts) with the same commission and date window."
    raw["fundamental_decision_dates"] = "First available trading date of each calendar month; scores use only fundamentals with available_date strictly before each date."
    raw["market_proxy"] = market_source
    raw["news"] = "Not used; this suite is independent of the unavailable GDELT feed."
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(raw, handle, indent=2, default=str)
    print("PRICE COMPARISON")
    print(price_comparison.to_string(index=False))
    print("FUNDAMENTAL COMPARISON")
    print(fundamental_comparison.to_string(index=False))
    print(json.dumps(raw, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
