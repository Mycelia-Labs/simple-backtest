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

Use ``--target-symbol SPY.US --market-proxy VOO.US --last-five-years`` to
discover the latest provider close, reserve 300 prior trading rows as warm-up,
and skip stock-only fundamentals for the ETF target.
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
    value_valid = "value_score" in scores and scores["value_score"].notna().any()
    quality_valid = "quality_score" in scores and scores["quality_score"].notna().any()
    validity = {
        "value_gate": (value_valid, "No valid dated value scores/cross-section."),
        "quality_gate": (quality_valid, "No valid dated quality scores/cross-section."),
        "value_quality_gate": (value_valid and quality_valid, "Both dated value and quality scores are required."),
    }
    for variant in FUNDAMENTAL_VARIANTS:
        valid, reason = validity[variant]
        if not valid:
            rows.append({"strategy": f"{symbol}:{variant}", "status": "unavailable", "unavailable_reason": reason})
            continue
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
    parser.add_argument("--target-symbol", default=None, help="Run one target, e.g. SPY.US, instead of default AAPL/VOO pair.")
    parser.add_argument("--market-proxy", default=None, help="Separately loaded broad-market proxy for residual mean reversion.")
    parser.add_argument("--last-five-years", action="store_true", help="Discover latest target close and derive a five-year evaluation interval plus 300-row warm-up.")
    parser.add_argument("--skip-fundamentals", action="store_true", help="Skip stock-only dated fundamentals for a price-only rerun.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.last_five_years:
        discovery_symbol = args.target_symbol or "SPY.US"
        discovery = load_ohlcv(discovery_symbol, "2000-01-01", None)
        latest = pd.Timestamp(discovery.index[-1])
        evaluation_start_target = latest - pd.DateOffset(years=5)
        evaluation_position = discovery.index.searchsorted(evaluation_start_target, side="left")
        warmup_position = max(0, evaluation_position - 300)
        args.start = pd.Timestamp(discovery.index[warmup_position]).date().isoformat()
        args.holdout_start = pd.Timestamp(discovery.index[evaluation_position]).date().isoformat()
        args.end = (latest + pd.Timedelta(days=1)).date().isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_symbols = [args.target_symbol] if args.target_symbol else list(PRICE_SYMBOLS)
    if any(not symbol or "." not in symbol for symbol in target_symbols):
        raise ValueError("target symbols must use SYMBOL.EXCHANGE format")
    raw: dict[str, Any] = {
        "settings": {
            "start": args.start,
            "end": args.end,
            "holdout_start": args.holdout_start,
            "target_symbols": target_symbols,
            "latest_available_close": None,
            "date_policy": "latest provider close minus five calendar years; 300 prior trading rows reserved as warm-up",
        }
    }

    data = {symbol: load_ohlcv(symbol, args.start, args.end) for symbol in target_symbols}
    comparison_common_dates = None
    if target_symbols == ["MSFT.US"]:
        voo_for_alignment = load_ohlcv("VOO.US", args.start, args.end)
        comparison_common_dates = data["MSFT.US"].index.intersection(voo_for_alignment.index)
        data["MSFT.US"] = data["MSFT.US"].loc[comparison_common_dates]
        voo_for_alignment = voo_for_alignment.loc[comparison_common_dates]
    market_symbol = args.market_proxy or ("VOO.US" if target_symbols == ["SPY.US"] else "SPY.US")
    market_prices = load_ohlcv(market_symbol, args.start, args.end)["Close"]
    if comparison_common_dates is not None:
        market_prices = market_prices.reindex(comparison_common_dates).dropna()
        data["MSFT.US"] = data["MSFT.US"].loc[data["MSFT.US"].index.intersection(market_prices.index)]
        comparison_common_dates = data["MSFT.US"].index
    raw["market_proxy"] = market_symbol
    raw["settings"]["latest_available_close"] = {symbol: pd.Timestamp(frame.index[-1]).date().isoformat() for symbol, frame in data.items()}
    raw["settings"]["warmup_start"] = pd.Timestamp(data[target_symbols[0]].index[0]).date().isoformat()
    raw["settings"]["evaluation_start"] = args.holdout_start
    raw["settings"]["evaluation_end"] = pd.Timestamp(data[target_symbols[0]].index[-1]).date().isoformat()
    price_rows = []
    for symbol in target_symbols:
        price_rows.extend(run_price_variants(symbol, data[symbol], market_prices, args.holdout_start))
    benchmark_names=[]
    for symbol in target_symbols:
        name=f"{symbol}:buy_hold"; b=FullyInvestedBuyAndHoldStrategy(name=name)
        price_rows.append(run_strategy(name,b,data[symbol],args.holdout_start)); benchmark_names.append(name)
    comparison_benchmark=None
    if target_symbols == ["MSFT.US"]:
        voo=load_ohlcv("VOO.US",args.start,args.end).loc[comparison_common_dates]
        name="VOO.US:buy_hold"; b=FullyInvestedBuyAndHoldStrategy(name=name)
        price_rows.append(run_strategy(name,b,voo,args.holdout_start)); comparison_benchmark=name
        voo.to_csv(args.output_dir/"VOO_US_ohlcv.csv")
    for symbol,frame in data.items(): frame.to_csv(args.output_dir/f"{symbol.replace('.', '_')}_ohlcv.csv")
    price_comparison=add_deltas(price_rows, benchmark_name=benchmark_names[0] if len(benchmark_names)==1 else None)
    if comparison_benchmark:
        bench=price_comparison.loc[price_comparison.strategy==comparison_benchmark].iloc[0]
        price_comparison["return_delta_vs_voo_benchmark_pct_points"]=price_comparison.total_return-bench.total_return
        price_comparison["sharpe_delta_vs_voo_benchmark"]=price_comparison.sharpe_ratio-bench.sharpe_ratio
        price_comparison["drawdown_delta_vs_voo_benchmark_pct_points"]=price_comparison.max_drawdown-bench.max_drawdown
    if comparison_common_dates is not None:
        heldout=comparison_common_dates[comparison_common_dates>=pd.Timestamp(args.holdout_start,tz="UTC")]
        for k,v in {"common_heldout_start":str(heldout[0].date()),"common_heldout_end":str(heldout[-1].date()),"common_heldout_count":int(len(heldout))}.items(): price_comparison[k]=v
    price_comparison.to_csv(args.output_dir / "price_comparison.csv", index=False)

    fundamental_rows: list[dict[str, Any]] = []
    fundamental_target = target_symbols[0] if target_symbols[0] in DEFAULT_STOCK_UNIVERSE else None
    if args.skip_fundamentals:
        fundamental_status = "unavailable"
        fundamental_reason = "Operator-requested price-only rerun; this is not the canonical MSFT suite."
        for variant in FUNDAMENTAL_VARIANTS:
            fundamental_rows.append({"strategy": f"{fundamental_target or target_symbols[0]}:{variant}", "status": "unavailable", "unavailable_reason": fundamental_reason})
    elif fundamental_target is None:
        fundamental_status = "unavailable"
        fundamental_reason = "Skipped: target is not in the stock-only universe; ETF fundamentals are not applied."
        for variant in FUNDAMENTAL_VARIANTS:
            fundamental_rows.append({"strategy": f"{target_symbols[0]}:{variant}", "status": "unavailable", "unavailable_reason": fundamental_reason})
    else:
        decision_dates = pd.DatetimeIndex(
            pd.Series(data[fundamental_target].index).groupby(data[fundamental_target].index.to_period("M")).min().values
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
            fundamental_status = "unavailable"
            fundamental_reason = f"Required point-in-time price inputs unavailable: {missing_price_inputs}"
            for variant in FUNDAMENTAL_VARIANTS:
                fundamental_rows.append({"strategy": f"{fundamental_target}:{variant}", "status": "unavailable", "unavailable_reason": fundamental_reason})
        else:
            try:
                fundamental_result = build_point_in_time_fundamental_signals(
                    DEFAULT_STOCK_UNIVERSE, stock_prices, decision_dates, exchange="US"
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
                fundamental_rows.extend(run_fundamental_variants(
                    fundamental_target, data[fundamental_target], market_prices,
                    fundamental_result.scores_by_symbol.get(fundamental_target), args.holdout_start
                ))
            except Exception as exc:
                fundamental_status = "unavailable"
                fundamental_reason = f"Itoflow dated fundamentals failed: {type(exc).__name__}: {exc}"
                for variant in FUNDAMENTAL_VARIANTS:
                    fundamental_rows.append({"strategy": f"{fundamental_target}:{variant}", "status": "unavailable", "unavailable_reason": fundamental_reason})
    fundamental_comparison = add_deltas(fundamental_rows, benchmark_name=None)
    fundamental_comparison.to_csv(args.output_dir / "fundamental_comparison.csv", index=False)

    raw["fundamentals"] = {
        "status": fundamental_status,
        "universe": list(DEFAULT_STOCK_UNIVERSE),
        "input_semantics": "Itoflow load_fundamentals_history available_date with publication_lag_days=1; derived PE/ROE/margins; no current screener snapshot backfill",
        "unavailable_reason": fundamental_reason,
    }
    raw["benchmark"] = f"{benchmark_names[0]} and VOO.US:buy_hold (when present) use Itoflow provider Close (adjusted series; raw_close retained in OHLCV artifacts) with the same commission and date window."
    if comparison_common_dates is not None:
        heldout_dates = comparison_common_dates[comparison_common_dates >= pd.Timestamp(args.holdout_start, tz="UTC")]
        raw["common_benchmark_calendar"] = {
            "full_coverage_start": pd.Timestamp(comparison_common_dates[0]).date().isoformat(),
            "full_coverage_end": pd.Timestamp(comparison_common_dates[-1]).date().isoformat(),
            "full_coverage_count": int(len(comparison_common_dates)),
            "heldout_start": pd.Timestamp(heldout_dates[0]).date().isoformat(),
            "heldout_end": pd.Timestamp(heldout_dates[-1]).date().isoformat(),
            "heldout_count": int(len(heldout_dates)),
            "price_treatment": "Itoflow provider adjusted Close for benchmark/strategy accounting; raw_close retained in OHLCV artifacts; dividends are reflected only insofar as the provider adjusted series reflects them.",
        }
    raw["fundamental_decision_dates"] = "Not used for ETF-only run; stock-only variants are explicitly unavailable." if fundamental_target is None else "First available trading date of each calendar month; scores use only fundamentals with available_date strictly before each date."
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
