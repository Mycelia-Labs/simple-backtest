"""Point-in-time stock fundamental signals built from Itoflow history.

The universe is intentionally a documented stock-only list.  This module never
uses the current bulk screener snapshot for past dates. It uses
``load_fundamentals_history`` and each row's ``available_date`` to construct a
cross-sectional table as it would have been known at a decision timestamp, then
passes the derived point-in-time metrics to Itoflow's value and quality signal
builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from ito_quant.alpha.fundamental_signals import build_quality_signal, build_value_signal
from ito_quant.market_data.fundamentals_history import FundamentalsHistoryResult, load_fundamentals_history


DEFAULT_STOCK_UNIVERSE = [
    "AAPL.US",
    "MSFT.US",
    "GOOGL.US",
    "AMZN.US",
    "META.US",
    "NVDA.US",
    "JPM.US",
    "JNJ.US",
    "XOM.US",
    "PG.US",
    "UNH.US",
    "HD.US",
    "CVX.US",
    "COST.US",
    "AVGO.US",
    "BAC.US",
    "WMT.US",
    "PFE.US",
    "KO.US",
    "ORCL.US",
    "CSCO.US",
    "CRM.US",
    "ADBE.US",
    "MRK.US",
]

_QUALITY_METRICS = {
    "total_equity",
    "total_assets",
    "total_revenue",
    "net_income",
    "operatingIncome",
}
_VALUE_METRICS = _QUALITY_METRICS | {"shares_outstanding"}


class FundamentalHistoryUnavailable(RuntimeError):
    """Raised when the dated inputs needed for a fundamental signal are absent."""


@dataclass(frozen=True)
class FundamentalSignalResult:
    scores_by_symbol: dict[str, pd.DataFrame]
    panel: pd.DataFrame
    diagnostics: pd.DataFrame
    universe: tuple[str, ...]
    status: str
    unavailable_reason: str | None = None


def _latest_metric_as_of(panel: pd.DataFrame, symbol: str, metric: str, as_of: pd.Timestamp) -> float | None:
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    rows = panel[
        (panel["symbol"] == symbol)
        & (panel["canonical_metric"] == metric)
        & (panel["available_date"] < as_of)
    ].sort_values("available_date")
    if rows.empty:
        return None
    value = pd.to_numeric(rows.iloc[-1]["value"], errors="coerce")
    return None if pd.isna(value) else float(value)


def _cross_section_at(
    panel: pd.DataFrame,
    prices: dict[str, pd.Series],
    as_of: pd.Timestamp,
    universe: Sequence[str],
) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    rows = []
    for symbol in universe:
        close = prices[symbol].loc[prices[symbol].index < as_of]
        if close.empty:
            continue
        price = float(close.iloc[-1])
        values = {
            metric: _latest_metric_as_of(panel, symbol, metric, as_of)
            for metric in _VALUE_METRICS
        }
        if any(values[metric] is None for metric in _QUALITY_METRICS):
            continue
        equity = values["total_equity"]
        assets = values["total_assets"]
        revenue = values["total_revenue"]
        net_income = values["net_income"]
        operating_income = values["operatingIncome"]
        shares = values["shares_outstanding"]
        if equity <= 0 or assets <= 0 or revenue <= 0:
            continue
        rows.append(
            {
                "symbol": symbol,
                "pe_ratio": price / (net_income / shares) if shares is not None and shares > 0 and net_income > 0 else float("nan"),
                "roe": net_income / equity,
                "profit_margin": net_income / revenue,
                "roa": net_income / assets,
                "operating_margin": operating_income / revenue,
                "as_of": as_of,
            }
        )
    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()


def _build_cross_section_scores(cross_section: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Build value and quality independently; one missing input cannot suppress the other."""
    value = pd.Series(float("nan"), index=cross_section.index, dtype=float)
    quality = pd.Series(float("nan"), index=cross_section.index, dtype=float)
    if cross_section["pe_ratio"].notna().sum() >= 20:
        value_frame = cross_section.loc[cross_section["pe_ratio"].notna()]
        value.loc[value_frame.index] = build_value_signal(value_frame, metric="pe_ratio")
    quality_columns = ["roe", "profit_margin", "roa", "operating_margin"]
    quality_frame = cross_section.loc[cross_section[quality_columns].notna().sum(axis=1).ge(1)]
    if len(quality_frame) >= 20:
        quality.loc[quality_frame.index] = build_quality_signal(quality_frame, metrics=quality_columns)
    return value, quality


def build_point_in_time_fundamental_signals(
    universe: Sequence[str],
    prices: dict[str, pd.Series],
    decision_dates: pd.DatetimeIndex,
    *,
    exchange: str = "US",
) -> FundamentalSignalResult:
    """Build dated value/quality ranks from Itoflow fundamentals history."""
    symbols = tuple(universe)
    history: FundamentalsHistoryResult = load_fundamentals_history(
        list(symbols),
        exchange=exchange,
        on_error="skip",
        publication_lag_days=1,
    )
    panel = history.panel.copy()
    panel["available_date"] = pd.to_datetime(panel["available_date"], utc=True)
    available_metrics = set(panel["canonical_metric"].dropna().unique())
    missing_quality = sorted(_QUALITY_METRICS - available_metrics)
    missing_value = sorted(_VALUE_METRICS - available_metrics)
    if missing_quality:
        return FundamentalSignalResult(
            scores_by_symbol={},
            panel=panel,
            diagnostics=history.symbol_diagnostics,
            universe=symbols,
            status="unavailable",
            unavailable_reason=f"Missing dated quality fundamental metrics: {missing_quality}; value metrics additionally missing: {missing_value}",
        )
    if set(history.successful_symbols) != set(symbols):
        failed = getattr(history, "failed_symbols", {})
        failed_symbols = list(failed) if isinstance(failed, dict) else list(failed or [])
        # Preserve independently usable quality scores when only a value input
        # (for example shares outstanding) is unavailable.
        available_symbols = tuple(symbol for symbol in symbols if symbol in history.successful_symbols)
        if len(available_symbols) < 20:
            return FundamentalSignalResult(
                scores_by_symbol={},
                panel=panel,
                diagnostics=history.symbol_diagnostics,
                universe=symbols,
                status="unavailable",
                unavailable_reason=f"Fundamentals history failed for symbols: {failed_symbols}; valid universe below 20.",
            )
    else:
        available_symbols = symbols

    rows_by_symbol: dict[str, list[dict[str, float | str | pd.Timestamp]]] = {symbol: [] for symbol in symbols}
    for as_of in decision_dates:
        cross_section = _cross_section_at(panel, prices, pd.Timestamp(as_of), available_symbols)
        if cross_section.empty:
            continue
        value, quality = _build_cross_section_scores(cross_section)
        for symbol in cross_section.index:
            rows_by_symbol[symbol].append(
                {
                    "as_of": pd.Timestamp(as_of),
                    "value_score": value.get(symbol, float("nan")),
                    "quality_score": quality.get(symbol, float("nan")),
                    "universe_size": len(cross_section),
                }
            )
    scores = {
        symbol: pd.DataFrame(rows).set_index("as_of").sort_index()
        if rows
        else pd.DataFrame(columns=["value_score", "quality_score", "universe_size"], index=pd.DatetimeIndex([], tz="UTC"))
        for symbol, rows in rows_by_symbol.items()
    }
    if not any(not frame.empty for frame in scores.values()):
        return FundamentalSignalResult(
            scores_by_symbol=scores,
            panel=panel,
            diagnostics=history.symbol_diagnostics,
            universe=symbols,
            status="unavailable",
            unavailable_reason="No cross-sectional dated rows had all required inputs.",
        )
    return FundamentalSignalResult(
        scores_by_symbol=scores,
        panel=panel,
        diagnostics=history.symbol_diagnostics,
        universe=symbols,
        status="available",
    )


__all__ = [
    "DEFAULT_STOCK_UNIVERSE",
    "FundamentalHistoryUnavailable",
    "FundamentalSignalResult",
    "build_point_in_time_fundamental_signals",
]
