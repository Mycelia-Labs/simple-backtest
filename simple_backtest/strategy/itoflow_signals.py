"""Deterministic strategies built from Itoflow signal helpers.

This module deliberately keeps the repository's original ``RSIStrategy`` out
of the new variants.  Each variant calls the requested Itoflow helper directly
on the pre-trade lookback window supplied by ``simple_backtest``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

import pandas as pd

from ito_quant.alpha import (
    calculate_dip_score,
    calculate_volatility_scaled_position_size,
    mean_reversion_signal,
)
from ito_quant.market_data.indicators import calculate_atr, calculate_rsi, calculate_supertrend

from simple_backtest.strategy.base import Strategy

Variant = Literal[
    "itoflow_rsi",
    "mean_reversion",
    "supertrend",
    "dip_score",
    "vol_scaled",
    "combined",
    "value_gate",
    "quality_gate",
    "value_quality_gate",
]


def _lower_ohlc(data: pd.DataFrame) -> pd.DataFrame:
    """Return the exact lowercase OHLCV schema expected by Itoflow indicators."""
    required = {"open", "high", "low", "close"}
    frame = data.rename(columns={column: str(column).lower() for column in data.columns}).copy()
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OHLC data missing required columns: {sorted(missing)}")
    return frame


def _finite_last(series: pd.Series) -> float | None:
    value = series.iloc[-1] if not series.empty else None
    if value is None or pd.isna(value):
        return None
    return float(value)


class ItoflowSignalStrategy(Strategy):
    """Long-only deterministic strategy for one price series.

    ``variant`` selects exactly one directional signal.  ``vol_scaled`` uses
    Itoflow RSI for direction and Itoflow ATR sizing, isolating sizing from the
    direction rule.  Fundamental variants consume a point-in-time signal series
    prepared by ``fundamental_signals.py``; no current snapshot is backfilled.
    """

    def __init__(
        self,
        variant: Variant,
        symbol: str,
        market_prices: pd.Series,
        fundamental_scores: pd.DataFrame | None = None,
        fixed_shares: float = 10.0,
        rsi_period: int = 14,
        lookback: int = 60,
        oversold: float = 30.0,
        overbought: float = 70.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or f"{variant}_{symbol}")
        if variant not in {
            "itoflow_rsi",
            "mean_reversion",
            "supertrend",
            "dip_score",
            "vol_scaled",
            "combined",
            "value_gate",
            "quality_gate",
            "value_quality_gate",
        }:
            raise ValueError(f"Unsupported variant: {variant}")
        if fixed_shares <= 0:
            raise ValueError("fixed_shares must be positive")
        self.variant = variant
        self.symbol = symbol
        self.market_prices = pd.to_numeric(market_prices, errors="coerce").dropna().sort_index()
        self.fundamental_scores = fundamental_scores
        self.fixed_shares = fixed_shares
        self.rsi_period = rsi_period
        self.lookback = lookback
        self.oversold = oversold
        self.overbought = overbought
        # Dip score needs a 52-week high; retain a full year plus indicator warm-up.
        self.required_history = max(lookback + 1, 252 + rsi_period + 1)
        if variant in {"value_gate", "quality_gate", "value_quality_gate"} and fundamental_scores is None:
            raise ValueError(f"{variant} requires point-in-time fundamental_scores")

    def _rsi(self, data: pd.DataFrame) -> float | None:
        values = calculate_rsi(_lower_ohlc(data)["close"], period=self.rsi_period)
        return _finite_last(values)

    def _mean_reversion(self, data: pd.DataFrame) -> float | None:
        close = _lower_ohlc(data)["close"]
        market = self.market_prices.reindex(close.index).ffill()
        if market.isna().any() or len(close) < self.lookback + 1:
            return None
        result = mean_reversion_signal(
            pd.DataFrame({self.symbol: close}),
            market,
            lookback=self.lookback,
            method="residual",
        )[self.symbol]
        return _finite_last(result)

    def _supertrend(self, data: pd.DataFrame) -> float | None:
        _, trend = calculate_supertrend(_lower_ohlc(data), period=10, multiplier=3.0)
        return _finite_last(trend)

    def _dip_score(self, data: pd.DataFrame) -> float | None:
        score = calculate_dip_score(_lower_ohlc(data), rsi_period=self.rsi_period, atr_period=14)
        return _finite_last(score["composite_score"])

    def _fundamental_score(self, as_of: pd.Timestamp) -> float | None:
        if self.fundamental_scores is None:
            return None
        scores = self.fundamental_scores
        eligible = scores.loc[scores.index < as_of]
        if eligible.empty:
            return None
        row = eligible.iloc[-1]
        if self.variant == "value_gate":
            value = row.get("value_score")
        elif self.variant == "quality_gate":
            value = row.get("quality_score")
        else:
            values = [row.get("value_score"), row.get("quality_score")]
            value = sum(v for v in values if pd.notna(v)) / sum(pd.notna(v) for v in values)
        return None if pd.isna(value) else float(value)

    def _direction(self, data: pd.DataFrame, as_of: pd.Timestamp) -> str:
        rsi = self._rsi(data)
        if rsi is None:
            return "hold"
        if self.variant == "itoflow_rsi" or self.variant == "vol_scaled":
            return "buy" if rsi < self.oversold else "sell" if rsi > self.overbought else "hold"
        if self.variant == "mean_reversion":
            signal = self._mean_reversion(data)
            return "buy" if signal is not None and signal > 1.0 else "sell" if signal is not None and signal < -1.0 else "hold"
        if self.variant == "supertrend":
            trend = self._supertrend(data)
            return "buy" if trend == 1 else "sell" if trend == -1 else "hold"
        if self.variant == "dip_score":
            score = self._dip_score(data)
            return "buy" if score is not None and score >= 60 else "sell" if score is not None and score < 35 else "hold"
        if self.variant == "combined":
            votes = [
                rsi < self.oversold,
                (self._mean_reversion(data) or 0.0) > 1.0,
                self._supertrend(data) == 1,
                (self._dip_score(data) or 0.0) >= 60,
            ]
            buy_votes = sum(votes)
            sell_votes = sum(
                [
                    rsi > self.overbought,
                    (self._mean_reversion(data) or 0.0) < -1.0,
                    self._supertrend(data) == -1,
                    (self._dip_score(data) or 100.0) < 35,
                ]
            )
            return "buy" if buy_votes >= 2 else "sell" if sell_votes >= 2 else "hold"
        fundamental = self._fundamental_score(as_of)
        return "buy" if rsi < self.oversold and fundamental is not None and fundamental >= 0.5 else "sell" if rsi > self.overbought else "hold"

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        as_of = getattr(self, "_portfolio_state", {}).get("timestamp") if self._portfolio_state else None
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(data.index[-1])
        direction = self._direction(data, as_of)
        if direction == "buy" and not self.has_position():
            if self.variant == "vol_scaled":
                ohlc = _lower_ohlc(data)
                atr = calculate_atr(ohlc, period=14)
                valid_atr = atr.dropna()
                current_atr = _finite_last(atr)
                normal_atr = float(valid_atr.tail(60).median()) if not valid_atr.empty else None
                if current_atr is None or normal_atr is None or normal_atr <= 0:
                    return self.hold()
                target_pct = calculate_volatility_scaled_position_size(
                    base_size=0.10,
                    current_atr=current_atr,
                    normal_atr=normal_atr,
                    min_size=0.02,
                    max_size=0.25,
                )
                return self.buy_percent(target_pct)
            return self.buy(self.fixed_shares)
        if direction == "sell" and self.has_position():
            return self.sell_all()
        return self.hold()


class FullyInvestedBuyAndHoldStrategy(Strategy):
    """Deterministic 100%-invested benchmark using the same price series/costs.

    ``buy_percent(0.999)`` leaves a tiny commission cushion because the engine
    charges commission on top of the requested notional; resulting exposure is
    effectively fully invested rather than rejected as unaffordable.
    """

    def __init__(self, name: str = "voo_buy_hold") -> None:
        super().__init__(name=name)
        self.bought = False

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.bought and not self.has_position():
            return self.buy_percent(0.999)
        return self.hold()

    def on_trade_executed(self, trade_info: Dict[str, Any]) -> None:
        if trade_info.get("signal") == "buy":
            self.bought = True

    def reset_state(self) -> None:
        super().reset_state()
        self.bought = False


__all__ = ["ItoflowSignalStrategy", "FullyInvestedBuyAndHoldStrategy"]
