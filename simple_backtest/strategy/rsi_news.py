"""RSI strategies with an optional, point-in-time news gate.

The original notebook's rolling-average RSI is preserved as the default so
baseline results remain reproducible.  Itoflow's RSI helper is a required,
direct dependency for the optional Itoflow backend; the comparison runner uses
that backend explicitly for the modified leg.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from simple_backtest.strategy.base import Strategy

from ito_quant.market_data.indicators import calculate_rsi as _itoflow_calculate_rsi



def _legacy_rsi(prices: pd.Series, period: int) -> pd.Series:
    """Match the simple-backtest notebook's rolling-average RSI exactly."""
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0.0)
    losses = -deltas.where(deltas < 0, 0.0)
    average_gain = gains.tail(period).mean()
    average_loss = losses.tail(period).mean()
    if average_loss == 0:
        return pd.Series(100.0, index=prices.index)
    relative_strength = average_gain / average_loss
    value = 100 - (100 / (1 + relative_strength))
    return pd.Series(float(value), index=prices.index)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Call Itoflow's required RSI helper directly."""
    if period < 2:
        raise ValueError("period must be at least 2")
    return _itoflow_calculate_rsi(prices, period=period)


class RSIStrategy(Strategy):
    """Long-only RSI mean-reversion strategy matching the repository example."""

    required_history = 15

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        shares: float = 10.0,
        name: str | None = None,
        use_itoflow_rsi: bool = False,
    ) -> None:
        super().__init__(name=name or f"RSI_{period}")
        if not 0 < oversold < overbought < 100:
            raise ValueError("thresholds must satisfy 0 < oversold < overbought < 100")
        if shares <= 0:
            raise ValueError("shares must be positive")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.shares = shares
        self.use_itoflow_rsi = use_itoflow_rsi
        self.required_history = period + 1

    def _indicator(self, prices: pd.Series) -> pd.Series:
        """Keep the original notebook backend unless Itoflow is explicitly requested."""
        if self.use_itoflow_rsi:
            return calculate_rsi(prices, period=self.period)
        return _legacy_rsi(prices, period=self.period)

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Buy oversold price action and exit on overbought price action."""
        if len(data) < self.period + 1:
            return self.hold()
        rsi = float(self._indicator(data["Close"]).iloc[-1])
        if pd.isna(rsi):
            return self.hold()
        if rsi < self.oversold and not self.has_position():
            return self.buy(self.shares)
        if rsi > self.overbought and self.has_position():
            return self.sell_all()
        return self.hold()


class NewsAwareRSIStrategy(RSIStrategy):
    """RSI strategy with a strictly point-in-time news risk gate.

    ``news_features`` must be indexed by UTC-normalized availability dates.
    A row dated 2024-01-10 is only eligible for a trade on a later bar, because
    the engine passes the strategy data window ending on the prior bar.  The
    strategy never reads a future news row and treats missing news as neutral.
    """

    def __init__(
        self,
        news_features: pd.DataFrame,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        shares: float = 10.0,
        min_news_for_entry: float = -0.25,
        news_exit_threshold: float = -0.60,
        max_news_age_days: int = 3,
        name: str | None = None,
        use_itoflow_rsi: bool = False,
    ) -> None:
        super().__init__(
            period=period,
            oversold=oversold,
            overbought=overbought,
            shares=shares,
            name=name or f"RSI_News_{period}",
            use_itoflow_rsi=use_itoflow_rsi,
        )
        if not -1 <= news_exit_threshold <= 1:
            raise ValueError("news_exit_threshold must be in [-1, 1]")
        if not -1 <= min_news_for_entry <= 1:
            raise ValueError("min_news_for_entry must be in [-1, 1]")
        if max_news_age_days < 0:
            raise ValueError("max_news_age_days must be non-negative")
        if "news_signal" not in news_features.columns:
            raise ValueError("news_features must contain a 'news_signal' column")
        features = news_features.copy()
        features.index = pd.to_datetime(features.index, utc=True)
        features = features[~features.index.duplicated(keep="last")].sort_index()
        features["news_signal"] = pd.to_numeric(features["news_signal"], errors="coerce")
        self.news_features = features
        self.min_news_for_entry = min_news_for_entry
        self.news_exit_threshold = news_exit_threshold
        self.max_news_age_days = max_news_age_days

    def _latest_news_signal(self, as_of: pd.Timestamp) -> float:
        """Return the latest still-fresh signal available before ``as_of``."""
        if self.news_features.empty:
            return 0.0
        as_of = pd.Timestamp(as_of)
        as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
        eligible = self.news_features.loc[self.news_features.index < as_of]
        if eligible.empty:
            return 0.0
        window_start = as_of - pd.Timedelta(days=self.max_news_age_days)
        recent = eligible.loc[eligible.index >= window_start]
        if recent.empty:
            return 0.0
        value = recent["news_signal"].mean()
        return 0.0 if pd.isna(value) else float(value)

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply RSI logic, blocking negative-news entries and exiting shocks."""
        base_signal = super().predict(data, trade_history)
        news_signal = self._latest_news_signal(pd.Timestamp(data.index[-1]))
        if base_signal["signal"] == "buy" and news_signal < self.min_news_for_entry:
            return self.hold()
        if self.has_position() and news_signal <= self.news_exit_threshold:
            return self.sell_all()
        return base_signal


__all__ = ["RSIStrategy", "NewsAwareRSIStrategy", "calculate_rsi"]
