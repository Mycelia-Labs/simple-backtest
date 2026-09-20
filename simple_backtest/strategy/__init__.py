"""Strategy module containing base class and implementations."""

from simple_backtest.strategy.base import Strategy
from simple_backtest.strategy.buy_and_hold import BuyAndHoldStrategy
from simple_backtest.strategy.dca import DCAStrategy
from simple_backtest.strategy.moving_average import MovingAverageStrategy
from simple_backtest.strategy.rsi_news import NewsAwareRSIStrategy, RSIStrategy

__all__ = [
    "Strategy",
    "BuyAndHoldStrategy",
    "DCAStrategy",
    "MovingAverageStrategy",
    "RSIStrategy",
    "NewsAwareRSIStrategy",
]
