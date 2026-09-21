<div align="center">

# Simple Backtest

**A small, transparent backtesting framework for long-only strategies**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/LGuillermoAngaritaG/simple-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/LGuillermoAngaritaG/simple-backtest/actions/workflows/ci.yml)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## 📖 About

Simple Backtest is a Python framework designed to make backtesting trading strategies straightforward and accessible. Whether you're testing a simple moving average crossover or a complex machine learning model, Simple Backtest provides the tools you need.

**Key Philosophy**: Bring your own data from any library, API, or file. The
package deliberately provides no market-data source. Inherit from `Strategy`,
`Commission`, or `Optimizer` to supply custom behavior; the built-in classes
and examples are starting points.

Simple Backtest is strictly a simulation library. It does not connect to brokers,
route orders, manage live risk, or operate trading accounts. Read the
[simulation scope and limitations](docs/SIMULATION_SCOPE.md) before using its
results to inform financial decisions.

## ✨ Features

<table>
<tr>
<td width="50%">

### 🚀 Performance
- **Parallel Execution**: Test multiple strategies simultaneously
- **Optimized Core**: Fast backtesting engine with efficient portfolio tracking
- **Efficient Accounting**: Constant-time position totals and optional strategy parallelism

</td>
<td width="50%">

### 📊 Analytics
- **20+ Metrics**: Sharpe, Sortino, Calmar, Win Rate, etc.
- **Benchmark Comparison**: Alpha, Beta, Information Ratio
- **Interactive Visualizations**: Plotly-powered charts

</td>
</tr>
<tr>
<td width="50%">

### 🎯 Design
- **Clean Architecture**: Strategy Pattern for extensibility
- **Type Safety**: Pydantic validation for configurations
- **Explicit Scope**: One long-only, cash-funded instrument per backtest

</td>
<td width="50%">

### 🔧 Flexibility
- **Custom Strategies**: Easy inheritance model
- **Commission Models**: Percentage, flat, tiered, custom
- **Parameter Optimization**: Grid search, random search, walk-forward

</td>
</tr>
</table>

### Supported Assets

Works with one OHLC(V) price series at a time. The accounting model supports
long-only, cash-funded spot instruments; it does not model short selling,
margin, leverage, borrowing, contract multipliers, funding, or FX conversion.

| Asset Type | Support | Notes |
|------------|---------|-------|
| 📈 **Stocks** | ✅ Supported | Long-only; fractional or whole shares |
| ₿ **Spot crypto** | ✅ Supported | Long-only fractional units |
| 📊 **ETFs** | ✅ Supported | Same accounting as stocks |
| 💱 **Forex** | ⚠️ Price signals only | No lots, leverage, rollover, or currency conversion |
| 🛢️ **Commodities** | ⚠️ Price signals only | No physical/contract mechanics |
| 📉 **Futures** | ❌ Accounting unsupported | No margin, multipliers, expiry, or roll logic |
| 📊 **Options** | ❌ No | Requires Greeks, strikes, expiration |

## 📓 Examples

### Interactive Notebooks

Explore comprehensive examples in Jupyter notebooks. Click "Open in Colab" to run them directly in your browser:

| Notebook | Description | Colab Link |
|----------|-------------|------------|
| **01_basic_usage.ipynb** | Introduction, data loading, commission setup, strategy comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/01_basic_usage.ipynb) |
| **02_candle_strategies.ipynb** | Candlestick patterns (Engulfing, Hammer, Doji, etc.) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/02_candle_strategies.ipynb) |
| **03_ta_strategies.ipynb** | Technical indicators (RSI, MACD, Bollinger Bands, etc.) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/03_ta_strategies.ipynb) |
| **04_ml_strategies.ipynb** | Machine learning strategies (Logistic Regression, Random Forest, Gradient Boosting) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/04_ml_strategies.ipynb) |
| **05_commission_usage.ipynb** | Commission models comparison and custom implementations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/05_commission_usage.ipynb) |
| **06_advanced_optimization.ipynb** | Grid search, random search, walk-forward optimization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/06_advanced_optimization.ipynb) |


## 📦 Installation

```bash
# Using pip
pip install simple-backtest

# Using uv (recommended)
uv add simple-backtest

# From source
git clone https://github.com/LGuillermoAngaritaG/simple-backtest.git
cd simple-backtest
uv sync --extra dev --extra notebooks
```

**Requirements**: Python 3.10+

## 🚀 Quick Start

Get up and running in 3 simple steps:

```python
# 1. Get data (using yfinance for demo, but you can use any other data source)
import yfinance as yf
data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")

# 2. Create strategy (you can use a basic one or create your own)
from simple_backtest import Backtest, BacktestConfig, MovingAverageStrategy

strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=10)

# 3. Run backtest
config = BacktestConfig.default(initial_capital=10000)
backtest = Backtest(data, config)
results = backtest.run([strategy])

# View results
print(results.get_strategy(strategy.get_name()).summary())
```

**Output:**
```
Total Return: 227.91%
CAGR: 36.84%
Sharpe Ratio: 1.09
Max Drawdown: 30.60%
Win Rate: 100.00%
```

## 📰 Historical News + RSI Research Example

The repository now includes a reusable `RSIStrategy` and
`NewsAwareRSIStrategy`. The latter keeps the original RSI entry/exit rules but
adds a conservative, point-in-time news gate: negative news can block an
oversold entry or force an exit. Itoflow's RSI helper is imported directly and
is required by this research module; the runner loads OHLCV through Itoflow's
provider-routed `get_daily_ohlcv` API. The comparison runs the original local
RSI baseline and an explicit `rsi_itoflow` leg under identical conditions, then
adds `rsi_itoflow_news` only when historical news is available.

Historical news is intentionally sourced separately from market data. Itoflow's
public quant modules used here provide market data, indicators, and diagnostics;
we found no Itoflow news-history API in the available library surface. The
example uses GDELT DOC 2.0 article-list results and treats GDELT's `seendate` as
an availability timestamp. It does not use a later article retrieval time or
assume that the market-data provider contains historical news. The signal is
aggregated into an event-level signal while preserving each article's exact
availability timestamp. The strategy uses a strict prior-timestamp cutoff; for
daily bars, the engine supplies the prior completed bar, so same-day intraday
articles cannot affect that morning's trade.

The GDELT client now retries HTTP 429 responses sequentially, honors a numeric
`Retry-After` header when supplied, and otherwise uses capped exponential
backoff. It writes fetched article and signal files as output artifacts for
inspection and later user-managed caching. A persistent 429 or non-JSON
response still fails closed. `yfinance` can provide a current/recent `Ticker.news`
feed, but it is not treated here as a complete historical point-in-time archive
for a 2022–2024 backtest; use it only with an independently captured, timestamped
news cache.

Run a held-out comparison (requires the Itoflow quant package and network
access to GDELT):

```bash
python scripts/run_rsi_news_comparison.py \\
  --symbol AAPL.US \\
  --start 2020-01-01 --end 2025-01-01 \\
  --holdout-start 2023-01-01 \\
  --output-dir research_outputs/aapl
```

The runner uses identical dates, initial capital, open execution, 0.1%
commission, RSI parameters, and final liquidation for all available strategies.
It writes `comparison.csv` with total return, annualized Sharpe ratio, maximum
drawdown, and trade count, plus `summary.json`, the news article cache, and the
daily news signal. If historical news is unavailable, it still runs and
reports both the original RSI baseline and `rsi_itoflow`; it marks only the
news-aware leg as unavailable rather than substituting an empty or fabricated
news signal. The default news thresholds are fixed before the held-out run;
they are not tuned on the holdout period.

The package tests use deterministic synthetic inputs and do not call external
services:

```bash
pytest tests/test_rsi_news.py -q
```

This remains a simulation/research example. It does not place orders or connect
to a broker.


## Signal-suite result from the held-out run

The price-based run completed on AAPL.US and VOO.US for 2024-01-01 through
2024-12-31 after 2021-01-01 warm-up. VOO buy-and-hold returned 26.4049% with
1.9322 Sharpe, 9.1160% maximum drawdown, 2 trades, 99.61% average exposure,
and 0.6156% average cash. The fixed-10-share AAPL RSI baseline returned 3.0467%
with 0.8125 Sharpe, 2.6156% drawdown, 8 trades, 10.99% exposure, and 89.01%
cash. The fixed-10-share VOO RSI baseline returned 5.5208% with 2.4620 Sharpe,
0.7793% drawdown, 4 trades, 5.48% exposure, and 94.52% cash. These exposure
figures are why the fully invested VOO benchmark is not a like-for-like capital
allocation comparison.

The new price variants are preserved in `price_comparison.csv`; the combined
variant is not described as an improvement merely because it trades. The dated
fundamental integration and both Itoflow fundamental builders are implemented
and tested, but the held-out evaluation is explicitly unavailable: after
`available_date` filtering, one required universe member failed the Itoflow
fundamentals route and the valid cross-section fell below the library's minimum
20 observations. No current screener snapshot was substituted.


The branch also includes `scripts/run_itoflow_signal_suite.py`, a deterministic
Python backtest that calls Itoflow's public functions directly—no strategy
agent is invoked at historical steps. It keeps the original `rsi_baseline`
unchanged and compares independently selectable variants:

- `itoflow_rsi` — Itoflow `calculate_rsi`; kept separate from new signals.
- `mean_reversion` — Itoflow `mean_reversion_signal(method="residual")` with a separately loaded `SPY.US` market proxy; it records a VOO fallback only if SPY is unavailable.
- `supertrend` — Itoflow `calculate_supertrend()`.
- `dip_score` — Itoflow `calculate_dip_score()`.
- `vol_scaled` — Itoflow RSI direction plus separately tested `calculate_volatility_scaled_position_size()` sizing.
- `combined` — a documented 2-of-4 vote across RSI, residual mean reversion, SuperTrend, and dip score.
- `voo_buy_hold` — explicit fully invested VOO benchmark, not RSI-traded VOO.

Run the full AAPL/VOO held-out suite:

```bash
PYTHONPATH=. python scripts/run_itoflow_signal_suite.py \\
  --start 2021-01-01 --end 2025-01-01 \\
  --holdout-start 2024-01-01 \\
  --output-dir research_outputs/itoflow_signal_suite
```

The suite uses a 300-row backtest lookback, giving the 252-session dip-score
warm-up and longer indicators enough history. Trades use the same $10,000
capital, open execution, 0.1% commission, final liquidation, and 2024 holdout.
It writes `price_comparison.csv` with return, Sharpe, maximum drawdown, trade
count, gross turnover, average exposure, average cash, and deltas versus the
AAPL RSI baseline and VOO buy-and-hold benchmark. It also writes
`fundamental_comparison.csv`, dated fundamentals diagnostics, and
`point_in_time_value_quality_scores.csv`.

The fundamental variants are evaluated only for AAPL using this documented
24-stock US universe: AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, JNJ, XOM, PG,
UNH, HD, CVX, COST, AVGO, BAC, WMT, PFE, KO, ORCL, CSCO, CRM, ADBE, and
MRK (all `.US`). They call Itoflow's `load_fundamentals_history()` and use
`available_date` strictly before monthly decision dates; derived PE, ROE,
profit margin, ROA, and operating margin are then passed to Itoflow's
`build_value_signal()` and `build_quality_signal()`. Today's screener snapshot
is never backfilled into history, and VOO ETF fundamentals are not invented.
If dated history or required price inputs fail, the fundamental rows are marked
unavailable with the exact reason while the price-based suite still completes.

The suite is independent of GDELT and the unavailable news feed. Itoflow must
be installed and its provider route configured; no broker credentials are used.

The focused tests cover direct Itoflow signal calls, warm-up, current-bar
look-ahead timing, volatility sizing, value/quality cross-sectional ranking,
and fully invested benchmark accounting:

```bash
PYTHONPATH=. pytest tests/test_rsi_news.py -q
PYTHONPATH=. pytest -q
```


Implement your own strategy by inheriting from `Strategy` and defining the `predict()` method:

```python
from simple_backtest import Strategy

class MyStrategy(Strategy):
    """Custom trading strategy."""

    # The engine and optimizers reject configurations with shorter lookbacks.
    required_history = 20

    def __init__(self, threshold=100, name=None):
        super().__init__(name=name or "MyStrategy")
        self.threshold = threshold

    def predict(self, data, trade_history):
        """Generate trading signal.

        Args:
            data: OHLCV DataFrame with lookback window
            trade_history: List of past trades

        Returns:
            Dict with keys: signal ("buy"/"hold"/"sell"), size, order_ids
        """
        current_price = data['Close'].iloc[-1]

        # Simple logic: buy below threshold, sell above
        if current_price < self.threshold and not self.has_position():
            return self.buy(10)  # Buy 10 shares
        elif current_price > self.threshold * 1.2 and self.has_position():
            return self.sell_all()  # Sell all positions
        else:
            return self.hold()  # Do nothing
```

**Strategy Helper Methods:**
- `self.has_position()` - Check if holding any shares
- `self.get_position()` - Get current share count
- `self.get_cash()` - Get available cash
- `self.get_portfolio_value()` - Get total portfolio value
- `self.buy(shares)` - Return buy signal
- `self.sell(shares)` - Return sell signal
- `self.sell_all()` - Sell all positions
- `self.buy_percent(percent)` - Buy shares worth % of portfolio
- `self.buy_cash(amount)` - Buy shares worth specific amount

### Configuration Presets

Quick configurations for common scenarios:

```python
from simple_backtest import BacktestConfig

# Zero commission (for testing)
config = BacktestConfig.zero_commission(initial_capital=10000)

# Dense bar-data preset (not a latency/order-book HFT simulator)
config = BacktestConfig.high_frequency(initial_capital=100000)

# Swing trading (longer lookback, typical retail commission)
config = BacktestConfig.swing_trading(initial_capital=10000)

# Low percentage commission preset (0.01%)
config = BacktestConfig.low_commission(initial_capital=10000)
```

### Comparing Multiple Strategies

```python
from simple_backtest import (
    Backtest,
    BacktestConfig,
    MovingAverageStrategy,
    BuyAndHoldStrategy,
    DCAStrategy
)

# Create strategies
strategies = [
    MovingAverageStrategy(short_window=10, long_window=30, shares=10),
    BuyAndHoldStrategy(shares=50),
    DCAStrategy(investment_amount=500, interval_days=30)
]

# Run backtest
config = BacktestConfig.default(initial_capital=10000)
backtest = Backtest(data, config)
results = backtest.run(strategies)

# Compare strategies
comparison = results.compare()
print(comparison)

# Get best strategy
best = results.best_strategy('sharpe_ratio')
print(f"Best: {best.name} (Sharpe: {best.metrics['sharpe_ratio']:.2f})")

# Visualize
results.plot_comparison().show()
```

### Parameter Optimization

Find optimal strategy parameters using built-in optimizers:

```python
from simple_backtest import GridSearchOptimizer, BacktestConfig

# Define parameter space
param_space = {
    'short_window': [5, 10, 15, 20],
    'long_window': [30, 40, 50, 60],
    'shares': [10]
}

# Run optimization
optimizer = GridSearchOptimizer(verbose=True)
results = optimizer.optimize(
    data=data,
    config=BacktestConfig.default(lookback_period=60),
    strategy_class=MovingAverageStrategy,
    param_space=param_space,
    metric='sharpe_ratio'
)

# View top results
print(results.head(5))
```

**Available Optimizers:**
- `GridSearchOptimizer` - Exhaustive search (best for small spaces)
- `RandomSearchOptimizer` - Random sampling (faster for large spaces)
- `WalkForwardOptimizer` - Expanding training windows with chronological out-of-sample folds

Set each strategy's `required_history` to the minimum number of rows its
indicators need. Optimizers record parameter combinations that exceed
`lookback_period` as failed candidates instead of running invalid simulations.
Use an explicit `random_state` for reproducible random searches.

### Custom Commission Models

Create custom commission structures:

```python
from simple_backtest import Commission

class TieredWithMinimum(Commission):
    """Tiered commission with minimum fee."""

    def __init__(self):
        super().__init__(name="TieredMin")

    def calculate(self, shares, price):
        trade_value = shares * price

        if trade_value < 1000:
            commission = max(trade_value * 0.002, 1.0)  # 0.2%, min $1
        elif trade_value < 10000:
            commission = trade_value * 0.001  # 0.1%
        else:
            commission = trade_value * 0.0005  # 0.05%

        return commission

# Pass custom behavior explicitly to Backtest
from simple_backtest import Backtest, BacktestConfig

config = BacktestConfig.default(
    commission_type="custom",
    commission_value=0.0,
)
backtest = Backtest(data, config, commission_calculator=TieredWithMinimum())
results = backtest.run([strategy])
```

Custom commission callbacks must be deterministic and side-effect free because
the engine evaluates them for benchmark affordability as well as strategy fills.

For a custom execution price, set `execution_price="custom"` and pass
`execution_price_extractor=` to `Backtest`. Strategy exceptions raise with
strategy/date/stage context by default; use `error_policy="continue"` only when
you intentionally want structured diagnostics in `StrategyResult.errors`.

### Execution and Timing Assumptions

Signals receive only rows strictly before the execution bar, so a signal formed
from the supplied window cannot see its own fill price. Orders fill at the
configured bar price (`open`, `close`, `typical`, or `custom`). The legacy
`vwap` option remains as a deprecated alias for the OHLC typical price
`(high + low + close) / 3`; one OHLCV bar is not enough to calculate true VWAP.

Execution realism is deterministic and opt-in:

```python
config = BacktestConfig.default(
    slippage_bps=5,                 # adverse to both buys and sells
    spread_bps=10,                  # half-spread applied in each direction
    max_volume_participation=0.05,  # at most 5% of the execution bar's volume
    final_liquidation=True,         # attempt to close on the final bar
)
```

When volume participation caps an order, only the capped quantity is filled and
the unfilled remainder is cancelled; the engine does not maintain resting
orders. `final_liquidation=False` is the default, so open positions remain
marked to the final close. Enabling it applies the same cost and volume rules to
both strategies and the benchmark.

Annualized metrics infer observations per year from the data's timestamp span.
For short, irregular, or mixed-frequency data, set `periods_per_year`
explicitly. DCA intervals are measured from successful fills, not rejected or
zero-sized attempts.

### Logging Control

Control framework verbosity:

```python
from simple_backtest.utils import setup_logging, disable_logging, enable_debug_logging
import logging

# Default: WARNING level (minimal output)

# For verbose output during optimization
setup_logging(level=logging.INFO)

# For debugging issues
enable_debug_logging()

# To suppress all output
disable_logging()
```

## 📊 Performance Metrics

The framework calculates 20+ metrics automatically:

### Returns
- Total Return (%)
- CAGR (Compound Annual Growth Rate)

### Risk Metrics
- Volatility (annualized standard deviation)
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted return)
- Calmar Ratio (return vs max drawdown)
- Max Drawdown (%)
- Max Drawdown Duration

### Trade Statistics
- Total Trades
- Win Rate (%)
- Profit Factor
- Trade Expectancy
- Average Win / Average Loss

### Benchmark Comparison
- Alpha (excess return vs benchmark)
- Beta (correlation with benchmark)
- Information Ratio


## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/LGuillermoAngaritaG/simple-backtest.git
cd simple-backtest

# Install with uv (recommended)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=simple_backtest

# Run specific test file
uv run pytest tests/test_strategy.py

# Run specific test
uv run pytest tests/test_strategy.py::test_strategy_initialization
```

### Code Quality

```bash
# Lint code
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Run pre-commit hooks
pre-commit run --all-files
```

### Pre-commit Hooks

Pre-commit hooks automatically run linting, formatting, and tests on commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🤝 Contributing

Contributions are welcome! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `uv run pytest`
5. **Run linting**: `uv run ruff check . && uv run ruff format --check .`
6. **Commit your changes**: `git commit -m "Add amazing feature"`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Pydantic](https://docs.pydantic.dev/) for configuration validation
- Uses [Plotly](https://plotly.com/) for interactive visualizations
- Parallel processing with [Joblib](https://joblib.readthedocs.io/)
- Testing with [Pytest](https://docs.pytest.org/)
- Code quality with [Ruff](https://github.com/astral-sh/ruff)

## 📬 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/LGuillermoAngaritaG/simple-backtest/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LGuillermoAngaritaG/simple-backtest/discussions)
- **Email**: guille2005_13@hotmail.com
