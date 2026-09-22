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


## Latest Five-Year MSFT Run

The latest multi-asset rerun supersedes the earlier embedded MSFT table. Use the
committed artifacts and manifest for the exact, current numbers:

```bash
PYTHONPATH=. python scripts/run_multi_asset_rerun.py \
  --output-dir research_outputs/multi_asset_rerun \
  --batch-size 6 --retries 2 --backoff 1
```

The multi-asset manifest records the common calendar, warm-up, fixed settings,
provider diagnosis, per-batch outcomes, exact command, code revision, and
artifact hashes. IWM.US is the investable Russell 2000 ETF proxy, not the index.
ETF fundamentals are not applicable; historical news is separate and unavailable
when GDELT is rate-limited.

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

## Multi-asset rerun: AAPL, MSFT, SPY, and IWM

The reproducible rerun command is:

```bash
PYTHONPATH=. python scripts/run_multi_asset_rerun.py \
  --output-dir research_outputs/multi_asset_rerun \
  --batch-size 6 --retries 2 --backoff 1
```

This requires Itoflow's `ito_quant` package and configured provider route. It
uses the latest common fully covered calendar, reserves 300 prior trading rows
for warm-up, keeps the existing $10,000/open/0.1% commission/final liquidation
settings, and writes `price_comparison.csv`, `fundamental_comparison.csv`, and
`diagnostics.json`. `IWM.US` is the investable Russell 2000 ETF proxy, not the
Russell 2000 index. `VOO.US` is the common S&P 500 ETF buy-and-hold benchmark.

The previous provider error was a network read timeout to `eodhd.com` over HTTPS
port 443; port 443 is not an HTTP status code. It does not establish
authentication failure or rate limiting. The rerun uses Itoflow `get_daily_prices`
in six-symbol batches with two retries and exponential backoff, and records
each batch outcome without logging credentials. Historical news is not part of
this rerun because GDELT remains unavailable/rate-limited.
