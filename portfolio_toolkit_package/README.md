# Portfolio Toolkit

A small reusable Python package for portfolio performance evaluation and allocation optimization.

This package was built from the original notebook functions:

- downside deviation
- max drawdown
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- max Sharpe allocation
- max Sortino allocation
- max Calmar allocation
- portfolio evaluation table

## Installation

From the project folder:

```bash
pip install -e .
```

Or after pushing to GitHub:

```bash
pip install git+https://github.com/YOUR_USERNAME/portfolio-toolkit.git
```

## Basic Usage

```python
import pandas as pd
from portfolio_toolkit import PortfolioAnalyzer

# returns: rows = dates, columns = assets
# Use monthly excess returns if freq=12, daily excess returns if freq=252.
returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)

pa = PortfolioAnalyzer(returns, freq=12)

# Equal-weight portfolio metrics
metrics = pa.evaluate()
print(metrics)

# Optimize by Sharpe ratio
res = pa.optimize_sharpe()
print(res.weights)
print(res.metrics)

# Optimize by Sortino ratio
sortino_res = pa.optimize_sortino()
print(sortino_res.weights)

# Optimize by Calmar ratio
calmar_res = pa.optimize_calmar()
print(calmar_res.weights)

# Compare all strategies
comparison = pa.compare_strategies()
print(comparison)

# Get weights table
weights = pa.weights_table()
print(weights)
```

## With Risk-Free Rate

If your `returns` are raw asset returns, pass a risk-free rate series with the same frequency.
The class will align dates and subtract the risk-free rate automatically.

```python
pa = PortfolioAnalyzer(returns, risk_free=rf, freq=12)
```

If your returns are already excess returns, do not pass `risk_free`.

## Frequency

Use:

- `freq=12` for monthly data
- `freq=252` for daily data
- `freq=52` for weekly data

## GitHub Upload

```bash
git init
git add .
git commit -m "Initial portfolio toolkit package"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/portfolio-toolkit.git
git push -u origin main
```

## Notes

- The optimizer uses `scipy.optimize.minimize` with SLSQP.
- Default optimization is long-only with weights summing to 1.
- You can allow shorting with `long_only=False`, or pass custom bounds.

Example:

```python
pa.optimize_sharpe(long_only=False, bounds=(-1, 1))
```
