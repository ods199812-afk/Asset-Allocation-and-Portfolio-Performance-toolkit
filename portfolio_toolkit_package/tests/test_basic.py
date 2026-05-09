import numpy as np
import pandas as pd

from portfolio_toolkit import PortfolioAnalyzer, evaluate_portfolio, max_drawdown


def test_max_drawdown_basic():
    r = pd.Series([0.1, -0.2, 0.05])
    assert max_drawdown(r) < 0


def test_evaluate_portfolio_keys():
    r = pd.Series([0.01, 0.02, -0.01, 0.03])
    metrics = evaluate_portfolio(r, freq=12)
    assert "Sharpe Ratio" in metrics
    assert "Calmar Ratio" in metrics


def test_optimizer_runs():
    returns = pd.DataFrame(
        np.random.normal(0.01, 0.03, size=(24, 3)),
        columns=["A", "B", "C"],
    )
    pa = PortfolioAnalyzer(returns, freq=12)
    res = pa.optimize_sharpe()
    assert len(res.weights) == 3
    assert abs(res.weights.sum() - 1) < 1e-6
