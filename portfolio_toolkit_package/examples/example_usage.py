import numpy as np
import pandas as pd

from portfolio_toolkit import PortfolioAnalyzer

# Example monthly excess returns for 4 assets
np.random.seed(42)
returns = pd.DataFrame(
    np.random.normal(0.008, 0.04, size=(60, 4)),
    columns=["Asset_A", "Asset_B", "Asset_C", "Asset_D"],
    index=pd.date_range("2020-01-31", periods=60, freq="M"),
)

pa = PortfolioAnalyzer(returns, freq=12)

print("Equal-weight metrics:")
print(pa.evaluate())

print("\nMax Sharpe weights:")
sharpe_result = pa.optimize_sharpe()
print(sharpe_result.weights)
print(sharpe_result.metrics)

print("\nStrategy comparison:")
print(pa.compare_strategies())

print("\nWeights table:")
print(pa.weights_table())
