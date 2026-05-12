from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import (
    _to_dataframe,
    evaluate_portfolio,
    max_drawdown,
    portfolio_excess_return,
    portfolio_volatility,
)


@dataclass
class OptimizationResult:
    objective: str
    weights: pd.Series
    portfolio_returns: pd.Series
    metrics: Dict[str, float]
    success: bool
    message: str


class PortfolioAnalyzer:
    """
    Reusable class for portfolio evaluation and allocation optimization.

    Parameters
    ----------
    returns:
        DataFrame of asset returns. Rows are dates, columns are assets.
        These can be raw returns or excess returns. If you pass raw returns and
        risk_free is provided, the class will convert them to excess returns.
    risk_free:
        Optional risk-free rate series. It should have the same frequency as returns.
    freq:
        Annualization frequency. Use 12 for monthly data, 252 for daily data.
    """

    def __init__(self, returns, risk_free: Optional[Union[float, int, pd.Series]] = None, freq: int = 12):
        self.raw_returns = _to_dataframe(returns, name="asset_returns")
        self.freq = freq

        if risk_free is None:
            self.risk_free = None
            self.excess_returns = self.raw_returns.copy()
        elif isinstance(risk_free, (int, float)):
            self.risk_free = float(risk_free)
            self.excess_returns = self.raw_returns.copy() - self.risk_free 
        elif isinstance(risk_free, pd.Series):
            rf = pd.Series(risk_free).astype(float)

            self.raw_returns.index = pd.to_datetime(self.raw_returns.index).normalize()
            rf.index = pd.to_datetime(rf.index).normalize()

            aligned_returns, aligned_rf = self.raw_returns.align(
                rf,
                axis=0,
                join="inner"
            )
            self.raw_returns = aligned_returns.dropna(how="any")
            self.risk_free = aligned_rf.loc[self.raw_returns.index]
            self.excess_returns = self.raw_returns.sub(self.risk_free, axis=0)
        if self.excess_returns.empty:
            raise ValueError("No valid return observations after alignment/dropna.")

        self.assets = list(self.excess_returns.columns)
        self.n_assets = len(self.assets)

    def mean_returns(self, annualize: bool = False, use_excess: bool = False) -> pd.Series:
        data = self.excess_returns if use_excess else self.raw_returns
        mu = data.mean()
        return mu * self.freq if annualize else mu

    def covariance(self, annualize: bool = False) -> pd.DataFrame:
        cov = self.raw_returns.cov()
        return cov * self.freq if annualize else cov

    def portfolio_excess_returns(self, weights) -> pd.Series:
        w = self._validate_weights(weights)
        return self.excess_returns @ w

    def portfolio_raw_returns(self, weights) -> pd.Series:
        w = self._validate_weights(weights)
        return self.raw_returns @ w 

    def evaluate(self, weights=None) -> Dict[str, float]:
        """Evaluate a weighted portfolio. Default is equal weight."""
        if weights is None:
            weights = self.equal_weights().values
        port_excess_returns = self.portfolio_excess_returns(weights)
        port_raw_returns = self.portfolio_raw_returns(weights)
        
        return evaluate_portfolio(port_raw_returns, port_excess_returns, freq=self.freq)

    def equal_weights(self) -> pd.Series:
        return pd.Series(np.repeat(1 / self.n_assets, self.n_assets), index=self.assets, name="weight")

    def optimize(
        self,
        objective: str = "sharpe",
        long_only: bool = True,
        bounds: Optional[Tuple[float, float]] = None,
        initial_weights=None,
    ) -> OptimizationResult:
        """
        Optimize weights by Sharpe, Sortino, or Calmar ratio.

        objective:
            One of {'sharpe', 'sortino', 'calmar'}.
        long_only:
            If True, each weight is constrained between 0 and 1.
        bounds:
            Optional custom bounds applied to every asset, e.g. (-1, 1).
        initial_weights:
            Optional starting point. Default is equal weight.
        """
        objective = objective.lower().strip()
        valid = {"sharpe", "sortino", "calmar"}
        if objective not in valid:
            raise ValueError(f"objective must be one of {valid}.")

        if initial_weights is None:
            x0 = self.equal_weights().values
        else:
            x0 = self._validate_weights(initial_weights)

        if bounds is None:
            bounds = (0.0, 1.0) if long_only else (-1.0, 1.0)
        opt_bounds = [bounds for _ in range(self.n_assets)]
        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

        result = minimize(
            fun=lambda w: self._negative_objective(w, objective),
            x0=x0,
            method="SLSQP",
            bounds=opt_bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        weights = pd.Series(result.x, index=self.assets, name=f"{objective}_weight")
        port_raw_returns = self.portfolio_raw_returns(weights.values)
        port_excess_returns = self.portfolio_excess_returns(weights.values)
        
        metrics = evaluate_portfolio(
            port_raw_returns,
            port_excess_returns,
            freq=self.freq
        )

        return OptimizationResult(
            objective=objective,
            weights=weights,
            portfolio_returns=port_raw_returns,
            metrics=metrics,
            success=bool(result.success),
            message=str(result.message),
        )

    def optimize_sharpe(self, **kwargs) -> OptimizationResult:
        return self.optimize(objective="sharpe", **kwargs)

    def optimize_sortino(self, **kwargs) -> OptimizationResult:
        return self.optimize(objective="sortino", **kwargs)

    def optimize_calmar(self, **kwargs) -> OptimizationResult:
        return self.optimize(objective="calmar", **kwargs)

    def compare_strategies(self, include_equal_weight: bool = True) -> pd.DataFrame:
        """Run equal weight, max Sharpe, max Sortino, and max Calmar, then compare metrics."""
        rows = {}

        if include_equal_weight:
            rows["Equal Weight"] = self.evaluate(self.equal_weights().values)

        for name, method in [
            ("Max Sharpe", self.optimize_sharpe),
            ("Max Sortino", self.optimize_sortino),
            ("Max Calmar", self.optimize_calmar),
        ]:
            res = method()
            rows[name] = res.metrics

        return pd.DataFrame(rows).T

    def weights_table(self) -> pd.DataFrame:
        """Return optimized weights for Sharpe, Sortino, and Calmar in one table."""
        results = {
            "Max Sharpe": self.optimize_sharpe().weights,
            "Max Sortino": self.optimize_sortino().weights,
            "Max Calmar": self.optimize_calmar().weights,
        }
        return pd.DataFrame(results)

    def _validate_weights(self, weights) -> np.ndarray:
        if isinstance(weights, pd.Series):
            weights = weights.reindex(self.assets).values
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or len(w) != self.n_assets:
            raise ValueError(f"weights must be 1D with length {self.n_assets}.")
        if np.any(np.isnan(w)):
            raise ValueError("weights contain NaN.")
        return w

    def _negative_objective(self, weights, objective: str) -> float:
        w = self._validate_weights(weights)
        
        port_raw_returns = self.portfolio_raw_returns(w)
        port_excess_returns = self.portfolio_excess_returns(w)
        
        if objective == "sharpe":
            mu = self.excess_returns.mean().values
            cov = self.raw_returns.cov().values
            ret = portfolio_excess_return(weights, mu)
            vol = portfolio_volatility(weights, cov)
            return 1e6 if vol == 0 else -ret / vol

        annual_excess_return = port_excess_returns.mean() * self.freq
        annual_raw_return = port_raw_returns.mean() * self.freq
        
        if objective == "sortino":
            downside_returns = np.minimum(port_returns, 0.0)
            downside_dev = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(self.freq)
            return 1e6 if downside_dev == 0 else -annual_excess_return / downside_dev

        if objective == "calmar":
            mdd = max_drawdown(port_raw_returns)
            return 1e6 if mdd == 0 else -annual_return / abs(mdd)

        raise ValueError(f"Unknown objective: {objective}")
