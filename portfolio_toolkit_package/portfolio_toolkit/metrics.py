from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd

ArrayLike = Union[pd.Series, pd.DataFrame, np.ndarray, list]


def _to_series(x: ArrayLike, name: str = "returns") -> pd.Series:
    """Convert 1D array-like data to a clean pandas Series."""
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} must be 1D. Got DataFrame with {x.shape[1]} columns.")
        s = x.iloc[:, 0].copy()
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D. Got shape {arr.shape}.")
        s = pd.Series(arr, name=name)
    return s.dropna().astype(float)


def _to_dataframe(x: ArrayLike, name: str = "returns") -> pd.DataFrame:
    """Convert 2D return data to a clean pandas DataFrame."""
    if isinstance(x, pd.DataFrame):
        df = x.copy()
    elif isinstance(x, pd.Series):
        df = x.to_frame()
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            df = pd.DataFrame(arr, columns=[name])
        elif arr.ndim == 2:
            df = pd.DataFrame(arr)
        else:
            raise ValueError(f"{name} must be 1D or 2D. Got shape {arr.shape}.")
    return df.dropna(how="all").astype(float)


def max_drawdown(returns: ArrayLike) -> float:
    """
    Calculate maximum drawdown from a return series.

    Parameters
    ----------
    returns:
        Periodic simple returns, e.g. monthly returns or daily returns.

    Returns
    -------
    float
        Maximum drawdown as a negative number. Example: -0.25 means -25%.
    """
    r = _to_series(returns)
    if r.empty:
        return np.nan

    cumulative = (1.0 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min())


def downside_deviation(returns: ArrayLike, freq: int = 12, annualize: bool = True) -> float:
    """Calculate downside deviation using min(return, 0)."""
    r = _to_series(returns)
    if r.empty:
        return np.nan

    downside_returns = np.minimum(r, 0.0)
    dd = float(np.sqrt(np.mean(downside_returns**2)))
    return dd * np.sqrt(freq) if annualize else dd


def evaluate_portfolio(returns: ArrayLike, excess_returns: ArrayLike, freq: int = 12) -> Dict[str, float]:
    """
    Evaluate one portfolio return series.

    The input should already be excess returns if you want excess-return metrics.
    For example, if your portfolio return is monthly return and risk-free rate is
    monthly risk-free rate, pass ``portfolio_return - risk_free_rate``.
    """
    r = _to_series(returns, name="portfolio_returns")
    er = _to_series(excess_returns, name = "portfolio_excess_returns")
    
    if r.empty:
        return {
            "Annual Excess Return": np.nan,
            "Annual Volatility": np.nan,
            "Annual Downside Deviation": np.nan,
            "Max Drawdown": np.nan,
            "Sharpe Ratio": np.nan,
            "Sortino Ratio": np.nan,
            "Calmar Ratio": np.nan,
        }

    annual_excess_return = float(er.mean() * freq)
    annual_volatility = float(r.std(ddof=1) * np.sqrt(freq))
    annual_downside_deviation = downside_deviation(r, freq=freq, annualize=True)
    mdd = max_drawdown(r)

    sharpe = np.nan if annual_volatility == 0 else annual_excess_return / annual_volatility
    sortino = np.nan if annual_downside_deviation == 0 else annual_excess_return / annual_downside_deviation
    calmar = np.nan if mdd == 0 else annual_excess_return / abs(mdd)

    return {
        "Annual Excess Return": annual_excess_return,
        "Annual Volatility": annual_volatility,
        "Annual Downside Deviation": float(annual_downside_deviation),
        "Max Drawdown": float(mdd),
        "Sharpe Ratio": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "Sortino Ratio": float(sortino) if not np.isnan(sortino) else np.nan,
        "Calmar Ratio": float(calmar) if not np.isnan(calmar) else np.nan,
    }


def portfolio_excess_return(weights: ArrayLike, mu: ArrayLike) -> float:
    """Expected portfolio excess return: w dot mu."""
    w = np.asarray(weights, dtype=float)
    m = np.asarray(mu, dtype=float)
    return float(np.dot(w, m))


def portfolio_volatility(weights: ArrayLike, cov: ArrayLike) -> float:
    """Portfolio volatility: sqrt(w.T @ cov @ w)."""
    w = np.asarray(weights, dtype=float)
    c = np.asarray(cov, dtype=float)
    variance = float(w.T @ c @ w)
    return float(np.sqrt(max(variance, 0.0)))
