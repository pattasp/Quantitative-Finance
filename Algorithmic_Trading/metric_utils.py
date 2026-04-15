# Evaluation performance metrics module :
# Here Equity is a pandas Series indexed by datetime.

import numpy as np
import pandas as pd

def compute_returns(equity):
    """
    Compute simple returns from an equity curve.
    R(t) = (C(t) - C(t-1)) / C(t-1)

    Parameters
    ----------
    equity : pd.Series
        Equity curve C(t)

    Returns
    -------
    pd.Series
        Return series aligned to equity index (first value dropped)
    """
    if equity is None:
        raise ValueError("equity is None")

    equity = pd.Series(equity).dropna()
    if len(equity) < 2:
        return pd.Series([], dtype=float)

    rets = equity.pct_change().dropna()
    rets.name = "returns"
    return rets


def cumulative_return(equity):
    """
    Cumulative return over the whole period.
    C(T)/C(0) - 1
    """
    equity = pd.Series(equity).dropna()
    if len(equity) == 0:
        return np.nan
    if equity.iloc[0] == 0:
        return np.nan
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def annualized_volatility(returns, annualization=252):
    """
    Annualized volatility = std(returns) * sqrt(annualization)
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(annualization))


def sharpe_ratio_from_equity(equity, annualization=252, rf=0.0):
    """
    Sharpe ratio computed from an equity curve.
    Uses daily returns and annualizes with sqrt(annualization).

    Sharpe = (mean(excess_returns) / std(excess_returns)) * sqrt(annualization)

    rf is the risk-free rate PER PERIOD (daily if daily data).
    For your project, rf=0 is fine.
    """
    rets = compute_returns(equity)
    if len(rets) < 2:
        return np.nan

    excess = rets - rf
    mu = excess.mean()
    sigma = excess.std(ddof=1)

    if sigma == 0 or np.isnan(sigma):
        return np.nan

    return float((mu / sigma) * np.sqrt(annualization))


def max_drawdown(equity):
    """
    Compute maximum drawdown (most negative peak-to-trough % drop).
    Drawdown(t) = C(t)/max_{s<=t} C(s) - 1
    MaxDD = min_t Drawdown(t)

    Returns
    -------
    float
        max drawdown as a negative number (e.g. -0.35 for -35%)
    """
    equity = pd.Series(equity).dropna()
    if len(equity) == 0:
        return np.nan

    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def compute_equity_metrics(equity, annualization=252, rf=0.0):
    """
    Compute a bundle of metrics from an equity curve.

    Returns dict with:
      - cum_return
      - sharpe
      - volatility
      - max_drawdown
      - final_equity
      - start_equity
      - n_periods
    """
    equity = pd.Series(equity).dropna()

    metrics = {}
    metrics["start_equity"] = float(equity.iloc[0]) if len(equity) > 0 else np.nan
    metrics["final_equity"] = float(equity.iloc[-1]) if len(equity) > 0 else np.nan
    metrics["n_periods"] = int(len(equity))

    rets = compute_returns(equity)
    metrics["cum_return"] = cumulative_return(equity)
    metrics["volatility"] = annualized_volatility(rets, annualization=annualization)
    metrics["sharpe"] = sharpe_ratio_from_equity(equity, annualization=annualization, rf=rf)
    metrics["max_drawdown"] = max_drawdown(equity)

    return metrics   
