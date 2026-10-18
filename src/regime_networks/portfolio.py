import numpy as np
import pandas as pd

from regime_networks import config
from regime_networks.regimes import empirical_transitions

HEADLINE_METRICS = ["CAGR", "Annualised vol", "Sharpe", "Max drawdown"]


# Daily return of an equal-weight portfolio bought on the first day and never rebalanced, so each stock's weight drifts with its own compounded return.
def equal_weight_drift_returns(stock_returns):
    value = (1.0 + stock_returns).cumprod().mean(axis=1)
    return (value / value.shift(1, fill_value=1.0) - 1.0).rename("equity_return")


# A market's number of trading sessions per year, measured from its own sequence of session dates (intervals between the first and last date over the calendar years they span).
def sessions_per_year(dates):
    return (len(dates) - 1) / ((dates[-1] - dates[0]).days / config.CALENDAR_DAYS_PER_YEAR)


# Constant per-session risk-free return compounding to the given annual rate over periods_per_year sessions.
def daily_risk_free(annual_rate, periods_per_year):
    return (1 + annual_rate) ** (1 / periods_per_year) - 1


# Daily return with the given fraction of capital in the equity sleeve and the rest earning the risk-free rate.
def portfolio_return(exposure, equity_return, risk_free):
    return exposure * equity_return + (1 - exposure) * risk_free


# Daily transaction cost of an exposure schedule: rate times the absolute change in exposure on each day, the first day paying for the purchase from cash.
def transaction_costs(exposure):
    return config.TRANSACTION_COST * exposure.diff().fillna(exposure.iloc[0]).abs()


# Compound annual growth rate of a daily return series with periods_per_year sessions a year.
def cagr(returns, periods_per_year):
    return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1


# Annualised standard deviation of a daily return series with periods_per_year sessions a year.
def annualised_vol(returns, periods_per_year):
    return returns.std() * np.sqrt(periods_per_year)


# Annualised Sharpe ratio of daily returns in excess of the risk-free rate, with periods_per_year sessions a year.
def sharpe_ratio(returns, risk_free, periods_per_year):
    excess = returns - risk_free
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


# Fractional fall of the compounded value below its running peak, per day, counting the starting value of 1 as the first peak so a first-day loss is a drawdown.
def drawdown_series(returns):
    cumulative = (1 + returns).cumprod()
    return cumulative / np.maximum(cumulative.cummax(), 1.0) - 1


# Deepest drawdown of a daily return series (a negative fraction).
def max_drawdown(returns):
    return drawdown_series(returns).min()


# The headline metrics of a daily return series with periods_per_year sessions a year, keyed as in HEADLINE_METRICS.
def headline_metrics(returns, risk_free, periods_per_year):
    return dict(zip(HEADLINE_METRICS, (cagr(returns, periods_per_year), annualised_vol(returns, periods_per_year), sharpe_ratio(returns, risk_free, periods_per_year), max_drawdown(returns))))


# Exposure statistics of a daily exposure schedule: total absolute change and mean exposure.
def exposure_stats(exposure):
    return {"turnover": exposure.diff().abs().sum(), "time_in_market": exposure.mean()}


# The regime in force on each trading day and the end date of the window that set it: the latest window ending strictly before the day, since a regime detected at a window's close sets exposure from the next session.
def regime_schedule(regimes, days):
    position = regimes.index.searchsorted(days, side="left") - 1

    if (position < 0).any():
        raise ValueError(f"{int((position < 0).sum())} trading days fall on or before the first window's end date ({regimes.index[0].date()}), when no regime is known yet")

    return pd.DataFrame({"window_end_date": regimes.index[position], "regime": regimes.to_numpy()[position]}, index=days)


# Soft regime state of each window: a softmax over negative squared distances to a fitted K-means model's centroids divided by a temperature, columns in regime-rank order (order maps rank to cluster); a temperature near zero gives back the hard K-means regime.
def soft_state(X, model, order, temperature):
    distance = ((X[:, None, :] - model.cluster_centers_[order][None, :, :]) ** 2).sum(axis=2)
    weight = np.exp(-(distance - distance.min(axis=1, keepdims=True)) / temperature)
    return weight / weight.sum(axis=1, keepdims=True)


# One-step regime forecast of each window, the distribution of the next window's regime: its soft state q_t times the transition matrix P (rows the regime now, columns the regime next, both in rank order); a window's forecast uses only its own embedding.
def forecast_next_regime(X, model, order, temperature, P):
    return soft_state(X, model, order, temperature) @ P


# Mean log probability that one-step regime forecasts give the regime observed next, each probability floored at FORECAST_PROBABILITY_FLOOR so a structural zero costs a finite penalty instead of minus infinity.
def mean_log_score(forecast, next_regime):
    return float(np.mean(np.log(np.maximum(forecast[np.arange(len(next_regime)), next_regime], config.FORECAST_PROBABILITY_FLOOR))))


# Fit the regime forecast on training windows: the empirical transition matrix, and the grid temperature that maximises the training log score (smallest wins ties); raises if the forecast is undefined.
def fit_regime_forecast(X_train, train_regime, model, order):
    P = empirical_transitions(train_regime, config.N_REGIMES, np.arange(config.N_REGIMES))[0]
    scale = np.median(((X_train[:, None, :] - model.cluster_centers_[None, :, :]) ** 2).sum(axis=2).min(axis=1))

    if np.isnan(P).any() or not scale > 0:
        raise ValueError(f"the regime forecast is undefined: regimes {np.flatnonzero(np.isnan(P).any(axis=1)).tolist()} have no training transition out of them, and the median squared distance to the nearest centroid is {scale}")

    multiples = np.logspace(*config.FORECAST_TEMPERATURE_MULTIPLES)
    scores = np.array([mean_log_score(forecast_next_regime(X_train[:-1], model, order, multiple * scale, P), train_regime[1:]) for multiple in multiples])
    best = int(np.argmax(scores))
    return {"P": P, "temperature": multiples[best] * scale, "multiple": multiples[best], "at_edge": best in (0, len(multiples) - 1)}


# Contiguous (first day, last day) spans on which a daily series equals the given regime name.
def regime_spans(regime_series, regime_name):
    is_regime = regime_series == regime_name
    episode_id = (is_regime != is_regime.shift(fill_value=False)).cumsum()
    return [(g.index.min(), g.index.max()) for _, g in regime_series[is_regime].groupby(episode_id[is_regime])]
