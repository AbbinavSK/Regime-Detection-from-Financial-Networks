import numpy as np
import pandas as pd

from regime_networks import config
from regime_networks.regimes import cluster_kmeans, fit_regime_order, apply_regime_order, empirical_transitions

HEADLINE_METRICS = ["CAGR", "Annualised vol", "Sharpe", "Max drawdown", "Calmar"]


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


# Split a strategy's compounded return into the part earned by its equity exposure alone and the cash carry earned while out of equity.
def carry_decomposition(exposure, equity_return, risk_free):
    total = (1 + portfolio_return(exposure, equity_return, risk_free)).prod() - 1
    no_carry = (1 + exposure * equity_return).prod() - 1
    return {"total_return": total, "return_ex_carry": no_carry, "carry_contribution": total - no_carry}


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


# CAGR per unit of maximum drawdown.
def calmar_ratio(returns, periods_per_year):
    return cagr(returns, periods_per_year) / abs(max_drawdown(returns))


# The five headline metrics of a daily return series with periods_per_year sessions a year, keyed as in HEADLINE_METRICS.
def headline_metrics(returns, risk_free, periods_per_year):
    return dict(zip(HEADLINE_METRICS, (cagr(returns, periods_per_year), annualised_vol(returns, periods_per_year), sharpe_ratio(returns, risk_free, periods_per_year),
                                       max_drawdown(returns), calmar_ratio(returns, periods_per_year))))


# Exposure statistics of a daily exposure schedule: total absolute change, mean exposure, and number of days on which it changes.
def exposure_stats(exposure):
    return {"turnover": exposure.diff().abs().sum(), "time_in_market": exposure.mean(), "n_exposure_changes": int((exposure.diff().fillna(0) != 0).sum())}


# Positions of one circular block-bootstrap resample of n observations, blocks wrapping from the end to the start so every observation is drawn equally often.
def circular_block_indices(n, block_length, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
    return ((starts[:, None] + np.arange(block_length)) % n).ravel()[:n]


# Headline metrics on circular block-bootstrap resamples (Politis and Romano 1992) of a daily return series; a fixed seed makes every strategy's resamples use the same days.
def block_bootstrap(returns, risk_free, periods_per_year, block_length=config.BOOTSTRAP_BLOCK_LENGTH, n_resamples=config.BOOTSTRAP_N_RESAMPLES, seed=config.SEED):
    rng = np.random.default_rng(seed)
    values = returns.to_numpy()
    return pd.DataFrame([headline_metrics(pd.Series(values[circular_block_indices(len(values), block_length, rng)]), risk_free, periods_per_year) for _ in range(n_resamples)])


# The regime in force on each trading day and the end date of the window that set it: the latest window ending strictly before the day, since a regime detected at a window's close sets exposure from the next session.
def regime_schedule(regimes, days):
    position = regimes.index.searchsorted(days, side="left") - 1

    if (position < 0).any():
        raise ValueError(f"{int((position < 0).sum())} trading days fall on or before the first window's end date ({regimes.index[0].date()}), when no regime is known yet")

    return pd.DataFrame({"window_end_date": regimes.index[position], "regime": regimes.to_numpy()[position]}, index=days)


# Annualised root-mean-square of daily returns, the volatility measure of the volatility-targeting strategies (not demeaned, so a short period's own drift does not bias it).
def rms_volatility(returns, periods_per_year):
    return float(np.sqrt(np.mean(np.square(np.asarray(returns))) * periods_per_year))


# Training volatility of the risk proxy per regime and overall, from training sessions only, each in the regime of the window that closes it; raises for a regime with no session.
def training_regime_volatility(train_regime, risk_proxy, periods_per_year, k=config.N_REGIMES):
    sessions = risk_proxy.loc[(risk_proxy.index > train_regime.index[0]) & (risk_proxy.index <= train_regime.index[-1])]
    regime = regime_schedule(pd.Series(train_regime.to_numpy()[1:], index=train_regime.index[:-1]), sessions.index)["regime"].to_numpy()
    empty = [r for r in range(k) if not (regime == r).any()]

    if empty:
        raise ValueError(f"regimes {empty} close no training session, so their training volatility is undefined")

    return np.array([rms_volatility(sessions[regime == r], periods_per_year) for r in range(k)]), rms_volatility(sessions, periods_per_year)


# Trailing volatility known at each window's close: the annualised root-mean-square of the risk proxy's last TRAILING_VOL_SESSIONS daily returns up to and including the window's end date (NaN if fewer are available).
def trailing_volatility(risk_proxy, window_dates, periods_per_year, sessions=config.TRAILING_VOL_SESSIONS):
    return np.sqrt(risk_proxy.pow(2).rolling(sessions).mean() * periods_per_year).reindex(window_dates).to_numpy()


# Realised volatility of each window's holding period, the sessions whose exposure it set (grouped by regime_schedule's window_end_date), annualised root-mean-square; NaN for a window that set no session.
def holding_period_volatility(risk_proxy, window_end_date, window_dates, periods_per_year):
    return np.sqrt(risk_proxy.loc[window_end_date.index].pow(2).groupby(window_end_date.to_numpy()).mean() * periods_per_year).reindex(window_dates).to_numpy()


# Mean QLIKE loss of volatility forecasts against realised volatility (Patton 2011), robust to noise in the realised measure; lower is better, and windows without a realised value are skipped.
def qlike(forecast_vol, realised_vol):
    keep = ~np.isnan(realised_vol)
    ratio = np.square(realised_vol[keep]) / np.square(np.asarray(forecast_vol, dtype=float)[keep])
    return float(np.mean(ratio - np.log(ratio) - 1))


# Exposure scaling the sleeve to a target volatility given each window's volatility forecast, min(cap, target / forecast), so the long-only portfolio never borrows; a missing, zero or negative forecast or target raises instead of producing an invalid exposure.
def vol_target_exposure(forecast_vol, target_vol, cap=config.VOL_TARGET_EXPOSURE_CAP):
    forecast_vol = np.asarray(forecast_vol, dtype=float)

    if not (np.isfinite(forecast_vol).all() and (forecast_vol > 0).all() and np.isfinite(target_vol) and target_vol > 0):
        raise ValueError(f"volatility targeting needs finite positive forecasts and target; got target {target_vol} and {int((~np.isfinite(forecast_vol) | (forecast_vol <= 0)).sum())} invalid forecasts")

    return np.minimum(cap, target_vol / forecast_vol)


# Soft regime state of each window: a softmax over negative squared distances to a fitted K-means model's centroids divided by a temperature, columns in regime-rank order (order maps rank to cluster); a temperature near zero gives back the hard K-means regime.
def soft_state(X, model, order, temperature):
    distance = ((X[:, None, :] - model.cluster_centers_[order][None, :, :]) ** 2).sum(axis=2)
    weight = np.exp(-(distance - distance.min(axis=1, keepdims=True)) / temperature)
    return weight / weight.sum(axis=1, keepdims=True)


# One-step regime forecast of each window, the distribution of the next window's regime: its soft state q_t times the transition matrix P (rows the regime now, columns the regime next, both in rank order); a window's forecast uses only its own embedding.
def forecast_next_regime(X, model, order, temperature, P):
    return soft_state(X, model, order, temperature) @ P


# Mean log probability that one-step regime forecasts give the regime observed next, each probability floored at FORECAST_PROBABILITY_FLOOR so a structural zero costs a finite penalty instead of minus infinity.
def mean_log_score(forecast, next_regime, floor=config.FORECAST_PROBABILITY_FLOOR):
    return float(np.mean(np.log(np.maximum(forecast[np.arange(len(next_regime)), next_regime], floor))))


# Mean Brier score of one-step regime forecasts against the regime observed next (0 is perfect, 2 is certain and wrong).
def mean_brier(forecast, next_regime, k=config.N_REGIMES):
    return float(np.mean(((forecast - np.eye(k)[next_regime]) ** 2).sum(axis=1)))


# Fit the regime forecast on training windows: the empirical transition matrix, and the grid temperature that maximises the training log score (smallest wins ties); raises if the forecast is undefined.
def fit_regime_forecast(X_train, train_regime, model, order, k=config.N_REGIMES):
    P = empirical_transitions(train_regime, k, np.arange(k))[0]
    scale = np.median(((X_train[:, None, :] - model.cluster_centers_[None, :, :]) ** 2).sum(axis=2).min(axis=1))

    if np.isnan(P).any() or not scale > 0:
        raise ValueError(f"the regime forecast is undefined: regimes {np.flatnonzero(np.isnan(P).any(axis=1)).tolist()} have no training transition out of them, and the median squared distance to the nearest centroid is {scale}")

    multiples = np.logspace(*config.FORECAST_TEMPERATURE_MULTIPLES)
    scores = np.array([mean_log_score(forecast_next_regime(X_train[:-1], model, order, multiple * scale, P), train_regime[1:]) for multiple in multiples])
    best = int(np.argmax(scores))
    return {"P": P, "temperature": multiples[best] * scale, "multiple": multiples[best], "at_edge": best in (0, len(multiples) - 1)}


# K-means regimes of the window mean correlation alone, ranked by training mu: the fitted model, its regime order, and the train and test ranks.
def mu_only_regimes(train_mu, test_mu):
    train_labels, test_labels, model = cluster_kmeans(np.reshape(train_mu, (-1, 1)), np.reshape(test_mu, (-1, 1)))
    order = fit_regime_order(train_labels, np.ravel(train_mu))[0]
    return model, order, apply_regime_order(train_labels, order), apply_regime_order(test_labels, order)


# Fit the soft-state Markov forecast of one K-means model on training data, then forecast each test window's next regime and the volatility of the holding period that next window closes.
def _markov_vol_forecast(X_train, X_test, train_regime, train_dates, model, order, risk_proxy, periods_per_year):
    fit = fit_regime_forecast(X_train, train_regime, model, order)
    regime_vol, long_run_vol = training_regime_volatility(pd.Series(train_regime, index=train_dates), risk_proxy, periods_per_year)
    forecast = forecast_next_regime(X_test, model, order, fit["temperature"], fit["P"])
    return forecast, forecast @ regime_vol, {**fit, "regime_vol": regime_vol, "long_run_vol": long_run_vol}


# Every forecast of one market, fitted on training data only: the four scored regime forecasts, the three volatility forecasts, the target volatility and both models' fits.
def build_forecasts(labels, model, order, X_train, X_test, risk_proxy, periods_per_year):
    is_train = (labels["split"] == "train").to_numpy()
    train_regime, test_regime = labels["kmeans_regime"].to_numpy()[is_train], labels["kmeans_regime"].to_numpy()[~is_train]
    soft_forecast, regime_forecast_vol, fit = _markov_vol_forecast(X_train, X_test, train_regime, labels.index[is_train], model, order, risk_proxy, periods_per_year)
    mu_train, mu_test = (labels["mu"].to_numpy()[split].reshape(-1, 1) for split in (is_train, ~is_train))
    mu_model, mu_order, mu_train_regime, _ = mu_only_regimes(mu_train, mu_test)
    _, mu_forecast_vol, mu_fit = _markov_vol_forecast(mu_train, mu_test, mu_train_regime, labels.index[is_train], mu_model, mu_order, risk_proxy, periods_per_year)
    regime_forecasts = {"Unconditional": np.tile(np.bincount(train_regime, minlength=config.N_REGIMES) / len(train_regime), (len(test_regime), 1)), "Persistence": np.eye(config.N_REGIMES)[test_regime],
                        "Hard-state Markov": fit["P"][test_regime], "Soft-state Markov": soft_forecast}
    vol_forecasts = {config.REGIME_FORECAST_STRATEGY: regime_forecast_vol, config.TRAILING_VOL_STRATEGY: trailing_volatility(risk_proxy, labels.index[~is_train], periods_per_year),
                     config.MU_ONLY_STRATEGY: mu_forecast_vol}
    target_vol = {"calm": fit["regime_vol"][0], "long_run": fit["long_run_vol"]}[config.VOL_TARGET]
    return regime_forecasts, vol_forecasts, target_vol, {**fit, "mu": mu_fit}


# Contiguous (first day, last day) spans on which a daily series equals the given regime name.
def regime_spans(regime_series, regime_name):
    is_regime = regime_series == regime_name
    episode_id = (is_regime != is_regime.shift(fill_value=False)).cumsum()
    return [(g.index.min(), g.index.max()) for _, g in regime_series[is_regime].groupby(episode_id[is_regime])]


# Name each span after the event (a {date_str: name} dict) nearest its start if within max_days, else after its start date, numbering repeats.
def label_episodes(spans, events, max_days=config.EPISODE_EVENT_MAX_DAYS):
    event_dates = {name: pd.Timestamp(date) for date, name in events.items()}
    labels, counts = [], {}

    for start, _ in spans:
        nearest = min(event_dates, key=lambda name: abs((event_dates[name] - start).days))
        label = nearest if abs((event_dates[nearest] - start).days) < max_days else f"{start.date()}"
        counts[label] = counts.get(label, 0) + 1
        labels.append(f"{label} ({counts[label]})" if counts[label] > 1 else label)

    return labels
