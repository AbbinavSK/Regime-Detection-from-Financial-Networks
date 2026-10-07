import argparse

import joblib
import numpy as np
import pandas as pd

from regime_networks import config
from regime_networks.data import check_data_files, cleaned_prices_path, file_sha256, manifest_status
from regime_networks.grid import headline_grid_point, labels_path, bundle_path
from regime_networks.reduction import transform_reduction
from regime_networks.regimes import apply_regime_order
from regime_networks.portfolio import (equal_weight_drift_returns, sessions_per_year, daily_risk_free, portfolio_return, carry_decomposition, headline_metrics, exposure_stats,
                                       block_bootstrap, regime_schedule, build_forecasts, mean_log_score, mean_brier, holding_period_volatility, qlike, vol_target_exposure,
                                       regime_spans, label_episodes)


# The headline labels and frozen K-means model of one market with its train and test embeddings, refusing a non-K-means headline or a bundle that no longer reproduces the saved regimes.
def load_headline_model(index_code):
    if config.HEADLINE_METHOD != "kmeans":
        raise ValueError(f"the portfolio's regime forecast is built on the headline K-means model, so it needs HEADLINE_METHOD = 'kmeans', not {config.HEADLINE_METHOD!r}")

    labels = pd.read_csv(labels_path(*headline_grid_point(index_code)), parse_dates=["window_date"]).set_index("window_date")
    bundle = joblib.load(bundle_path(*headline_grid_point(index_code)))

    if not {"g2v", "test_raw"} <= set(bundle):
        raise ValueError(f"{index_code}: the regime forecast needs the headline Graph2Vec bundle's stored train and test embeddings, which this bundle lacks; re-run scripts/run_sweeps.py")

    model, order = bundle["kmeans_model"], bundle["kmeans_order"]
    X_train, X_test = (transform_reduction(raw, bundle["scaler"], bundle["pca"]) for raw in (bundle["g2v"].train_vectors, bundle["test_raw"]))

    if not (np.array_equal(model.predict(X_train), model.labels_) and np.array_equal(apply_regime_order(np.concatenate([model.labels_, model.predict(X_test)]), order), labels["kmeans_regime"].to_numpy())):
        raise ValueError(f"{index_code}: the frozen K-means model no longer reproduces the saved regimes from its bundle's embeddings -- the bundle or labels are stale; re-run scripts/run_sweeps.py")

    return labels, model, order, X_train, X_test


# Diagnostics to trace a bad sleeve input: the file and its manifest status, the panel's shape and gaps, the return statistics, and checks for a mis-read panel or a mis-computed return.
def _sleeve_diagnostics(index_code, path, prices, returns, equity_return):
    finite = returns.to_numpy()[np.isfinite(returns.to_numpy())]
    values, test_rows = prices.to_numpy(), prices.index[1:] >= returns.index[0] if len(returns) else np.zeros(len(prices) - 1, dtype=bool)
    numpy_returns = (values[1:] / values[:-1] - 1)[test_rows]
    return "\n".join([f"  market: {index_code}", f"  file: {path} (sha256 {file_sha256(path)}, {manifest_status(path)})",
                      f"  price panel: {prices.shape[0]} dates x {prices.shape[1]} stocks, {prices.index.min().date()} to {prices.index.max().date()}",
                      f"  missing prices: {int(prices.isna().to_numpy().sum())}, non-positive prices: {int((prices <= 0).to_numpy().sum())}",
                      f"  test-period days whose every price equals the previous day's (a mis-read panel): {int((np.diff(values, axis=0)[test_rows] == 0).all(axis=1).sum())} of {int(test_rows.sum())}",
                      f"  test-period returns: {returns.shape[0]} days, missing {int(returns.isna().to_numpy().sum())}, exactly zero {int((finite == 0).sum())} of {finite.size}"
                      + (f", min {finite.min():.4f}, max {finite.max():.4f}" if finite.size else ""),
                      f"  the same returns recomputed with NumPy (a mis-computed return if they disagree): exactly zero {int((numpy_returns == 0).sum())} of {numpy_returns.size}; pandas use_numexpr={pd.get_option('compute.use_numexpr')}",
                      f"  equal-weight sleeve daily return std: {equity_return.std():.3e} (floor MIN_EQUITY_RETURN_STD={config.MIN_EQUITY_RETURN_STD})"])


# Backtest every strategy over one market's test period and score its forecasts, stopping with diagnostics on missing or non-positive prices or degenerate sleeve returns.
def backtest_market(index_code):
    labels, model, order, X_train, X_test = load_headline_model(index_code)
    is_test = (labels["split"] == "test").to_numpy()
    test_dates, ranks = labels.index[is_test], labels[f"{config.HEADLINE_METHOD}_regime"].to_numpy()[is_test]
    path = cleaned_prices_path(index_code)
    check_data_files([path])
    prices = pd.read_csv(path, index_col="Date", parse_dates=["Date"]).sort_index()
    stock_returns = prices.pct_change(fill_method=None).iloc[1:]
    days = stock_returns.index[stock_returns.index > test_dates[0]]
    returns = stock_returns.loc[days]
    equity_return = equal_weight_drift_returns(returns)

    if prices.isna().to_numpy().any() or (prices <= 0).to_numpy().any():
        raise ValueError(f"{index_code}: the cleaned price panel has missing or non-positive prices, which the screen should have removed; returns cannot be computed without filling them, which would fake zero returns:\n{_sleeve_diagnostics(index_code, path, prices, returns, equity_return)}")

    if not equity_return.std() > config.MIN_EQUITY_RETURN_STD:
        raise ValueError(f"{index_code}: the equal-weight sleeve's daily returns are degenerate (standard deviation not above MIN_EQUITY_RETURN_STD):\n{_sleeve_diagnostics(index_code, path, prices, returns, equity_return)}")

    periods_per_year = sessions_per_year(days)
    risk_free = daily_risk_free(config.RISK_FREE_RATES_ANNUAL[index_code], periods_per_year)
    risk_proxy = stock_returns.mean(axis=1)
    regime_forecasts, vol_forecasts, target_vol, fit = build_forecasts(labels, model, order, X_train, X_test, risk_proxy, periods_per_year)

    daily = regime_schedule(pd.Series(np.array(config.REGIME_NAMES)[ranks], index=test_dates, name="regime"), days).assign(equity_return=equity_return)
    window_exposure = pd.DataFrame({**{name: np.array(levels)[ranks] for name, levels in config.PORTFOLIO_STRATEGIES.items()}, **{name: vol_target_exposure(vol, target_vol) for name, vol in vol_forecasts.items()}}, index=test_dates)
    exposure = window_exposure.reindex(daily["window_end_date"]).set_axis(days, axis=0)
    strategy_returns, exposure_rows = {}, []

    for name in config.PORTFOLIO_STRATEGY_NAMES:
        strategy_returns[name] = daily[f"{name} return"] = portfolio_return(exposure[name], equity_return, risk_free)
        daily[f"{name} exposure"] = exposure[name]
        exposure_rows.append({"index_code": index_code, "strategy": name, **exposure_stats(exposure[name]), **carry_decomposition(exposure[name], equity_return, risk_free)})

    crisis_spans = regime_spans(daily["regime"], config.REGIME_NAMES[-1])
    episode_names = label_episodes(crisis_spans, config.CRISIS_EVENTS)
    episode_rows, crisis_days = [], pd.DatetimeIndex([])

    for (start, end), episode in zip(crisis_spans, episode_names):
        mask = (days >= start) & (days <= end)
        crisis_days = crisis_days.append(days[mask])
        episode_rows.append({"index_code": index_code, "episode": episode, "start": start.date(), "end": end.date(), "days": int(mask.sum()), **{name: (1 + ret[mask]).prod() - 1 for name, ret in strategy_returns.items()}})

    outside = ~days.isin(crisis_days)
    episode_rows.append({"index_code": index_code, "episode": "Outside Crisis episodes", "days": int(outside.sum()), **{name: (1 + ret[outside]).prod() - 1 for name, ret in strategy_returns.items()}})

    samples = {"full": np.ones(len(days), dtype=bool)}
    excluded_spans = [span for span, episode in zip(crisis_spans, episode_names) if episode == config.EXCLUDED_EPISODE or episode.startswith(f"{config.EXCLUDED_EPISODE} (")]

    if excluded_spans:
        samples[f"excluding_{config.EXCLUDED_EPISODE}"] = ~np.logical_or.reduce([(days >= start) & (days <= end) for start, end in excluded_spans])
        print(f"{index_code}: {config.EXCLUDED_EPISODE} episode excluded from the second sample: {', '.join(f'{start.date()} to {end.date()}' for start, end in excluded_spans)}, {int((~samples[f'excluding_{config.EXCLUDED_EPISODE}']).sum())} days")
    else:
        print(f"{index_code}: no Crisis episode labelled {config.EXCLUDED_EPISODE} in the test period, so only the full sample is reported")

    metric_rows = [{"index_code": index_code, "sample": sample, "strategy": name, "days": int(keep.sum()), "sessions_per_year": periods_per_year, **headline_metrics(ret[keep], risk_free, periods_per_year)} for sample, keep in samples.items() for name, ret in strategy_returns.items()]

    bootstrap_rows = []

    for name, ret in strategy_returns.items():
        quantiles = block_bootstrap(ret, risk_free, periods_per_year).quantile(list(config.BOOTSTRAP_QUANTILES))
        bootstrap_rows += [{"index_code": index_code, "strategy": name, "metric": metric, **{f"q{round(100 * q)}": quantiles.loc[q, metric] for q in config.BOOTSTRAP_QUANTILES}} for metric in quantiles.columns]

    next_regime, switches = ranks[1:], ranks[1:] != ranks[:-1]
    realised_vol = holding_period_volatility(risk_proxy, daily["window_end_date"], test_dates, periods_per_year)
    score_rows = [{"index_code": index_code, "forecast": name, "windows": windows, "n_windows": int(keep.sum()), "log_score": mean_log_score(forecast[:-1][keep], next_regime[keep]), "brier": mean_brier(forecast[:-1][keep], next_regime[keep])}
                  for name, forecast in regime_forecasts.items() for windows, keep in (("all", np.ones(len(next_regime), dtype=bool)), ("before a switch", switches)) if keep.any()]
    score_rows += [{"index_code": index_code, "forecast": name, "windows": "all", "n_windows": int(np.isfinite(realised_vol).sum()), "qlike": qlike(vol, realised_vol)} for name, vol in vol_forecasts.items()]
    print(f"{index_code}: {len(test_dates)} test windows, {len(days)} test days ({days[0].date()} to {days[-1].date()}), {returns.shape[1]} stocks, {periods_per_year:.1f} sessions a year, cash rate {config.RISK_FREE_RATES_ANNUAL[index_code]:.2%}")

    for name, model_fit in (("headline", fit), ("mu-only", fit["mu"])):
        print(f"{index_code}: {name} soft-state temperature {model_fit['multiple']:.4g} x the median squared centroid distance" + (" (AT THE EDGE of the searched grid)" if model_fit["at_edge"] else "")
              + f"; training volatility {dict(zip(config.REGIME_NAMES, model_fit['regime_vol'].round(4).tolist()))}")

    print(f"{index_code}: target ({config.VOL_TARGET}) {target_vol:.4f}; risk proxy and sleeve correlate {np.corrcoef(equity_return, risk_proxy.loc[days])[0, 1]:.3f} over the test period")
    return daily, {"metrics": metric_rows, "exposure_carry": exposure_rows, "episodes": episode_rows, "bootstrap": bootstrap_rows, "forecast_scores": score_rows}


# Backtest every market, saving each market's daily series to outputs/portfolio/{INDEX}_daily.csv and the metric, exposure and carry, episode, bootstrap and forecast-score tables of all markets to outputs/portfolio/.
def main():
    folder = config.OUTPUTS_DIR / "portfolio"
    folder.mkdir(parents=True, exist_ok=True)
    tables = {}

    for index_code in config.INDEX_CODES:
        daily, market_tables = backtest_market(index_code)
        daily.rename_axis("date").to_csv(folder / f"{index_code}_daily.csv", lineterminator="\n")

        for name, rows in market_tables.items():
            tables[name] = tables.get(name, []) + rows

    for name, rows in tables.items():
        table = pd.DataFrame(rows)
        table.to_csv(folder / f"{name}.csv", index=False, lineterminator="\n")
        print(f"\n{name}\n{table.round(4).to_string(index=False)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Backtest the fixed-exposure and volatility-targeting strategies on every market's test period, and score the regime and volatility forecasts, into outputs/portfolio/.").parse_args()
    main()
