import argparse

import joblib
import numpy as np
import pandas as pd

from regime_networks import config
from regime_networks.data import check_data_files, cleaned_prices_path, file_sha256, manifest_status
from regime_networks.grid import grid_tag, labels_path, bundle_path
from regime_networks.reduction import transform_reduction
from regime_networks.regimes import apply_regime_order
from regime_networks.portfolio import (equal_weight_drift_returns, sessions_per_year, daily_risk_free, portfolio_return, transaction_costs, headline_metrics, exposure_stats, regime_schedule,
                                       fit_regime_forecast, forecast_next_regime)


# The Graph2Vec grid point of one market and construction at the headline window length.
def cell_grid_point(index_code, graph_type):
    return index_code, config.HEADLINE_WINDOW[0], graph_type, "graph2vec"


# The labels and frozen K-means model of one market and construction with its train and test embeddings, refusing a non-K-means headline method or a bundle that no longer reproduces the saved regimes.
def load_cell_model(index_code, graph_type):
    if config.HEADLINE_METHOD != "kmeans":
        raise ValueError(f"the portfolio's regime forecast is built on the K-means model, so it needs HEADLINE_METHOD = 'kmeans', not {config.HEADLINE_METHOD!r}")

    grid_point = cell_grid_point(index_code, graph_type)
    labels = pd.read_csv(labels_path(*grid_point), parse_dates=["window_date"]).set_index("window_date")
    bundle = joblib.load(bundle_path(*grid_point))

    if not {"g2v", "test_raw"} <= set(bundle):
        raise ValueError(f"{grid_tag(*grid_point[:3])}: the regime forecast needs the Graph2Vec bundle's stored train and test embeddings, which this bundle lacks; re-run scripts/run_sweeps.py")

    model, order = bundle["kmeans_model"], bundle["kmeans_order"]
    X_train, X_test = (transform_reduction(raw, bundle["scaler"], bundle["pca"]) for raw in (bundle["g2v"].train_vectors, bundle["test_raw"]))

    if not (np.array_equal(model.predict(X_train), model.labels_) and np.array_equal(apply_regime_order(np.concatenate([model.labels_, model.predict(X_test)]), order), labels["kmeans_regime"].to_numpy())):
        raise ValueError(f"{grid_tag(*grid_point[:3])}: the frozen K-means model no longer reproduces the saved regimes from its bundle's embeddings -- the bundle or labels are stale; re-run scripts/run_sweeps.py")

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


# One market's test-period trading days after the first test window's end date, its buy-and-hold sleeve returns, sessions per year and daily cash rate, stopping with diagnostics on missing or non-positive prices or degenerate sleeve returns.
def market_sleeve(index_code, first_test_window_end):
    path = cleaned_prices_path(index_code)
    check_data_files([path])
    prices = pd.read_csv(path, index_col="Date", parse_dates=["Date"]).sort_index()
    stock_returns = prices.pct_change(fill_method=None).iloc[1:]
    days = stock_returns.index[stock_returns.index > first_test_window_end]
    returns = stock_returns.loc[days]
    equity_return = equal_weight_drift_returns(returns)

    if prices.isna().to_numpy().any() or (prices <= 0).to_numpy().any():
        raise ValueError(f"{index_code}: the cleaned price panel has missing or non-positive prices, which the screen should have removed; returns cannot be computed without filling them, which would fake zero returns:\n{_sleeve_diagnostics(index_code, path, prices, returns, equity_return)}")

    if not equity_return.std() > config.MIN_EQUITY_RETURN_STD:
        raise ValueError(f"{index_code}: the equal-weight sleeve's daily returns are degenerate (standard deviation not above MIN_EQUITY_RETURN_STD):\n{_sleeve_diagnostics(index_code, path, prices, returns, equity_return)}")

    periods_per_year = sessions_per_year(days)
    print(f"{index_code}: {len(days)} test days ({days[0].date()} to {days[-1].date()}), {returns.shape[1]} stocks, {periods_per_year:.1f} sessions a year, cash rate {config.RISK_FREE_RATES_ANNUAL[index_code]:.2%}")
    return days, equity_return, periods_per_year, daily_risk_free(config.RISK_FREE_RATES_ANNUAL[index_code], periods_per_year)


# Backtest every strategy for one market and construction over the full test period, gross and net of transaction costs: the fixed-exposure strategies from the current regime, the regime forecast from the soft state times the training transition matrix, and a constant exposure at each timed strategy's average.
def backtest_cell(index_code, graph_type, sleeve):
    days, equity_return, periods_per_year, risk_free = sleeve
    labels, model, order, X_train, X_test = load_cell_model(index_code, graph_type)
    is_test = (labels["split"] == "test").to_numpy()
    test_dates, ranks = labels.index[is_test], labels["kmeans_regime"].to_numpy()[is_test]
    fit = fit_regime_forecast(X_train, labels["kmeans_regime"].to_numpy()[~is_test], model, order)
    forecast = forecast_next_regime(X_test, model, order, fit["temperature"], fit["P"])

    daily = regime_schedule(pd.Series(np.array(config.REGIME_NAMES)[ranks], index=test_dates, name="regime"), days).assign(equity_return=equity_return)
    window_exposure = pd.DataFrame({**{name: np.array(levels)[ranks] for name, levels in config.PORTFOLIO_STRATEGIES.items()}, config.REGIME_FORECAST_STRATEGY: 1 - forecast[:, -1]}, index=test_dates)
    exposure = window_exposure.reindex(daily["window_end_date"]).set_axis(days, axis=0)

    for timed, constant in config.CONSTANT_EXPOSURE_STRATEGIES.items():
        exposure[constant] = exposure[timed].mean()

    cell = {"index_code": index_code, "graph_type": graph_type}
    metric_rows, exposure_rows = [], []

    for name in config.PORTFOLIO_STRATEGY_NAMES:
        gross = portfolio_return(exposure[name], equity_return, risk_free)
        net = gross - transaction_costs(exposure[name])
        daily[f"{name} exposure"], daily[f"{name} gross return"], daily[f"{name} return"] = exposure[name], gross, net
        metric_rows += [{**cell, "costs": costs, "strategy": name, "days": len(days), "sessions_per_year": periods_per_year, **headline_metrics(returns, risk_free, periods_per_year)} for costs, returns in (("gross", gross), ("net", net))]
        exposure_rows.append({**cell, "strategy": name, **exposure_stats(exposure[name])})

    forecast_by_regime = pd.Series(forecast[:, -1]).groupby(ranks).mean().reindex(range(config.N_REGIMES))
    print(f"{index_code} {graph_type}: soft-state temperature {fit['multiple']:.4g} x the median squared centroid distance" + (" (AT THE EDGE of the searched grid)" if fit["at_edge"] else "")
          + f"; training P(Transitional -> Crisis) {fit['P'][1, -1]:.3f}; mean test forecast P(Crisis next) by current regime {dict(zip(config.REGIME_NAMES, forecast_by_regime.round(3).tolist()))}")
    return daily, metric_rows, exposure_rows, {**cell, "temperature_multiple": fit["multiple"], "temperature_at_edge": fit["at_edge"], **{f"P_{a.lower()}_{b.lower()}": fit["P"][i, j] for i, a in enumerate(config.REGIME_NAMES) for j, b in enumerate(config.REGIME_NAMES)},
                                               **{f"test_forecast_crisis_given_{name.lower()}": forecast_by_regime[r] for r, name in enumerate(config.REGIME_NAMES)}}


# Backtest every market under every construction, saving the headline construction's daily series to outputs/portfolio/{tag}_daily.csv (the figures' input) and the metric, exposure and forecast-fit tables of all cells to outputs/portfolio/.
def main():
    folder = config.OUTPUTS_DIR / "portfolio"
    folder.mkdir(parents=True, exist_ok=True)
    tables = {"metrics": [], "exposure": [], "forecast_fit": []}

    for index_code in config.INDEX_CODES:
        labels = pd.read_csv(labels_path(*cell_grid_point(index_code, config.PORTFOLIO_GRAPH_TYPES[0])), parse_dates=["window_date"])
        sleeve = market_sleeve(index_code, labels.loc[labels["split"] == "test", "window_date"].iloc[0])

        for graph_type in config.PORTFOLIO_GRAPH_TYPES:
            daily, metric_rows, exposure_rows, fit_row = backtest_cell(index_code, graph_type, sleeve)

            if graph_type == config.HEADLINE_GRAPH_TYPE:
                daily.rename_axis("date").to_csv(folder / f"{grid_tag(*cell_grid_point(index_code, graph_type)[:3])}_daily.csv", lineterminator="\n")

            tables["metrics"] += metric_rows
            tables["exposure"] += exposure_rows
            tables["forecast_fit"].append(fit_row)

    for name, rows in tables.items():
        table = pd.DataFrame(rows)
        table.to_csv(folder / f"{name}.csv", index=False, lineterminator="\n")
        print(f"\n{name}\n{table.round(4).to_string(index=False)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Backtest buy-and-hold, the regime-timed and regime-forecast strategies and their constant-exposure benchmarks, gross and net of transaction costs, for every market and construction over the full test period into outputs/portfolio/.").parse_args()
    main()
