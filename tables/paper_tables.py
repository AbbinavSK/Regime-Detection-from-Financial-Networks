import argparse

import numpy as np
import pandas as pd

from regime_networks import config
from regime_networks.data import NO_PRICES_REASON, DOWNLOAD_ERROR_REASON, exclusion_reasons, cleaned_prices_path
from regime_networks.grid import grid_position, headline_grid_point, labels_path
from regime_networks.regimes import empirical_transitions, aic_minimum_location
from regime_networks.portfolio import HEADLINE_METRICS

SCREEN_COLUMNS = {"nonpositive": "nonpositive_price", "incomplete": "incomplete_price_history", "jump": "return_jump", "flat_run": "flat_run"}
DIAGNOSTIC_STATISTICS = ["density", "avg_clustering", "degree_assortativity", "nmi_sector"]


# Initial constituents, removals under each screening reason and final stock count per market; a composite index sums its components' records with its own post-intersection screen, and unaccounted should be zero.
def data_summary():
    screen_columns = {reason: SCREEN_COLUMNS[rule] for rule, reason in exclusion_reasons().items()}
    rows = []

    for index_code in config.INDEX_CODES:
        parts = config.INDEX_COMPONENTS.get(index_code, [index_code])
        initial = sum(len(pd.read_csv(config.DATA_DIR / f"{code}_constituents.csv")) for code in parts)
        exclusions = pd.concat([pd.read_csv(config.DATA_DIR / f"{code}_exclusions.csv") for code in dict.fromkeys(parts + [index_code])], ignore_index=True)
        reasons = exclusions["reason"].map(lambda reason: "download_failed" if reason.startswith((NO_PRICES_REASON, DOWNLOAD_ERROR_REASON)) else screen_columns[reason])
        final = len(pd.read_csv(cleaned_prices_path(index_code), nrows=0).columns) - 1
        counts = {column: int((reasons == column).sum()) for column in ["download_failed", *SCREEN_COLUMNS.values()]}
        rows.append({"index_code": index_code, "initial_n": initial, **counts, "final_n": final, "unaccounted": initial - len(exclusions) - final})

    return pd.DataFrame(rows)


# Whether each sweep row's HMM fit has an absorbing state: an expected dwell (in windows) longer than the whole training sample, so the fitted chain effectively never leaves that regime.
def _absorbing_hmm(results):
    return results[[f"hmm_dwell_{name.lower()}" for name in config.REGIME_NAMES]].max(axis=1) > results["n_train_windows"]


# Number of grid points each descriptor was actually run over, counted from its sweep results.
def experimental_grid():
    rows = []

    for descriptor in config.DESCRIPTORS:
        results = pd.read_csv(config.OUTPUTS_DIR / f"{descriptor}_sweep_results.csv")
        rows.append({"descriptor": descriptor, "markets": results["index_code"].nunique(), "window_lengths": results["window_length"].nunique(),
                     "constructions": results["graph_type"].nunique() if "graph_type" in results else 1, "configurations": len(results)})

    return pd.DataFrame(rows)


# Out-of-sample VIX F1 of the S&P 500 Crisis regime for every Graph2Vec construction and window length, flagging HMM fits with an absorbing state or with regimes absent from the test windows.
def sp500_f1():
    results = pd.read_csv(config.OUTPUTS_DIR / "graph2vec_sweep_results.csv")
    results = results[results["index_code"] == config.VIX_INDEX].assign(order=lambda df: df["graph_type"].map(config.GRAPH_TYPES.index), hmm_absorbing=_absorbing_hmm)
    return results.sort_values(["order", "window_length"])[["graph_type", "window_length", "kmeans_test_point_f1", "hmm_test_point_f1", "hmm_absorbing", "hmm_regimes_absent_test"]]


# Test-period empirical self-transition probability (in windows) and occupancy of each regime per market and clustering method, for the headline configuration.
def cross_market_persistence():
    rows = []

    for index_code in config.INDEX_CODES:
        labels = pd.read_csv(labels_path(*headline_grid_point(index_code)))
        test = labels[labels["split"] == "test"]

        for method in ("kmeans", "hmm"):
            P, occupancy = empirical_transitions(test[f"{method}_regime"].to_numpy(), config.N_REGIMES, np.arange(config.N_REGIMES))
            rows.append({"index_code": index_code, "method": method, **{f"{name}_persistence": P[r, r] for r, name in enumerate(config.REGIME_NAMES)},
                         **{f"{name}_windows": int(occupancy[r]) for r, name in enumerate(config.REGIME_NAMES)}})

    return pd.DataFrame(rows)


# Mean out-of-sample S&P 500 VIX F1 over each descriptor's grid, with the grid size and how many of its HMM fits (all kept in the mean) have an absorbing state or regimes absent from the test windows.
def descriptor_f1():
    rows = []

    for descriptor in config.DESCRIPTORS:
        results = pd.read_csv(config.OUTPUTS_DIR / f"{descriptor}_sweep_results.csv")
        sp500 = results[results["index_code"] == config.VIX_INDEX]
        rows.append({"descriptor": descriptor, "grid_size": len(sp500), "kmeans_test_point_f1": sp500["kmeans_test_point_f1"].mean(), "hmm_test_point_f1": sp500["hmm_test_point_f1"].mean(),
                     "hmm_absorbing_fits": int(_absorbing_hmm(sp500).sum()), "hmm_fits_with_regimes_absent_test": int((sp500["hmm_regimes_absent_test"] > 0).sum())})

    return pd.DataFrame(rows)


# Per-regime medians of density, average clustering, degree assortativity and sector NMI under each construction (grouped by its own model's regimes), over training, test and all windows, with the number of windows.
def network_diagnostics():
    summary = pd.read_csv(config.OUTPUTS_DIR / "network_topology" / "regime_summary.csv")
    keys = ["graph_type", "split", "regime"]
    medians = summary[summary["statistic"].isin(DIAGNOSTIC_STATISTICS)].pivot_table(index=keys, columns="statistic", values="median", sort=False)
    table = summary.loc[summary["statistic"] == DIAGNOSTIC_STATISTICS[0], [*keys, "n_windows"]].merge(medians.reset_index(), on=keys, how="left")
    return table[[*keys, "n_windows", *DIAGNOSTIC_STATISTICS]].rename(columns={statistic: f"median_{statistic}" for statistic in DIAGNOSTIC_STATISTICS})


# Mean Graph2Vec OOV rate over all windows, by window length (rows) and construction (columns).
def geometry_oov():
    results = pd.read_csv(config.OUTPUTS_DIR / "embedding_geometry_results.csv")
    table = results.pivot(index="window_length", columns="graph_type", values="mean_oov_rate")
    return table[[g for g in config.GRAPH_TYPES if g in table.columns]].reset_index()


# Headline portfolio metrics per market and strategy, over the full test period and with the excluded episode removed.
def portfolio_performance():
    return pd.read_csv(config.OUTPUTS_DIR / "portfolio" / "metrics.csv")[["index_code", "sample", "strategy", *HEADLINE_METRICS]]


# Forecast quality per market on test windows: log and Brier scores of each one-step regime forecast against the next window's regime (all windows, and those before a regime change), and QLIKE of each volatility-targeting strategy's forecast against its holding period's realised volatility.
def forecast_scores():
    return pd.read_csv(config.OUTPUTS_DIR / "portfolio" / "forecast_scores.csv")


# For every grid point, the number of regimes with the lowest HMM AIC and whether that minimum is interior to the searched range, how far the AIC at the chosen number of regimes lies above it, and the relative fall in K-means inertia into and out of the chosen number.
def model_selection():
    results = pd.read_csv(config.OUTPUTS_DIR / "model_selection.csv")
    k, rows = config.N_REGIMES, []

    for _, grid_point in results.groupby("tag", sort=False):
        by_k = grid_point.set_index("k")
        aic = by_k["hmm_aic"]
        best_k = int(aic.idxmin()) if aic.notna().any() else np.nan
        rows.append({**grid_point.iloc[0][["descriptor", "index_code", "window_length", "graph_type"]].to_dict(), "best_k_aic": best_k, "aic_minimum_location": aic_minimum_location(best_k),
                     "aic_above_best_at_chosen_k": aic[k] - aic.min(), "inertia_drop_into_chosen_k": 1 - by_k.loc[k, "kmeans_inertia"] / by_k.loc[k - 1, "kmeans_inertia"],
                     "inertia_drop_after_chosen_k": 1 - by_k.loc[k + 1, "kmeans_inertia"] / by_k.loc[k, "kmeans_inertia"]})

    table = pd.DataFrame(rows)
    return table.iloc[sorted(range(len(table)), key=lambda i: grid_position(**table.iloc[i][["descriptor", "index_code", "window_length", "graph_type"]].to_dict()))].reset_index(drop=True)


# Every paper table's name and the function that builds it, in the order they are written.
def table_builders():
    return {"data_summary": data_summary, "experimental_grid": experimental_grid, "sp500_f1": sp500_f1, "cross_market_persistence": cross_market_persistence,
            "descriptor_f1": descriptor_f1, "network_diagnostics": network_diagnostics, "geometry_oov": geometry_oov, "portfolio_performance": portfolio_performance,
            "forecast_scores": forecast_scores, "model_selection": model_selection}


# A booktabs tabular of the table with floats at TABLE_DECIMALS decimals and missing values as "--", ready to wrap in a LaTeX table environment.
def _latex_tabular(table):
    cell = lambda value: "--" if pd.isna(value) else (f"{value:.{config.TABLE_DECIMALS}f}" if isinstance(value, float) else str(value).replace("_", "\\_"))
    header = " & ".join(str(column).replace("_", "\\_") for column in table.columns)
    body = [" & ".join(cell(value) for value in row) + " \\\\" for row in table.itertuples(index=False)]
    return "\n".join([f"\\begin{{tabular}}{{l{'r' * (table.shape[1] - 1)}}}", "\\toprule", header + " \\\\", "\\midrule", *body, "\\bottomrule", "\\end{tabular}", ""])


# Build each requested table from the saved outputs and write it to results/tables/ as CSV and a LaTeX tabular, skipping any whose inputs have not been produced yet and deleting that table's earlier files so a stale version cannot pass for a current one.
def main(names=None):
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    builders = table_builders()

    for name in builders if names is None else names:
        try:
            table = builders[name]()
        except FileNotFoundError as exc:
            stale = [path for path in (config.TABLES_DIR / f"{name}.csv", config.TABLES_DIR / f"{name}.tex") if path.exists()]

            for path in stale:
                path.unlink()

            print(f"skipped {name}: missing input {exc.filename}" + (f"; removed its earlier {', '.join(path.name for path in stale)}" if stale else ""))
            continue

        table.to_csv(config.TABLES_DIR / f"{name}.csv", index=False, lineterminator="\n")

        with open(config.TABLES_DIR / f"{name}.tex", "w", encoding="utf-8", newline="\n") as f:
            f.write(_latex_tabular(table))

        print(f"\n{name}\n{table.round(config.TABLE_DECIMALS).to_string(index=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble the paper's tables from the saved outputs into results/tables/ as CSV and LaTeX.")
    parser.add_argument("--table", choices=list(table_builders()) + ["all"], default="all", help="Table to build (default: all)")
    args = parser.parse_args()
    main(None if args.table == "all" else [args.table])
