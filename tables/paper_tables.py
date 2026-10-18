import argparse

import numpy as np
import pandas as pd

from regime_networks import config
from regime_networks.data import NO_PRICES_REASON, DOWNLOAD_ERROR_REASON, exclusion_reasons, cleaned_prices_path
from regime_networks.grid import grid_position, headline_grid_point, labels_path, descriptor_grid
from regime_networks.regimes import empirical_transitions, aic_minimum_location
from regime_networks.validation import binary_f1, mu_rule_crisis, lagged_f1
from regime_networks.portfolio import HEADLINE_METRICS
from regime_networks.geometry import geometry_path, regime_separation

SCREEN_COLUMNS = {"nonpositive": "nonpositive_price", "incomplete": "incomplete_price_history", "jump": "return_jump", "flat_run": "flat_run"}
PROFILE_STATISTICS = ["density", "avg_clustering", "degree_assortativity", "sector_modularity"]


# Initial constituents, removals under each screening reason and final stock count per market; a composite index sums its components' records with its own post-intersection screen, and the step stops if a market's constituents, removals and retained stocks do not add up.
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

        if initial - len(exclusions) - final != 0:
            raise ValueError(f"{index_code}: {initial} constituents minus {len(exclusions)} removals is not the {final} stocks retained -- the constituent, exclusion and cleaned-price files in data/ disagree; download the market again")

        rows.append({"index_code": index_code, "initial_n": initial, **counts, "final_n": final})

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


# Out-of-sample S&P 500 VIX F1 of every descriptor at every construction and window length, under both clustering methods, with each HMM fit's degeneracy flags; one row per grid point, in grid order, with no ranking.
def descriptor_benchmark():
    rows = []

    for descriptor in config.DESCRIPTORS:
        results = pd.read_csv(config.OUTPUTS_DIR / f"{descriptor}_sweep_results.csv")
        sp500 = results[results["index_code"] == config.VIX_INDEX].assign(graph_type=lambda df: df["graph_type"] if "graph_type" in df else None, hmm_absorbing=_absorbing_hmm)
        rows += [{"descriptor": descriptor, "graph_type": row.graph_type if isinstance(row.graph_type, str) else "--", "window_length": row.window_length, "kmeans_test_point_f1": row.kmeans_test_point_f1,
                  "hmm_test_point_f1": row.hmm_test_point_f1, "hmm_absorbing": bool(row.hmm_absorbing), "hmm_regimes_absent_test": row.hmm_regimes_absent_test} for row in sp500.itertuples()]

    table = pd.DataFrame(rows)
    position = [grid_position(config.VIX_INDEX, t, None if g == "--" else g, d) for d, g, t in zip(table["descriptor"], table["graph_type"], table["window_length"])]
    return table.iloc[sorted(range(len(table)), key=position.__getitem__)].reset_index(drop=True)


# Agreement of each Graph2Vec K-means Crisis regime with the mu-quantile rule matched to its training Crisis share, over all and over test windows, per market, window length and construction; for the S&P 500 also both test VIX F1 scores.
def mu_rule_agreement():
    rows = []

    for index_code, window_length, graph_type in descriptor_grid("graph2vec"):
        labels = pd.read_csv(labels_path(index_code, window_length, graph_type, "graph2vec"))
        is_train, is_test = (labels["split"] == "train").to_numpy(), (labels["split"] == "test").to_numpy()
        crisis = (labels["kmeans_regime"] == config.N_REGIMES - 1).astype(int).to_numpy()
        rule = mu_rule_crisis(labels["mu"].to_numpy()[is_train], labels["mu"].to_numpy(), crisis[is_train].mean())
        row = {"index_code": index_code, "window_length": window_length, "graph_type": graph_type, "agreement_all": (crisis == rule).mean(), "agreement_test": (crisis[is_test] == rule[is_test]).mean()}

        if index_code == config.VIX_INDEX:
            vix_crisis = (labels["vix"] > config.VIX_CRISIS_THRESHOLD).astype(int).to_numpy()[is_test]
            row.update({"model_test_f1": binary_f1(crisis[is_test], vix_crisis)[2], "mu_rule_test_f1": binary_f1(rule[is_test], vix_crisis)[2]})

        rows.append(row)

    return pd.DataFrame(rows)


# Training and test K-means transition structure of every market's Graph2Vec regimes under each construction at the headline window length: the direct Calm-to-Crisis and Crisis-to-Calm probabilities and every regime's self-transition probability.
def transition_structure():
    rows = []

    for index_code in config.INDEX_CODES:
        for graph_type in config.GRAPH_TYPES:
            labels = pd.read_csv(labels_path(index_code, config.HEADLINE_WINDOW[0], graph_type, "graph2vec"))

            for split in ("train", "test"):
                P = empirical_transitions(labels.loc[labels["split"] == split, "kmeans_regime"].to_numpy(), config.N_REGIMES, np.arange(config.N_REGIMES))[0]
                rows.append({"index_code": index_code, "graph_type": graph_type, "split": split, "calm_to_crisis": P[0, -1], "crisis_to_calm": P[-1, 0],
                             **{f"{name.lower()}_persistence": P[r, r] for r, name in enumerate(config.REGIME_NAMES)}, "transitional_to_crisis": P[1, -1]})

    return pd.DataFrame(rows)


# How far each S&P 500 Graph2Vec K-means Crisis regime trails VIX stress, per window length and construction: the F1 over all windows without a lag, and the lag in windows (and sessions, and as a share of the window) that maximises it within VIX_LAG_WINDOWS, with that F1.
def vix_lag():
    rows = []

    for _, window_length, graph_type in descriptor_grid("graph2vec", [config.VIX_INDEX]):
        labels = pd.read_csv(labels_path(config.VIX_INDEX, window_length, graph_type, "graph2vec"))
        pred, actual = (labels["kmeans_regime"] == config.N_REGIMES - 1).astype(int).to_numpy(), (labels["vix"] > config.VIX_CRISIS_THRESHOLD).astype(int).to_numpy()
        scores = {lag: lagged_f1(pred, actual, lag) for lag in config.VIX_LAG_WINDOWS}
        best, shift = max(scores, key=scores.get), dict(config.WINDOW_CONFIGS)[window_length]
        rows.append({"window_length": window_length, "graph_type": graph_type, "f1_no_lag": scores[0], "best_lag_windows": best, "best_lag_sessions": best * shift, "best_lag_share_of_window": best * shift / window_length, "f1_at_best_lag": scores[best]})

    return pd.DataFrame(rows)


# The regime profile: per-regime medians over all windows of density, average clustering, degree assortativity and sector modularity, for every market, window length and construction (grouped by its own model's regimes), with the number of windows.
def regime_profile():
    summary = pd.read_csv(config.OUTPUTS_DIR / "network_topology" / "regime_summary.csv")
    summary = summary[summary["split"] == "all"]
    keys = ["index_code", "window_length", "graph_type", "regime"]
    medians = summary[summary["statistic"].isin(PROFILE_STATISTICS)].pivot_table(index=keys, columns="statistic", values="median", sort=False)
    table = summary.loc[summary["statistic"] == PROFILE_STATISTICS[0], [*keys, "n_windows"]].merge(medians.reset_index(), on=keys, how="left")
    return table[[*keys, "n_windows", *PROFILE_STATISTICS]].rename(columns={statistic: f"median_{statistic}" for statistic in PROFILE_STATISTICS})


# How well the Graph2Vec embedding separates the regimes, and how far its test windows lie from its training windows, per window length and construction of GEOMETRY_INDEX: silhouettes on the cosine distances of the geometry step and mean distances within and between the two periods.
def geometry_separation():
    rows = []

    for window_length in config.WINDOW_LENGTHS:
        for graph_type in config.GRAPH_TYPES:
            with np.load(geometry_path(config.GEOMETRY_INDEX, window_length, graph_type)) as saved:
                rows.append({"window_length": window_length, "graph_type": graph_type, "stress": float(saved["stress"]), **regime_separation(saved["distance"], saved["regime_labels"], int(saved["n_train"]))})

    return pd.DataFrame(rows)


# Sensitivity of each market's headline regimes to the Graph2Vec seed alone, from the seed check (outputs/seed_robustness.csv): the minimum and maximum over the robustness seeds of test agreement, adjusted Rand index, Crisis Jaccard index and test Crisis share against the SEED fit, and its test Crisis share; for the S&P 500 also the test VIX F1.
def seed_robustness():
    results = pd.read_csv(config.OUTPUTS_DIR / "seed_robustness.csv")
    reference = results[results["reference"]].set_index("index_code")
    rows = []

    for index_code, seeds in results[~results["reference"]].groupby("index_code", sort=False):
        row = {"index_code": index_code, "n_seeds": len(seeds), "reference_test_crisis_share": reference.loc[index_code, "test_crisis_share"]}

        for column in ("agreement_test", "adjusted_rand_index", "crisis_jaccard", "test_crisis_share", "test_vix_f1"):
            row.update({f"{column}_min": seeds[column].min(), f"{column}_max": seeds[column].max()})

        rows.append(row)

    return pd.DataFrame(rows)


# Pearson and Spearman correlation of the threshold network's density with mu over every window, per market and window length: the check that threshold density is the market-mode proxy.
def density_mu():
    windows = pd.read_csv(config.OUTPUTS_DIR / "network_topology" / "window_topology.csv")
    threshold = windows[windows["graph_type"] == "threshold"]
    return pd.DataFrame([{"index_code": index_code, "window_length": window_length, "n_windows": len(cell), "pearson": cell["density"].corr(cell["mu"]), "spearman": cell["density"].corr(cell["mu"], method="spearman")}
                         for (index_code, window_length), cell in threshold.groupby(["index_code", "window_length"], sort=False)])


# Headline portfolio metrics per market, construction and strategy over the full test period, gross and net of transaction costs, with each strategy's average exposure and turnover.
def portfolio_performance():
    keys = ["index_code", "graph_type", "strategy"]
    metrics = pd.read_csv(config.OUTPUTS_DIR / "portfolio" / "metrics.csv")[[*keys[:2], "costs", keys[2], *HEADLINE_METRICS]]
    exposure = pd.read_csv(config.OUTPUTS_DIR / "portfolio" / "exposure.csv")[[*keys, "time_in_market", "turnover"]].rename(columns={"time_in_market": "average_exposure"})
    return metrics.merge(exposure, on=keys, how="left")


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
    return {"data_summary": data_summary, "experimental_grid": experimental_grid, "cross_market_persistence": cross_market_persistence,
            "descriptor_benchmark": descriptor_benchmark, "mu_rule_agreement": mu_rule_agreement, "transition_structure": transition_structure, "regime_profile": regime_profile, "density_mu": density_mu, "geometry_separation": geometry_separation, "seed_robustness": seed_robustness, "vix_lag": vix_lag, "portfolio_performance": portfolio_performance, "model_selection": model_selection}


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
