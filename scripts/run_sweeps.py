import argparse
import pickle

import joblib
import numpy as np
import pandas as pd

from regime_networks import config
from regime_networks.networks import load_networks_cached
from regime_networks.grid import grid_tag, grid_position, descriptor_grid, merge_grid_results, labels_path, bundle_path, spectral_features_path
from regime_networks.graph2vec import Graph2Vec
from regime_networks.descriptors import SPECTRAL_FEATURE_COLUMNS, window_centrality, window_spectral
from regime_networks.reduction import fit_reduction, transform_reduction
from regime_networks.regimes import fit_and_rank_regimes, empirical_transitions, apply_regime_order
from regime_networks.validation import split_train_test, load_vix, validate_regime


# Raw train/test descriptor matrices for one grid point, plus the descriptor's own results columns and the objects its model bundle stores; the spectral feature table is saved to results_directory.
def descriptor_features(descriptor, train, test, graph_type, stock_names, index_code, window_length, tag, results_directory):
    if descriptor == "graph2vec":
        key = config.GRAPH_KEYS[graph_type]
        g2v = Graph2Vec()
        train_raw, train_oov = g2v.fit(train[key])
        test_raw, test_oov = g2v.infer(test[key])
        columns = {"oov_rate_train_mean": float(np.mean(train_oov)) if train_oov else float("nan"),
                   "oov_rate_test_mean": float(np.mean(test_oov)) if test_oov else float("nan")}
        return train_raw, test_raw, columns, {"g2v": g2v, "test_raw": test_raw, "test_oov": test_oov}

    if descriptor == "eigencentrality":
        key = config.GRAPH_KEYS[graph_type]
        stats = pd.DataFrame([window_centrality(G, stock_names) for G in train[key] + test[key]])
        mean_pr = stats["pr"].mean()

        if mean_pr < config.PR_LOCALISATION_FLAG:
            print(f"  FLAG: {tag} mean PR ({mean_pr:.2f}) is below {config.PR_LOCALISATION_FLAG} out of N -- v1 has localised onto a hub.")

        v1 = np.stack(stats["v1"].tolist())
        n_train = len(train["window_dates"])
        columns = {"mean_pr": mean_pr, "mean_lcc_frac": stats["lcc_frac"].mean(), "pr_localised_flag": bool(mean_pr < config.PR_LOCALISATION_FLAG)}
        return v1[:n_train], v1[n_train:], columns, {}

    stats = pd.DataFrame([window_spectral(C, stock_names, window_length) for C in train["correlation_matrices"] + test["correlation_matrices"]])
    stats.index = pd.DatetimeIndex(pd.to_datetime(list(train["window_dates"]) + list(test["window_dates"])))
    stats.index.name = "window_date"
    stats.to_pickle(spectral_features_path(index_code, window_length, results_directory))
    X = stats[SPECTRAL_FEATURE_COLUMNS].to_numpy()
    n_train = len(train["window_dates"])
    return X[:n_train], X[n_train:], {}, {}


# Output folders of one sweep (results tables and transitions, per-window labels, model bundles); a quick smoke run writes under outputs/quick/ so it never overwrites the full sweep.
def sweep_directories(quick=False):
    if quick:
        return {"results": config.OUTPUTS_DIR / "quick", "labels": config.OUTPUTS_DIR / "quick" / "labels", "models": config.OUTPUTS_DIR / "quick" / "models"}

    return {"results": config.OUTPUTS_DIR, "labels": config.LABELS_DIR, "models": config.MODELS_DIR}


# Pickle a descriptor's transitions, one entry per grid point, in full-sweep grid order.
def _save_transitions(transitions, path):
    with open(path, "wb") as f:
        pickle.dump(dict(sorted(transitions.items(), key=lambda item: grid_position(*item[0]))), f)


# Run one (descriptor, index, window length, graph construction) grid point end to end on its network data: descriptor, reduction, K-means/HMM, VIX F1 (S&P 500), transitions, per-window labels and model bundle; every train-vs-test comparison uses causally decoded train labels.
def run_grid_point(descriptor, index_code, window_length, graph_type, data, vix, n_seeds, directories):
    tag = grid_tag(index_code, window_length, graph_type)
    train, test = split_train_test(data, cutoff=config.TRAIN_TEST_CUTOFF)
    train_mu, test_mu = train["mu_values"], test["mu_values"]
    train_dates, test_dates = train["window_dates"], test["window_dates"]

    X_train_raw, X_test_raw, descriptor_columns, bundle_extras = descriptor_features(descriptor, train, test, graph_type, data["stock_names"], index_code, window_length, tag, directories["results"])
    X_train, scaler, pca = fit_reduction(X_train_raw)
    X_test = transform_reduction(X_test_raw, scaler, pca)

    row = {"index_code": index_code, "window_length": window_length, **({"graph_type": graph_type} if graph_type else {}),
           "n_train_windows": len(train_dates), "n_test_windows": len(test_dates), "pca_dims": pca.n_components_, **descriptor_columns}

    results = fit_and_rank_regimes(X_train, X_test, train_mu, n_seeds)
    labels = pd.DataFrame({"window_date": pd.to_datetime(list(train_dates) + list(test_dates)), "split": ["train"] * len(train_dates) + ["test"] * len(test_dates), "mu": list(train_mu) + list(test_mu)})
    transitions_entry = {}

    for method_name, method_result in results.items():
        train_labels, test_labels = method_result["train_causal_labels"], method_result["test_labels"]
        labels[f"{method_name}_regime"] = apply_regime_order(np.concatenate([train_labels, test_labels]), method_result["order"])

        if method_name == "hmm":
            labels["hmm_regime_smoothed"] = pd.array(list(apply_regime_order(method_result["train_labels"], method_result["order"])) + [pd.NA] * len(test_dates), dtype="Int64")

            for r, name in enumerate(config.REGIME_NAMES):
                row[f"hmm_dwell_{name.lower()}"] = float(method_result["dwell"][r])

        if index_code == config.VIX_INDEX:
            train_validation = validate_regime(train_labels, train_dates, vix, config.VIX_CRISIS_THRESHOLD, method_result["crisis_id"])
            test_validation = validate_regime(test_labels, test_dates, vix, config.VIX_CRISIS_THRESHOLD, method_result["crisis_id"])
            row[f"{method_name}_train_point_precision"], row[f"{method_name}_train_point_recall"], row[f"{method_name}_train_point_f1"] = train_validation["point"]
            row[f"{method_name}_test_point_precision"], row[f"{method_name}_test_point_recall"], row[f"{method_name}_test_point_f1"] = test_validation["point"]
            labels["vix"] = np.concatenate([train_validation["series_eod"], test_validation["series_eod"]])

        train_transitions, train_occupancy = empirical_transitions(train_labels, config.N_REGIMES, method_result["order"])
        test_transitions, test_occupancy = empirical_transitions(test_labels, config.N_REGIMES, method_result["order"])
        row[f"{method_name}_regimes_absent_test"] = int(((train_occupancy > 0) & (test_occupancy == 0)).sum())
        transitions_entry[method_name] = {"train": train_transitions, "test": test_transitions, "train_occupancy": train_occupancy, "test_occupancy": test_occupancy}

    directories["labels"].mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_path(index_code, window_length, graph_type, descriptor, directories["labels"]), index=False, lineterminator="\n")
    directories["models"].mkdir(parents=True, exist_ok=True)
    joblib.dump({**bundle_extras, "scaler": scaler, "pca": pca,
                 "kmeans_model": results["kmeans"]["model"], "kmeans_order": results["kmeans"]["order"], "hmm_model": results["hmm"]["model"], "hmm_order": results["hmm"]["order"]},
                bundle_path(index_code, window_length, graph_type, descriptor, directories["models"]))

    grid_key = (index_code, window_length) + ((graph_type,) if graph_type else ())
    return row, grid_key, transitions_entry


# Delete everything an earlier run left for a grid point that has now failed (its results row, transitions entry, labels and model bundle), so no stale output can pass for a current one.
def _discard_grid_point(descriptor, index_code, window_length, graph_type, results_path, transitions, transitions_path, directories):
    for path in (labels_path(index_code, window_length, graph_type, descriptor, directories["labels"]), bundle_path(index_code, window_length, graph_type, descriptor, directories["models"])):
        path.unlink(missing_ok=True)

    if results_path.exists():
        results = pd.read_csv(results_path, float_precision="round_trip")
        stale = (results["index_code"] == index_code) & (results["window_length"] == window_length) & ((results["graph_type"] == graph_type) if "graph_type" in results else True)
        results[~stale].to_csv(results_path, index=False, lineterminator="\n")

    if transitions.pop((index_code, window_length) + ((graph_type,) if graph_type else ()), None) is not None:
        _save_transitions(transitions, transitions_path)


# Sweep one descriptor over the index x window-length x graph-construction grid, merging each finished grid point into its results CSV and transitions pickle so partial and repeated runs keep every other grid point, and removing the earlier outputs of any grid point that fails.
def run_descriptor(descriptor, index_codes, window_lengths, graph_types, vix, n_seeds, directories):
    results_path = directories["results"] / f"{descriptor}_sweep_results.csv"
    transitions_path = directories["results"] / f"{descriptor}_transition_matrices.pkl"
    transitions = {}

    if transitions_path.exists():
        with open(transitions_path, "rb") as f:
            transitions = pickle.load(f)

    rows, failures, loaded = [], [], {}
    grid = [(i, t, g) for i, t, g in descriptor_grid(descriptor, index_codes, window_lengths) if graph_types is None or g is None or g in graph_types]

    for index_code, window_length, graph_type in grid:
        label = f"{descriptor} / {index_code} / {window_length}d" + (f" / {graph_type}" if graph_type else "")
        print(f"running {label} ...")

        try:
            row, grid_key, transitions_entry = run_grid_point(descriptor, index_code, window_length, graph_type, load_networks_cached(loaded, index_code, window_length), vix, n_seeds, directories)
        except Exception as e:
            print(f"grid point FAILED: {label}: {e!r}")
            failures.append(label)
            _discard_grid_point(descriptor, index_code, window_length, graph_type, results_path, transitions, transitions_path, directories)
            continue

        rows.append(row)
        transitions[grid_key] = transitions_entry
        merge_grid_results(results_path, rows)
        _save_transitions(transitions, transitions_path)

    print(f"wrote {len(rows)} rows to {results_path} and {len(transitions)} entries to {transitions_path}")

    if failures:
        print(f"{len(failures)}/{len(grid)} grid points failed: {failures}")

    return failures


# Run the requested descriptors' sweeps (every index and window length unless given); returns True only if every grid point succeeded.
def main(descriptors=None, index_codes=None, window_lengths=None, graph_types=None, n_seeds=None, quick=False):
    directories = sweep_directories(quick)
    directories["results"].mkdir(parents=True, exist_ok=True)
    vix = load_vix()
    descriptors = config.DESCRIPTORS if descriptors is None else descriptors
    index_codes = config.INDEX_CODES if index_codes is None else index_codes
    window_lengths = config.WINDOW_LENGTHS if window_lengths is None else window_lengths
    n_seeds = config.HMM_N_SEEDS if n_seeds is None else n_seeds
    failures = [f for descriptor in descriptors for f in run_descriptor(descriptor, index_codes, window_lengths, graph_types, vix, n_seeds, directories)]
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep the regime-detection grid (index x window length x graph construction) for one or all descriptors.")
    parser.add_argument("--descriptor", choices=config.DESCRIPTORS + ["all"], default="all", help="Descriptor to sweep (default: all three)")
    parser.add_argument("--n-seeds", type=int, default=config.HMM_N_SEEDS, help=f"HMM restarts (default: {config.HMM_N_SEEDS})")
    parser.add_argument("--quick", action="store_true", help="Smoke-test one grid point per descriptor (config.QUICK_SWEEP_POINT) with at most config.QUICK_HMM_SEEDS HMM restarts, writing under outputs/quick/")
    args = parser.parse_args()

    descriptors = config.DESCRIPTORS if args.descriptor == "all" else [args.descriptor]

    if args.quick:
        index_code, window_length, graph_type = config.QUICK_SWEEP_POINT
        ok = main(descriptors, index_codes=[index_code], window_lengths=[window_length], graph_types=[graph_type], n_seeds=min(args.n_seeds, config.QUICK_HMM_SEEDS), quick=True)
    else:
        ok = main(descriptors, n_seeds=args.n_seeds)

    raise SystemExit(0 if ok else 1)
