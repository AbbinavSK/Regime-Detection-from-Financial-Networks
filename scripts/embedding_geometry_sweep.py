import argparse

import numpy as np
import pandas as pd

from regime_networks import config
from regime_networks.networks import load_networks
from regime_networks.grid import merge_grid_results
from regime_networks.validation import split_train_test
from regime_networks import geometry


# Load the train/test split once for one window_length; every graph_type within it reuses this split.
def _load_split(window_length):
    data = load_networks(config.GEOMETRY_INDEX, window_length)
    train, test = split_train_test(data, cutoff=config.TRAIN_TEST_CUTOFF)
    dates = list(train["window_dates"]) + list(test["window_dates"])
    n_train = len(train["window_dates"])
    return train, test, dates, n_train


# Build one graph_type's Graph2Vec vector/distance/regime-labels from one window_length's shared train/test split.
def _graph2vec_entry(train, test, dates, n_train, window_length, graph_type):
    graph_key = config.GRAPH_KEYS[graph_type]
    train_graphs, test_graphs = train[graph_key], test[graph_key]
    edge_counts = [G.number_of_edges() for G in train_graphs + test_graphs]

    bundle, train_raw, test_raw, train_oov, test_oov = geometry.load_graph2vec_raw(config.GEOMETRY_INDEX, window_length, graph_type)

    if len(test_raw) != len(test_graphs):
        raise ValueError(f"the stored bundle has {len(test_raw)} test windows but the current networks have {len(test_graphs)} -- the bundle is stale (e.g. the data were extended); re-run scripts/run_sweeps.py --descriptor graph2vec")

    prepared = geometry.prepare_graph2vec(np.concatenate([train_raw, test_raw]))
    regime_labels = geometry.bundle_regime_labels(bundle, test_raw, n_train)

    return {"oov": train_oov + test_oov, "dates": dates, "n_train": n_train,
            "edge_counts": edge_counts, "regime_labels": regime_labels,
            "distance": geometry.distance_matrix(prepared)}


# Build one outputs/embedding_geometry_results.csv row for one (window_length, graph_type) from its geometry entry and fitted MDS stress.
def _summary_row(entry, stress, window_length, graph_type):
    mn, med, mx = geometry.offdiag_distance_summary(entry["distance"])
    return {"index_code": config.GEOMETRY_INDEX, "window_length": window_length, "graph_type": graph_type,
            "descriptor": "graph2vec", "metric": "cosine",
            "n_windows": entry["distance"].shape[0], "n_train": entry["n_train"], "stress": float(stress),
            "offdiag_dist_min": mn, "offdiag_dist_median": med, "offdiag_dist_max": mx,
            "mean_oov_rate": float(np.mean(entry["oov"])), "median_oov_rate": float(np.median(entry["oov"])),
            "mean_edge_count": float(np.mean(entry["edge_counts"])), "median_edge_count": float(np.median(entry["edge_counts"]))}


# Save everything the geometry figures need for one (window_length, graph_type): distances, MDS coordinates and stress, regime labels, edge counts, OOV rates, window dates and the train size.
def _save_geometry(entry, coords, stress, window_length, graph_type):
    path = geometry.geometry_path(config.GEOMETRY_INDEX, window_length, graph_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, distance=entry["distance"], coords=coords, stress=stress, regime_labels=entry["regime_labels"],
                        edge_counts=np.array(entry["edge_counts"]), oov=np.array(entry["oov"]), dates=np.array(pd.to_datetime(entry["dates"]).strftime("%Y-%m-%d"), dtype="U10"), n_train=entry["n_train"])


# Run every graph_type for one already-loaded window_length: fit 2-D MDS on each cosine-distance matrix, save the geometry file and return the results rows.
def _run_window_length(window_length, train, test, dates, n_train, graph_types):
    rows, failed = [], []

    for graph_type in graph_types:
        try:
            entry = _graph2vec_entry(train, test, dates, n_train, window_length, graph_type)
        except Exception as e:
            print(f"  window_length={window_length} graph_type={graph_type} FAILED: {config.GEOMETRY_INDEX}: {e!r} -- skipped")
            failed.append(graph_type)
            continue

        coords, stress = geometry.fit_mds(entry["distance"])
        _save_geometry(entry, coords, stress, window_length, graph_type)
        rows.append(_summary_row(entry, stress, window_length, graph_type))

    return rows, failed


# Run the embedding-geometry analysis for every window length and construction, merging each window length's rows into outputs/embedding_geometry_results.csv as it goes; returns True only if every combination succeeded.
def main(window_lengths=None, graph_types=None):
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    window_lengths = config.WINDOW_LENGTHS if window_lengths is None else window_lengths
    graph_types = config.GRAPH_TYPES if graph_types is None else graph_types
    results_path = config.OUTPUTS_DIR / "embedding_geometry_results.csv"

    failures = []

    for window_length in window_lengths:
        train, test, dates, n_train = _load_split(window_length)
        rows, failed = _run_window_length(window_length, train, test, dates, n_train, graph_types)
        del train, test

        if rows:
            merge_grid_results(results_path, rows)
            print(f"merged {len(rows)} rows into {results_path}")

        failures += [f"{window_length}d/{graph_type}" for graph_type in failed]

    if failures:
        print(f"{len(failures)} window length/construction combinations failed and keep their previous outputs, if any: {failures}")

    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embedding-geometry pipeline: S&P 500 / Graph2Vec / cosine / 2D MDS, swept over window lengths and network constructions.")
    parser.add_argument("--window-lengths", nargs="+", type=int, default=config.WINDOW_LENGTHS, choices=config.WINDOW_LENGTHS,
                         help="Window lengths to run (default: all three).")
    parser.add_argument("--graph-types", nargs="+", default=config.GRAPH_TYPES, choices=config.GRAPH_TYPES,
                         help="Graph constructions to run (default: all three).")
    args = parser.parse_args()
    raise SystemExit(0 if main(args.window_lengths, args.graph_types) else 1)
