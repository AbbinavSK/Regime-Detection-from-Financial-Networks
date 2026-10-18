import argparse

import numpy as np
import pandas as pd

from regime_networks import config
from regime_networks.networks import load_networks
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


# Save everything the geometry figures need for one (window_length, graph_type): distances, MDS coordinates and stress, regime labels, edge counts, OOV rates, window dates and the train size.
def _save_geometry(entry, coords, stress, window_length, graph_type):
    path = geometry.geometry_path(config.GEOMETRY_INDEX, window_length, graph_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, distance=entry["distance"], coords=coords, stress=stress, regime_labels=entry["regime_labels"],
                        edge_counts=np.array(entry["edge_counts"]), oov=np.array(entry["oov"]), dates=np.array(pd.to_datetime(entry["dates"]).strftime("%Y-%m-%d"), dtype="U10"), n_train=entry["n_train"])


# Run every graph_type for one already-loaded window_length: fit 2-D MDS on each cosine-distance matrix and save the geometry file; returns the constructions that failed.
def _run_window_length(window_length, train, test, dates, n_train, graph_types):
    failed = []

    for graph_type in graph_types:
        try:
            entry = _graph2vec_entry(train, test, dates, n_train, window_length, graph_type)
        except Exception as e:
            print(f"  window_length={window_length} graph_type={graph_type} FAILED: {config.GEOMETRY_INDEX}: {e!r} -- skipped")
            failed.append(graph_type)
            continue

        coords, stress = geometry.fit_mds(entry["distance"])
        _save_geometry(entry, coords, stress, window_length, graph_type)
        print(f"  {config.GEOMETRY_INDEX} T={window_length} {graph_type}: MDS stress {stress:.3f}")

    return failed


# Run the embedding-geometry analysis for every window length and construction into outputs/geometry/; returns True only if every combination succeeded.
def main(window_lengths=None, graph_types=None):
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    window_lengths = config.WINDOW_LENGTHS if window_lengths is None else window_lengths
    graph_types = config.GRAPH_TYPES if graph_types is None else graph_types
    failures = []

    for window_length in window_lengths:
        train, test, dates, n_train = _load_split(window_length)
        failed = _run_window_length(window_length, train, test, dates, n_train, graph_types)
        del train, test
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
