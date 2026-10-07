from pathlib import Path

import joblib
import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from regime_networks import config
from regime_networks.grid import grid_tag, bundle_path
from regime_networks.reduction import transform_reduction
from regime_networks.regimes import apply_regime_order


# Geometry file (distances, MDS coordinates, labels) of one grid point, written by the geometry step and read by its figures, in directory (outputs/geometry/ by default).
def geometry_path(index_code, window_length, graph_type, directory=None):
    return Path(config.OUTPUTS_DIR / "geometry" if directory is None else directory) / f"{grid_tag(index_code, window_length, graph_type)}.npz"


# Load one grid point's frozen Graph2Vec bundle and its train/test raw vectors and OOV rates exactly as the sweep produced them (re-inferring would not reproduce the sweep's regime labels).
def load_graph2vec_raw(index_code, window_length, graph_type, directory=None):
    path = bundle_path(index_code, window_length, graph_type, "graph2vec", directory)
    bundle = joblib.load(path)

    if not {"test_raw", "test_oov"} <= set(bundle):
        raise ValueError(f"{path.name} has no stored test embeddings -- it predates the fix that stopped re-inferring them. Re-run scripts/run_sweeps.py --descriptor graph2vec to refresh it.")

    g2v = bundle["g2v"]
    return bundle, g2v.train_vectors, bundle["test_raw"], g2v.train_oov_rate, bundle["test_oov"]


# L2-normalise each row of the raw Graph2Vec embedding matrix so distances measure shape, not magnitude.
def prepare_graph2vec(raw):
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


# Compute the n_windows x n_windows cosine distance matrix for a prepared vector matrix.
def distance_matrix(vectors):
    return pairwise_distances(vectors, metric="cosine")


# Return (min, median, max) off-diagonal distance from a square distance matrix.
def offdiag_distance_summary(D):
    mask = ~np.eye(D.shape[0], dtype=bool)
    values = D[mask]
    return float(values.min()), float(np.median(values)), float(values.max())


# Fit 2-D non-metric MDS on a precomputed distance matrix and return coordinates plus normalised Kruskal stress-1.
def fit_mds(D):
    model = MDS(n_components=2, metric=False, dissimilarity="precomputed",
                random_state=config.SEED, n_init=config.MDS_N_INIT, normalized_stress=True)
    coords = model.fit_transform(D)
    return coords, model.stress_


# Reproduce a Graph2Vec grid point's frozen K-means regime labels (train, then test); test_raw must be the bundle's own un-normalised vectors.
def bundle_regime_labels(bundle, test_raw, n_train):
    scaler, pca = bundle["scaler"], bundle["pca"]
    kmeans_model, kmeans_order = bundle["kmeans_model"], bundle["kmeans_order"]

    n_expected = scaler.n_features_in_

    if test_raw.shape[1] != n_expected:
        raise ValueError(
            f"test_raw has {test_raw.shape[1]} columns but the frozen bundle's scaler expects {n_expected} -- "
            "the bundle is stale relative to the current Graph2Vec implementation (e.g. its dimensions/"
            "wl_iterations hyperparameters changed since the bundle was written). Re-run scripts/run_sweeps.py --descriptor graph2vec "
            "to refresh it."
        )

    train_labels = kmeans_model.labels_

    if len(train_labels) != n_train:
        raise ValueError(
            f"frozen bundle's kmeans_model was fit on {len(train_labels)} train windows, but the live split has "
            f"{n_train} -- the bundle is stale relative to the current data (e.g. data_processing.py was "
            "re-run since the bundle was written). Re-run scripts/run_sweeps.py --descriptor graph2vec to refresh it."
        )

    X_test = transform_reduction(test_raw, scaler, pca)
    test_labels = kmeans_model.predict(X_test)

    all_labels = np.concatenate([train_labels, test_labels])
    return apply_regime_order(all_labels, kmeans_order)
