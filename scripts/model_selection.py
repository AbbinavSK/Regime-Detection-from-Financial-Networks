import argparse

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from regime_networks import config
from regime_networks.networks import load_networks_cached
from regime_networks.grid import output_stem, parse_output_stem, descriptor_grid, bundle_path, spectral_features_path
from regime_networks.validation import split_train_test
from regime_networks.descriptors import SPECTRAL_FEATURE_COLUMNS, window_centrality
from regime_networks.reduction import transform_reduction
from regime_networks.regimes import fit_best_hmm


# The raw train descriptor matrix a grid point's models were fitted on: Graph2Vec's stored train vectors, the spectral features the sweep saved, or eigencentrality recomputed on the train networks (whose data is kept in the loaded dict for the next grid point of the same market and window length).
def _train_raw(bundle, grid_point, loaded):
    if grid_point["descriptor"] == "graph2vec":
        return bundle["g2v"].train_vectors

    n_train = len(bundle["kmeans_model"].labels_)

    if grid_point["descriptor"] == "spectral":
        return pd.read_pickle(spectral_features_path(grid_point["index_code"], grid_point["window_length"]))[SPECTRAL_FEATURE_COLUMNS].to_numpy()[:n_train]

    data = load_networks_cached(loaded, grid_point["index_code"], grid_point["window_length"])
    train, _ = split_train_test(data, cutoff=config.TRAIN_TEST_CUTOFF)
    return np.stack([window_centrality(G, data["stock_names"])["v1"] for G in train[config.GRAPH_KEYS[grid_point["graph_type"]]]])


# The reduced train embedding a grid point's models were fitted on, checked against the bundle's own K-means labels so a stale bundle or a changed descriptor fails loudly; loaded is the one-entry network cache shared across grid points.
def train_embedding(stem, loaded):
    grid_point = parse_output_stem(stem)
    bundle = joblib.load(bundle_path(**grid_point))
    X_train = transform_reduction(_train_raw(bundle, grid_point, loaded), bundle["scaler"], bundle["pca"])

    if not np.array_equal(bundle["kmeans_model"].predict(X_train), bundle["kmeans_model"].labels_):
        raise ValueError(f"{stem}: the rebuilt train embedding does not reproduce the bundle's K-means labels -- the bundle is stale; re-run scripts/run_sweeps.py")

    return X_train


# Model bundle stem of every grid point the sweep is configured to run, in sweep order.
def configured_stems():
    return [output_stem(index_code, window_length, graph_type, descriptor) for descriptor in config.DESCRIPTORS for index_code, window_length, graph_type in descriptor_grid(descriptor)]


# K-means inertia and best-of-restarts HMM AIC for one candidate number of regimes on one train embedding (AIC is NaN if every HMM restart fails).
def score_k(X_train, k):
    inertia = KMeans(n_clusters=k, random_state=config.SEED, n_init=config.KMEANS_N_INIT).fit(X_train).inertia_

    try:
        aic = fit_best_hmm(X_train, k, config.MODEL_SELECTION_HMM_SEEDS).aic(X_train)
    except RuntimeError:
        aic = float("nan")

    return inertia, aic


# K-means inertia and HMM AIC for every candidate number of regimes on the train embedding of every configured sweep grid point, fitted in parallel, saved to outputs/model_selection.csv.
def main(n_jobs=None):
    stems = configured_stems()
    missing = [stem for stem in stems if not bundle_path(**parse_output_stem(stem)).exists()]

    if missing:
        raise FileNotFoundError(f"{len(missing)} configured grid points have no model bundle (run scripts/run_sweeps.py first): {missing}")

    loaded = {}
    embeddings = {stem: train_embedding(stem, loaded) for stem in stems}
    loaded.clear()
    print(f"rebuilt the train embeddings of {len(stems)} grid points; scoring K in {list(config.MODEL_SELECTION_K_RANGE)}")
    jobs = [(stem, k) for stem in stems for k in config.MODEL_SELECTION_K_RANGE]
    scores = Parallel(n_jobs=config.MODEL_SELECTION_N_JOBS if n_jobs is None else n_jobs)(delayed(score_k)(embeddings[stem], k) for stem, k in jobs)
    results = pd.DataFrame([{"tag": stem, **parse_output_stem(stem), "n_train_windows": embeddings[stem].shape[0], "pca_dims": embeddings[stem].shape[1], "k": k, "kmeans_inertia": inertia, "hmm_aic": aic}
                            for (stem, k), (inertia, aic) in zip(jobs, scores)])
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.OUTPUTS_DIR / "model_selection.csv", index=False, lineterminator="\n")
    best = results.dropna(subset=["hmm_aic"]).loc[lambda df: df.groupby("tag")["hmm_aic"].idxmin()]
    print(f"lowest-AIC number of regimes across {len(best)} grid points: {best['k'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score K-means inertia and HMM AIC for each candidate number of regimes on every sweep grid point's train embedding into outputs/model_selection.csv.")
    parser.add_argument("--n-jobs", type=int, default=config.MODEL_SELECTION_N_JOBS, help=f"Worker processes for the fits (default: {config.MODEL_SELECTION_N_JOBS}); results agree across settings up to floating-point rounding")
    main(parser.parse_args().n_jobs)
