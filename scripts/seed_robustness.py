import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

from regime_networks import config
from regime_networks.networks import load_networks
from regime_networks.grid import headline_grid_point, labels_path
from regime_networks.graph2vec import Graph2Vec
from regime_networks.reduction import fit_reduction, transform_reduction
from regime_networks.regimes import cluster_kmeans, fit_regime_order, apply_regime_order
from regime_networks.validation import split_train_test, load_vix, validate_regime

SUMMARY_COLUMNS = ["agreement_test", "adjusted_rand_index", "crisis_jaccard", "test_crisis_share", "test_vix_f1"]


# K-means regime ranks of every window (train, then test) and the PCA dimensions kept, for one grid point's graphs embedded by Graph2Vec seeded with seed, every later step as in the sweep (scaler, PCA and K-means with SEED, clusters ranked by training mu).
def seed_regimes(train_graphs, test_graphs, train_mu, seed):
    g2v = Graph2Vec(seed=seed)
    train_raw, _ = g2v.fit(train_graphs)
    test_raw, _ = g2v.infer(test_graphs)
    X_train, scaler, pca = fit_reduction(train_raw)
    train_labels, test_labels, _ = cluster_kmeans(X_train, transform_reduction(test_raw, scaler, pca))
    order = fit_regime_order(train_labels, train_mu)
    return apply_regime_order(np.concatenate([train_labels, test_labels]), order), pca.n_components_


# Agreement of one seed's regime ranks with the reference ranks: the share of windows with the same regime (all and test), the adjusted Rand index of the two partitions, the Jaccard index of their Crisis windows, and the seed's test-period Crisis share.
def compare_regimes(ranks, reference, n_train):
    same = ranks == reference
    crisis, reference_crisis = ranks == config.N_REGIMES - 1, reference == config.N_REGIMES - 1
    return {"agreement_all": same.mean(), "agreement_test": same[n_train:].mean(), "adjusted_rand_index": adjusted_rand_score(reference, ranks),
            "crisis_jaccard": (crisis & reference_crisis).sum() / (crisis | reference_crisis).sum(), "test_crisis_share": crisis[n_train:].mean()}


# Refit one market's headline grid point with SEED and every robustness seed in parallel, refuse to continue unless SEED reproduces the saved K-means regimes, and compare every seed's regimes with them (VIX F1 of the test period for the S&P 500); one row per seed.
def market_rows(index_code, vix, n_jobs):
    grid_point = headline_grid_point(index_code)
    data = load_networks(index_code, grid_point[1])
    train, test = split_train_test(data)
    key = config.GRAPH_KEYS[grid_point[2]]
    seeds = [config.SEED, *config.SEED_ROBUSTNESS_SEEDS]
    print(f"{index_code}: refitting {labels_path(*grid_point).stem} with Graph2Vec seeds {seeds}", flush=True)
    fits = Parallel(n_jobs=n_jobs)(delayed(seed_regimes)(train[key], test[key], train["mu_values"], seed) for seed in seeds)
    reference = pd.read_csv(labels_path(*grid_point))["kmeans_regime"].to_numpy()

    if not np.array_equal(fits[0][0], reference):
        raise ValueError(f"{index_code}: Graph2Vec seeded with SEED={config.SEED} does not reproduce the saved K-means regimes of {labels_path(*grid_point).name} -- the labels or networks are stale; re-run scripts/run_sweeps.py --descriptor graph2vec")

    n_train = len(train["window_dates"])
    rows = []

    for seed, (ranks, pca_dims) in zip(seeds, fits):
        test_vix_f1 = validate_regime(ranks[n_train:], test["window_dates"], vix, config.VIX_CRISIS_THRESHOLD, config.N_REGIMES - 1)["point"][2] if index_code == config.VIX_INDEX else float("nan")
        rows.append({"index_code": index_code, "seed": seed, "reference": seed == config.SEED, "n_train_windows": n_train, "n_test_windows": len(ranks) - n_train, "pca_dims": pca_dims,
                     **compare_regimes(ranks, reference, n_train), "test_vix_f1": test_vix_f1})

    return rows


# Run the Doc2Vec seed robustness check for every market's headline grid point, save one row per market and seed to outputs/seed_robustness.csv, and print each market's range over the robustness seeds.
def main(n_jobs=None):
    if config.HEADLINE_DESCRIPTOR != "graph2vec":
        raise ValueError(f"the seed robustness check refits Graph2Vec, so it needs HEADLINE_DESCRIPTOR = 'graph2vec', not {config.HEADLINE_DESCRIPTOR!r}")

    if config.SEED in config.SEED_ROBUSTNESS_SEEDS:
        raise ValueError(f"SEED_ROBUSTNESS_SEEDS {config.SEED_ROBUSTNESS_SEEDS} must not contain SEED={config.SEED}, the reference")

    vix = load_vix()
    n_jobs = config.SEED_ROBUSTNESS_N_JOBS if n_jobs is None else n_jobs
    results = pd.DataFrame([row for index_code in config.INDEX_CODES for row in market_rows(index_code, vix, n_jobs)])
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.OUTPUTS_DIR / "seed_robustness.csv", index=False, lineterminator="\n")
    summary = results[~results["reference"]].groupby("index_code", sort=False)[SUMMARY_COLUMNS].agg(["min", "median", "max"])
    reference = results[results["reference"]].set_index("index_code")[["test_crisis_share", "test_vix_f1"]]
    print(f"\nreference (SEED={config.SEED}):\n{reference.round(3).to_string()}\n\nrange over seeds {list(config.SEED_ROBUSTNESS_SEEDS)}:\n{summary.round(3).to_string()}")
    print(f"\nwrote {config.OUTPUTS_DIR / 'seed_robustness.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refit every market's headline grid point with several Graph2Vec (Doc2Vec) seeds and compare its K-means regimes with the canonical ones into outputs/seed_robustness.csv.")
    parser.add_argument("--n-jobs", type=int, default=config.SEED_ROBUSTNESS_N_JOBS, help=f"Worker processes for the fits (default: {config.SEED_ROBUSTNESS_N_JOBS}); each fit is single-threaded and seeded, so results do not depend on it")
    main(parser.parse_args().n_jobs)
