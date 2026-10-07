import numpy as np
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

from regime_networks import config


# Cluster embeddings into regimes with K-means, fit on train only.
def cluster_kmeans(X_train, X_test, k=config.N_REGIMES):
    model = KMeans(n_clusters=k, random_state=config.SEED, n_init=config.KMEANS_N_INIT).fit(X_train)
    return model.labels_, model.predict(X_test), model


# Decode each window from only the observations up to and including it (Viterbi on the history followed by each growing prefix), so no label depends on later observations and the first window is not forced into the fitted start state.
def _causal_decode(model, X, history=None):
    past = np.empty((0, X.shape[1])) if history is None else history
    return np.array([model.predict(np.vstack([past, X[: t + 1]]))[-1] for t in range(len(X))])


# Fit a k-state diagonal Gaussian HMM from n_seeds random restarts and keep the one with the highest train log-likelihood, skipping restarts whose train Viterbi path leaves a state unused when every state must be a regime.
def fit_best_hmm(X_train, k, n_seeds, require_all_states=False):
    best_model, best_ll, n_failed, n_unused = None, -np.inf, 0, 0

    for seed in range(n_seeds):
        m = GaussianHMM(n_components=k, covariance_type=config.HMM_COVARIANCE_TYPE, n_iter=config.HMM_N_ITER, tol=config.HMM_TOL, random_state=seed)

        try:
            m.fit(X_train)
            ll = m.score(X_train)
        except Exception:
            n_failed += 1
            continue

        if require_all_states and len(np.unique(m.predict(X_train))) < k:
            n_unused += 1
            continue

        if ll > best_ll:
            best_model, best_ll = m, ll

    if best_model is None:
        raise RuntimeError(f"no usable {k}-state HMM from {n_seeds} restarts: {n_failed} failed to fit (X may be degenerate, e.g. a collapsed embedding) and {n_unused} left a state unused on the training windows")

    return best_model


# Fit a Gaussian HMM on training embeddings and return full-sequence training labels (for ranking), causal training and test labels, and the model.
def cluster_hmm(X_train, X_test, k=config.N_REGIMES, n_seeds=config.HMM_N_SEEDS):
    best_model = fit_best_hmm(X_train, k, n_seeds, require_all_states=True)
    return best_model.predict(X_train), _causal_decode(best_model, X_train), _causal_decode(best_model, X_test, history=X_train), best_model


# Rank clusters by mean correlation to find the crisis regime; a cluster with no train windows has no mean to rank, so it raises instead of silently sorting last (as Crisis).
def fit_regime_order(labels, mu, k=config.N_REGIMES):
    labels = np.asarray(labels)
    mu = np.asarray(mu)

    empty = [c for c in range(k) if not np.any(labels == c)]

    if empty:
        raise ValueError(f"clusters {empty} have no train windows, so they cannot be ranked by mean mu -- the fitted model uses fewer than k={k} regimes on train")

    cluster_mean_mu = np.array([mu[labels == c].mean() for c in range(k)])
    order = np.argsort(cluster_mean_mu)
    rank_of_cluster = np.argsort(order)
    crisis_id = order[-1]

    return order, rank_of_cluster, crisis_id, cluster_mean_mu


# Relabel cluster IDs into rank order.
def apply_regime_order(labels, order):
    rank_of_cluster = np.argsort(order)
    return rank_of_cluster[np.asarray(labels)]


# Where the lowest AIC over MODEL_SELECTION_K_RANGE lies: "interior", or "lower edge"/"upper edge" at an end of the range, where the true minimum may lie outside it; "none" (best_k NaN) if every fit failed.
def aic_minimum_location(best_k):
    if np.isnan(best_k):
        return "none"

    return "lower edge" if best_k == min(config.MODEL_SELECTION_K_RANGE) else "upper edge" if best_k == max(config.MODEL_SELECTION_K_RANGE) else "interior"


# Compute an empirical transition matrix from a label sequence.
def empirical_transitions(labels, k, order):
    rank_of_cluster = np.argsort(order)
    ranks = rank_of_cluster[np.asarray(labels)]

    occupancy = np.array([(ranks == r).sum() for r in range(k)])
    counts = np.zeros((k, k))

    for a, b in zip(ranks[:-1], ranks[1:]):
        counts[a, b] += 1

    row_sums = counts.sum(axis=1)
    P = np.full((k, k), np.nan)
    nonzero = row_sums > 0
    P[nonzero] = counts[nonzero] / row_sums[nonzero, None]
    return P, occupancy


# Fit K-means and HMM on train, predict both, mu-rank each set of regimes from train only, and compute the HMM's expected dwell 1 / (1 - p_ii) per regime rank; train_causal_labels are the train labels to compare with test labels (identical to train_labels for K-means, whose labels are pointwise).
def fit_and_rank_regimes(X_train, X_test, train_mu, n_seeds=config.HMM_N_SEEDS):
    kmeans_train_labels, kmeans_test_labels, kmeans_model = cluster_kmeans(X_train, X_test)
    kmeans_order, kmeans_rank_of_cluster, kmeans_crisis_id, kmeans_mean_mu = fit_regime_order(kmeans_train_labels, train_mu)

    hmm_train_labels, hmm_train_causal_labels, hmm_test_labels, hmm_model = cluster_hmm(X_train, X_test, n_seeds=n_seeds)
    hmm_order, hmm_rank_of_cluster, hmm_crisis_id, hmm_mean_mu = fit_regime_order(hmm_train_labels, train_mu)
    hmm_dwell = 1.0 / (1.0 - np.diag(hmm_model.transmat_)[hmm_order])

    return {
        "kmeans": {"model": kmeans_model, "train_labels": kmeans_train_labels, "train_causal_labels": kmeans_train_labels, "test_labels": kmeans_test_labels,
                   "order": kmeans_order, "rank_of_cluster": kmeans_rank_of_cluster,
                   "crisis_id": kmeans_crisis_id, "mean_mu": kmeans_mean_mu},
        "hmm": {"model": hmm_model, "train_labels": hmm_train_labels, "train_causal_labels": hmm_train_causal_labels, "test_labels": hmm_test_labels,
                "order": hmm_order, "rank_of_cluster": hmm_rank_of_cluster,
                "crisis_id": hmm_crisis_id, "mean_mu": hmm_mean_mu, "dwell": hmm_dwell},
    }
