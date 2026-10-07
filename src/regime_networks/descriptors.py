import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh

from regime_networks.networks import mu_offdiag, sigma_offdiag

SPECTRAL_FEATURE_COLUMNS = [
    "mu", "sigma_offdiag",
    "lambda_2_share", "lambda_3_share", "lambda_4_share",
    "n_above_mp", "spectral_entropy_bulk", "pr_corr_v1", "pr_corr_v2",
]


# Largest connected component of G, with node order following stock_names.
def largest_connected_component(G, stock_names):
    lcc_nodes = max(nx.connected_components(G), key=len)
    node_order = [s for s in stock_names if s in lcc_nodes]
    return G.subgraph(node_order), node_order


# Unit-norm, positive leading eigenvector of an adjacency matrix, from a fixed ARPACK start vector so the result does not depend on earlier calls.
def leading_eigenvector(adjacency):
    _, eigenvectors = eigsh(adjacency, k=1, which="LA", v0=np.linspace(1.0, 2.0, adjacency.shape[0]))
    v1 = eigenvectors[:, 0]

    if v1.sum() < 0:
        v1 = -v1

    return v1 / np.linalg.norm(v1)


# Eigenvector centrality of one window's graph on its largest connected component, with its participation ratio, its most central stock and the component's share of the stocks.
def window_centrality(G, stock_names):
    lcc_subgraph, node_order = largest_connected_component(G, stock_names)
    v1 = leading_eigenvector(nx.to_numpy_array(lcc_subgraph, nodelist=node_order, weight=None))
    return {"v1": v1, "pr": float(1.0 / (v1 ** 4).sum()), "argmax_id": node_order[int(np.argmax(v1))], "lcc_frac": len(node_order) / len(stock_names)}


# Participation ratio of a vector (unit-L2-normalised first), mirroring the eigenvector-centrality PR on a different object.
def participation_ratio(v):
    v = v / np.linalg.norm(v)
    return float(1.0 / (v ** 4).sum())


# Spectral and random-matrix features of one window's correlation matrix, with the Marchenko-Pastur edge scaled by the variance the market mode leaves, 1 - lambda_1/N.
def window_spectral(C, stock_names, window_length):
    N = len(stock_names)
    T = window_length
    q = T / N

    lambdas, V = np.linalg.eigh(C)
    order = np.argsort(lambdas)[::-1]
    lambdas, V = lambdas[order], V[:, order]
    mp_edge = (1 - lambdas[0] / N) * (1 + np.sqrt(N / T)) ** 2

    n_nonzero = min(N, T - 1)
    n_zero_eigenvalues = N - n_nonzero
    lambdas_nonzero = lambdas[:n_nonzero]

    bulk = lambdas_nonzero[lambdas_nonzero < mp_edge]

    if bulk.size > 1:
        p = bulk / bulk.sum()
        log_p = np.zeros_like(p)
        np.log(p, where=p > 0, out=log_p)
        spectral_entropy_bulk = float(-(p * log_p).sum() / np.log(bulk.size))
    else:
        spectral_entropy_bulk = float("nan")

    return {
        "mu": mu_offdiag(C),
        "sigma_offdiag": sigma_offdiag(C),
        "lambda_2_share": float(lambdas[1] / N),
        "lambda_3_share": float(lambdas[2] / N),
        "lambda_4_share": float(lambdas[3] / N),
        "n_above_mp": int((lambdas > mp_edge).sum()),
        "spectral_entropy_bulk": spectral_entropy_bulk,
        "pr_corr_v1": participation_ratio(V[:, 0]),
        "pr_corr_v2": participation_ratio(V[:, 1]),
        "n_zero_eigenvalues": int(n_zero_eigenvalues),
        "q": float(q),
        "mp_edge": float(mp_edge),
    }
