import heapq
import pickle
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree

from regime_networks import config


# Mean of the off-diagonal (upper-triangle, k=1) entries of a correlation matrix.
def mu_offdiag(C):
    return float(C[np.triu_indices_from(C, k=1)].mean())


# Standard deviation of the off-diagonal (upper-triangle, k=1) entries of a correlation matrix.
def sigma_offdiag(C):
    return float(C[np.triu_indices_from(C, k=1)].std())


# Convert a correlation matrix to the standard correlation-distance matrix.
def correlation_distance(C):
    return np.sqrt(2 * (1 - C))


# Row/column index pairs (i<j) whose correlation is at or above the threshold, found by a vectorised upper-triangle scan.
def _above_threshold_pairs(C, threshold):
    i_idx, j_idx = np.triu_indices_from(C, k=1)
    above = C[i_idx, j_idx] >= threshold
    return i_idx[above], j_idx[above]


# Threshold graph, an edge for every pair whose correlation is at or above the cutoff.
def threshold_graph(C, D, stock_names, threshold):
    G = nx.Graph()
    G.add_nodes_from(stock_names)
    i_idx, j_idx = _above_threshold_pairs(C, threshold)

    for i, j in zip(i_idx, j_idx):
        G.add_edge(stock_names[i], stock_names[j], weight=float(D[i, j]), edge_type="threshold")

    return G


# Minimum spanning tree of the distance matrix, as an nx.Graph with distance weights; scipy reads a zero distance as a missing edge, so a pair with correlation 1 (identical returns in the window) would silently give a spanning tree that is not minimum, and raises instead.
def mst_graph(D, stock_names):
    if (D[np.triu_indices_from(D, k=1)] == 0).any():
        raise ValueError("two stocks have correlation distance zero (identical returns in the window), which scipy's minimum_spanning_tree reads as a missing edge")

    mst_coo = minimum_spanning_tree(D).tocoo()
    G = nx.Graph()
    G.add_nodes_from(stock_names)

    for u, v, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        G.add_edge(stock_names[u], stock_names[v], weight=float(w), edge_type="mst")

    return G


# MST backbone augmented with every correlation edge above the threshold.
def mst_threshold_graph(C, D, stock_names, threshold):
    G = mst_graph(D, stock_names)
    i_idx, j_idx = _above_threshold_pairs(C, threshold)

    for i, j in zip(i_idx, j_idx):
        if not G.has_edge(stock_names[i], stock_names[j]):
            G.add_edge(stock_names[i], stock_names[j], weight=float(D[i, j]), edge_type="threshold")

    return G


# Triangulated Maximally Filtered Graph (Massara/Aste).
def tmfg_graph(C, D, stock_names):
    n = len(stock_names)
    W = C.copy()
    np.fill_diagonal(W, 0.0)

    v_score = (W * (W > W.mean())).sum(axis=1)
    seed = list(np.argsort(v_score)[::-1][:4])

    placed = np.zeros(n, dtype=bool)
    placed[seed] = True

    triangles = [
        [seed[0], seed[1], seed[2]],
        [seed[0], seed[1], seed[3]],
        [seed[0], seed[2], seed[3]],
        [seed[1], seed[2], seed[3]],
    ]

    G = nx.Graph()
    G.add_nodes_from(stock_names)

    # Add the TMFG edge between stocks a and b (by index) unless it is already present.
    def add_edge(a, b):
        if not G.has_edge(stock_names[a], stock_names[b]):
            G.add_edge(stock_names[a], stock_names[b], weight=float(D[a, b]), edge_type="tmfg")

    for a in range(4):
        for b in range(a + 1, 4):
            add_edge(seed[a], seed[b])

    # Best unplaced stock to insert into triangle tri and the total correlation it would add.
    def best_gain(tri):
        gains = W[:, tri[0]] + W[:, tri[1]] + W[:, tri[2]]
        gains = np.where(placed, -np.inf, gains)
        k = int(np.argmax(gains))
        return float(gains[k]), k

    pq = []

    for ti, tri in enumerate(triangles):
        gain, vertex = best_gain(tri)
        heapq.heappush(pq, (-gain, vertex, ti))

    inserted = 0

    while inserted < n - 4:
        _, new_vertex, triangle_index = heapq.heappop(pq)

        if placed[new_vertex]:
            gain, vertex = best_gain(triangles[triangle_index])
            heapq.heappush(pq, (-gain, vertex, triangle_index))
            continue

        triangle = triangles[triangle_index]

        for vertex in triangle:
            add_edge(new_vertex, vertex)

        placed[new_vertex] = True
        inserted += 1

        triangles[triangle_index] = [triangle[0], triangle[1], new_vertex]
        triangles.append([triangle[0], triangle[2], new_vertex])
        triangles.append([triangle[1], triangle[2], new_vertex])

        updated_triangles = (triangle_index, len(triangles) - 2, len(triangles) - 1)

        for ti in updated_triangles:
            gain, vertex = best_gain(triangles[ti])
            heapq.heappush(pq, (-gain, vertex, ti))

    return G


# TMFG backbone unioned with all above-threshold edges (additive; floors at 3(n-2)).
def tmfg_threshold_graph(C, D, stock_names, threshold):
    G = tmfg_graph(C, D, stock_names)
    a_idx, b_idx = _above_threshold_pairs(C, threshold)

    for a, b in zip(a_idx, b_idx):
        if not G.has_edge(stock_names[a], stock_names[b]):
            G.add_edge(stock_names[a], stock_names[b], weight=float(D[a, b]), edge_type="threshold")

    return G


# Build every window's networks at CORRELATION_THRESHOLD: clip the correlation matrix, compute mu and the Mantegna distance, then the threshold, MST-threshold and TMFG-threshold graphs.
def build_networks(correlation_matrices, stock_names, window_length, window_dates, index_code):
    threshold = config.CORRELATION_THRESHOLD
    clipped_matrices = []
    mu_values = []
    threshold_graphs = []
    mst_threshold_graphs = []
    tmfg_threshold_graphs = []

    for C_raw in correlation_matrices:
        C = np.clip(C_raw, -1.0, 1.0)
        mu_values.append(mu_offdiag(C))
        D = correlation_distance(C)

        clipped_matrices.append(C)
        threshold_graphs.append(threshold_graph(C, D, stock_names, threshold))
        mst_threshold_graphs.append(mst_threshold_graph(C, D, stock_names, threshold))
        tmfg_threshold_graphs.append(tmfg_threshold_graph(C, D, stock_names, threshold))

    return {
        "index_code": index_code,
        "window_length": window_length,
        "threshold": threshold,
        "stock_names": stock_names,
        "window_dates": window_dates,
        "correlation_matrices": clipped_matrices,
        "mu_values": mu_values,
        "threshold_graphs": threshold_graphs,
        "mst_threshold_graphs": mst_threshold_graphs,
        "tmfg_threshold_graphs": tmfg_threshold_graphs,
    }


# Path of one index's pickled network data for one window length, in directory (outputs/ by default).
def networks_path(index_code, window_length, directory=None):
    return Path(config.OUTPUTS_DIR if directory is None else directory) / f"{index_code}_network_data_{window_length}d.pkl"


# Pickle one index/window-length's network data.
def save_networks(data, directory=None):
    path = networks_path(data["index_code"], data["window_length"], directory)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f)


# Load one index/window-length's pickled network data.
def load_networks(index_code, window_length, directory=None):
    with open(networks_path(index_code, window_length, directory), "rb") as f:
        return pickle.load(f)


# One index/window-length's network data through a one-entry cache (the dict passed in), reloaded only when the index or window length changes, so consecutive grid points of the same index and window length share one read of the pickle.
def load_networks_cached(cache, index_code, window_length):
    if (index_code, window_length) not in cache:
        cache.clear()
        cache[(index_code, window_length)] = load_networks(index_code, window_length)

    return cache[(index_code, window_length)]
