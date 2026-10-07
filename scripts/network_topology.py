import argparse
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import normalized_mutual_info_score

from regime_networks import config
from regime_networks.grid import labels_path
from regime_networks.networks import load_networks

SPLITS = ["train", "test", "all"]
TOPOLOGY_STATISTICS = ["density", "n_components", "mean_degree", "avg_clustering", "degree_assortativity", "avg_shortest_path_lcc", "diameter_lcc", "largest_clique", "n_maximal_cliques", "nmi_sector"]


# Normalised mutual information between the sector labels and the greedy-modularity communities, which partition every node (an isolated node is its own community).
def nmi_vs_sector(G, stock_names, sector_of):
    node_to_comm = {node: cid for cid, comm in enumerate(nx.algorithms.community.greedy_modularity_communities(G)) for node in comm}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return normalized_mutual_info_score([sector_of[s] for s in stock_names], [node_to_comm[s] for s in stock_names])


# Largest maximal-clique size and number of maximal cliques, enumerating at most CLIQUE_ENUM_CAP cliques on an integer-relabelled copy so the capped search order (and result) is the same in every process.
def clique_stats(G):
    max_size, count = 0, 0

    for clique in nx.find_cliques(nx.convert_node_labels_to_integers(G)):
        count += 1
        max_size = max(max_size, len(clique))

        if count >= config.CLIQUE_ENUM_CAP:
            return max_size, count, True

    return max_size, count, False


# Mean shortest-path length and diameter, in edges, of a graph's largest connected component (zero for a single node), from breadth-first searches on its adjacency matrix.
def lcc_path_stats(G):
    lcc = G.subgraph(max(nx.connected_components(G), key=len))

    if lcc.number_of_nodes() < 2:
        return 0.0, 0

    distances = shortest_path(nx.to_scipy_sparse_array(lcc, weight=None), unweighted=True, directed=False)
    off_diagonal = distances[~np.eye(len(distances), dtype=bool)]
    return float(off_diagonal.mean()), int(off_diagonal.max())


# Every topology statistic of one window's graph; degree assortativity is undefined (NaN) without edges or when every node has the same degree.
def window_topology_stats(G, stock_names, sector_of):
    path_mean, diameter = lcc_path_stats(G)
    largest, n_cliques, capped = clique_stats(G)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        assortativity = nx.degree_assortativity_coefficient(G) if G.number_of_edges() else float("nan")

    return {"density": nx.density(G), "n_components": nx.number_connected_components(G), "mean_degree": 2 * G.number_of_edges() / len(stock_names), "avg_clustering": nx.average_clustering(G),
            "degree_assortativity": float(assortativity), "avg_shortest_path_lcc": path_mean, "diameter_lcc": diameter, "largest_clique": largest, "n_maximal_cliques": n_cliques,
            "clique_enumeration_capped": capped, "nmi_sector": nmi_vs_sector(G, stock_names, sector_of)}


# Split and regime name of every window of one construction, from the sweep labels of the headline descriptor built on that construction (its own model's regimes, headline method), checked to cover exactly the network windows.
def construction_regimes(graph_type, window_dates):
    path = labels_path(config.TOPOLOGY_INDEX, config.TOPOLOGY_WINDOW_LENGTH, graph_type, config.HEADLINE_DESCRIPTOR)
    labels = pd.read_csv(path, parse_dates=["window_date"])

    if len(labels) != len(window_dates) or (labels["window_date"].to_numpy() != pd.DatetimeIndex(window_dates).to_numpy()).any():
        raise ValueError(f"{path.name} does not cover the network data's windows -- the labels are stale; re-run scripts/run_sweeps.py --descriptor {config.HEADLINE_DESCRIPTOR}")

    return labels["split"].to_numpy(), np.array(config.REGIME_NAMES)[labels[f"{config.HEADLINE_METHOD}_regime"].to_numpy()]


# Count, mean, median and interquartile range of each topology statistic for every construction, split (train, test and all windows) and regime; a count excludes windows where the statistic is undefined.
def regime_summary(windows):
    pooled = pd.concat([windows, windows.assign(split="all")], ignore_index=True)
    values = pooled.melt(id_vars=["graph_type", "split", "regime"], value_vars=TOPOLOGY_STATISTICS, var_name="statistic")
    summary = values.groupby(["graph_type", "split", "regime", "statistic"])["value"].agg(n_windows="count", mean="mean", median="median", q25=lambda v: v.quantile(0.25), q75=lambda v: v.quantile(0.75)).reset_index()
    position = [(config.GRAPH_TYPES.index(g), SPLITS.index(s), config.REGIME_NAMES.index(r), TOPOLOGY_STATISTICS.index(t)) for g, s, r, t in zip(summary["graph_type"], summary["split"], summary["regime"], summary["statistic"])]
    return summary.iloc[sorted(range(len(summary)), key=position.__getitem__)].reset_index(drop=True)


# For each construction and regime, the window whose mu is nearest the median mu of all that regime's windows (train and test), the regime's example in the layout figure.
def representative_windows(windows):
    rows = []

    for graph_type in config.GRAPH_TYPES:
        for regime in config.REGIME_NAMES:
            part = windows[(windows["graph_type"] == graph_type) & (windows["regime"] == regime)]

            if part.empty:
                continue

            median_mu = part["mu"].median()
            chosen = part.loc[(part["mu"] - median_mu).abs().idxmin()]
            rows.append({"graph_type": graph_type, "regime": regime, "window": int(chosen["window"]), "window_date": chosen["window_date"], "split": chosen["split"], "mu": chosen["mu"],
                         "regime_median_mu": median_mu, "regime_windows": len(part)})

    return pd.DataFrame(rows)


# Node degrees pooled over every window of each construction and regime, as counts per degree.
def degree_counts(data, windows):
    counts = {}

    for graph_type, window, regime in zip(windows["graph_type"], windows["window"], windows["regime"]):
        for _, degree in data[config.GRAPH_KEYS[graph_type]][window].degree():
            counts[(graph_type, regime, degree)] = counts.get((graph_type, regime, degree), 0) + 1

    keys = sorted(counts, key=lambda key: (config.GRAPH_TYPES.index(key[0]), config.REGIME_NAMES.index(key[1]), key[2]))
    return pd.DataFrame([(*key, counts[key]) for key in keys], columns=["graph_type", "regime", "degree", "count"])


# Compute every window's topology under each construction, grouped by that construction's own detected regimes, and save the summaries to outputs/network_topology/.
def main():
    data = load_networks(config.TOPOLOGY_INDEX, config.TOPOLOGY_WINDOW_LENGTH)
    stock_names = data["stock_names"]
    sector_of = {s: s.split("_")[0] for s in stock_names}
    window_dates = pd.DatetimeIndex(pd.to_datetime(data["window_dates"]))
    folder = config.OUTPUTS_DIR / "network_topology"
    folder.mkdir(parents=True, exist_ok=True)

    rows = []

    for graph_type in config.GRAPH_TYPES:
        splits, regimes = construction_regimes(graph_type, window_dates)

        for i, G in enumerate(data[config.GRAPH_KEYS[graph_type]]):
            rows.append({"graph_type": graph_type, "window": i, "window_date": window_dates[i], "split": splits[i], "regime": regimes[i], "mu": data["mu_values"][i], **window_topology_stats(G, stock_names, sector_of)})

    windows = pd.DataFrame(rows)
    windows.to_csv(folder / "window_topology.csv", index=False, lineterminator="\n")
    regime_summary(windows).to_csv(folder / "regime_summary.csv", index=False, lineterminator="\n")
    representative_windows(windows).to_csv(folder / "representative_windows.csv", index=False, lineterminator="\n")
    degree_counts(data, windows).to_csv(folder / "degree_counts.csv", index=False, lineterminator="\n")

    edge_sets = {gt: [{frozenset(e) for e in G.edges()} for G in data[config.GRAPH_KEYS[gt]]] for gt in config.GRAPH_TYPES}
    nesting = pd.DataFrame([{"window_date": window_dates[i], "threshold_not_in_mst": len(t - m), "threshold_not_in_tmfg": len(t - f), "mst_not_in_tmfg": len(m - f), "n_mst_threshold_edges": len(m)}
                            for i, (t, m, f) in enumerate(zip(edge_sets["threshold"], edge_sets["mst"], edge_sets["tmfg"]))])
    nesting.to_csv(folder / "edge_nesting.csv", index=False, lineterminator="\n")
    occupancy = windows.groupby(["graph_type", "split"], sort=False)["regime"].value_counts().unstack(fill_value=0).reindex(columns=config.REGIME_NAMES, fill_value=0)
    print(f"{config.TOPOLOGY_INDEX} T={config.TOPOLOGY_WINDOW_LENGTH}: {len(window_dates)} windows per construction; windows per regime:\n{occupancy.to_string()}")
    print(f"MST-threshold edges outside TMFG-threshold in {(nesting['mst_not_in_tmfg'] > 0).mean():.1%} of windows")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Compare the network topology of each construction's windows across the regimes its own model detected, over train, test and all windows, into outputs/network_topology/.").parse_args()
    main()
