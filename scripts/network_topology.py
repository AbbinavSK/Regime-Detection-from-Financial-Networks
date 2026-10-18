import argparse

import numpy as np
import pandas as pd
import networkx as nx

from regime_networks import config
from regime_networks.grid import labels_path
from regime_networks.networks import load_networks

SPLITS = ["train", "test", "all"]
TOPOLOGY_STATISTICS = ["density", "avg_clustering", "degree_assortativity", "sector_modularity"]
CELL_KEYS = ["index_code", "window_length"]


# Density, average clustering (zero for a node of degree below two, averaged over every node), degree assortativity (Newman's: the Pearson correlation of the degrees at the two ends of an edge, in exact integer arithmetic) and modularity of the sector partition, of one window's unweighted adjacency matrix with nodes in stock order; assortativity is undefined without edges or when every degree is equal, and modularity without edges.
def window_topology_stats(A, sector_codes):
    n, degree = len(A), A.sum(axis=1)
    m = degree.sum() / 2
    A2 = A @ A
    node_triangles = (A2 * A).sum(axis=1) / 2
    pairs = degree * (degree - 1) / 2
    clustering = np.divide(node_triangles, pairs, out=np.zeros(n), where=pairs > 0)
    same_sector = sector_codes[:, None] == sector_codes[None, :]
    within_edges = (A * same_sector).sum() / 2
    sector_degree = np.bincount(sector_codes, weights=degree)
    modularity = within_edges / m - np.sum((sector_degree / (2 * m)) ** 2) if m > 0 else float("nan")
    d, A_int = degree.astype(np.int64), A.astype(np.int64)
    two_m, s2, s3 = int(d.sum()), int(d @ d), int(d @ (d * d))
    denominator = two_m * s3 - s2 * s2
    assortativity = (two_m * int(d @ (A_int @ d)) - s2 * s2) / denominator if denominator else float("nan")
    return {"density": 2 * m / (n * (n - 1)), "avg_clustering": float(clustering.mean()), "degree_assortativity": float(assortativity), "sector_modularity": float(modularity)}


# Split and regime name of every window of one construction, from the sweep labels of the headline descriptor built on that construction (its own model's regimes, headline method), checked to cover exactly the network windows.
def construction_regimes(index_code, window_length, graph_type, window_dates):
    path = labels_path(index_code, window_length, graph_type, config.HEADLINE_DESCRIPTOR)
    labels = pd.read_csv(path, parse_dates=["window_date"])

    if len(labels) != len(window_dates) or (labels["window_date"].to_numpy() != pd.DatetimeIndex(window_dates).to_numpy()).any():
        raise ValueError(f"{path.name} does not cover the network data's windows -- the labels are stale; re-run scripts/run_sweeps.py --descriptor {config.HEADLINE_DESCRIPTOR}")

    return labels["split"].to_numpy(), np.array(config.REGIME_NAMES)[labels[f"{config.HEADLINE_METHOD}_regime"].to_numpy()]


# Every window's topology statistics under each construction of one market and window length, each window in the regime its construction's own model detected.
def cell_windows(index_code, window_length):
    data = load_networks(index_code, window_length)
    stock_names = data["stock_names"]
    sector_codes = np.unique([s.split("_")[0] for s in stock_names], return_inverse=True)[1]
    window_dates = pd.DatetimeIndex(pd.to_datetime(data["window_dates"]))
    stats = {graph_type: [window_topology_stats(nx.to_numpy_array(G, nodelist=stock_names, weight=None), sector_codes) for G in data[config.GRAPH_KEYS[graph_type]]] for graph_type in config.GRAPH_TYPES}
    rows = []

    for graph_type in config.GRAPH_TYPES:
        splits, regimes = construction_regimes(index_code, window_length, graph_type, window_dates)

        for i, window_stats in enumerate(stats[graph_type]):
            rows.append({"index_code": index_code, "window_length": window_length, "graph_type": graph_type, "window": i, "window_date": window_dates[i], "split": splits[i], "regime": regimes[i],
                         "mu": data["mu_values"][i], **window_stats})

    nesting = None

    if (index_code, window_length) == (config.TOPOLOGY_INDEX, config.TOPOLOGY_WINDOW_LENGTH):
        edge_sets = {gt: [{frozenset(e) for e in G.edges()} for G in data[config.GRAPH_KEYS[gt]]] for gt in config.GRAPH_TYPES}
        nesting = pd.DataFrame([{"window_date": window_dates[i], "threshold_not_in_mst": len(t - m), "threshold_not_in_tmfg": len(t - f), "mst_not_in_tmfg": len(m - f)}
                                for i, (t, m, f) in enumerate(zip(edge_sets["threshold"], edge_sets["mst"], edge_sets["tmfg"]))])

    return pd.DataFrame(rows), nesting


# Count and median of each topology statistic for every market, window length, construction, split (train, test and all windows) and regime; a count excludes windows where the statistic is undefined.
def regime_summary(windows):
    pooled = pd.concat([windows, windows.assign(split="all")], ignore_index=True)
    values = pooled.melt(id_vars=[*CELL_KEYS, "graph_type", "split", "regime"], value_vars=TOPOLOGY_STATISTICS, var_name="statistic")
    summary = values.groupby([*CELL_KEYS, "graph_type", "split", "regime", "statistic"])["value"].agg(n_windows="count", median="median").reset_index()
    position = [(config.INDEX_CODES.index(i), config.WINDOW_LENGTHS.index(t), config.GRAPH_TYPES.index(g), SPLITS.index(s), config.REGIME_NAMES.index(r), TOPOLOGY_STATISTICS.index(k))
                for i, t, g, s, r, k in zip(summary["index_code"], summary["window_length"], summary["graph_type"], summary["split"], summary["regime"], summary["statistic"])]
    return summary.iloc[sorted(range(len(summary)), key=position.__getitem__)].reset_index(drop=True)


# For each construction and regime of the drawn market and window length, the window whose mu is nearest the median mu of all that regime's windows (train and test), the regime's example in the layout figure.
def representative_windows(windows):
    cell = windows[(windows["index_code"] == config.TOPOLOGY_INDEX) & (windows["window_length"] == config.TOPOLOGY_WINDOW_LENGTH)]
    rows = []

    for graph_type in config.GRAPH_TYPES:
        for regime in config.REGIME_NAMES:
            part = cell[(cell["graph_type"] == graph_type) & (cell["regime"] == regime)]

            if part.empty:
                continue

            median_mu = part["mu"].median()
            chosen = part.loc[(part["mu"] - median_mu).abs().idxmin()]
            rows.append({"graph_type": graph_type, "regime": regime, "window": int(chosen["window"]), "window_date": chosen["window_date"], "split": chosen["split"], "mu": chosen["mu"]})

    return pd.DataFrame(rows)


# Compute every window's topology under each construction for every market and window length, grouped by each construction's own detected regimes, and save the windows, their summaries, the drawn windows and the drawn cell's edge nesting to outputs/network_topology/.
def main():
    folder = config.OUTPUTS_DIR / "network_topology"
    folder.mkdir(parents=True, exist_ok=True)
    cells, nesting = [], None

    for index_code in config.INDEX_CODES:
        for window_length in config.WINDOW_LENGTHS:
            windows, cell_nesting = cell_windows(index_code, window_length)
            cells.append(windows)
            nesting = cell_nesting if cell_nesting is not None else nesting
            occupancy = windows.groupby(["graph_type", "split"], sort=False)["regime"].value_counts().unstack(fill_value=0).reindex(columns=config.REGIME_NAMES, fill_value=0)
            print(f"{index_code} T={window_length}: {windows['window'].nunique()} windows per construction; windows per regime:\n{occupancy.to_string()}", flush=True)

    windows = pd.concat(cells, ignore_index=True)
    windows.to_csv(folder / "window_topology.csv", index=False, lineterminator="\n")
    regime_summary(windows).to_csv(folder / "regime_summary.csv", index=False, lineterminator="\n")
    representative_windows(windows).to_csv(folder / "representative_windows.csv", index=False, lineterminator="\n")

    if nesting is not None:
        nesting.to_csv(folder / "edge_nesting.csv", index=False, lineterminator="\n")
        print(f"{config.TOPOLOGY_INDEX} T={config.TOPOLOGY_WINDOW_LENGTH}: MST-threshold edges outside TMFG-threshold in {(nesting['mst_not_in_tmfg'] > 0).mean():.1%} of windows")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Compare the network topology of each construction's windows across the regimes its own model detected, for every market and window length, into outputs/network_topology/.").parse_args()
    main()
