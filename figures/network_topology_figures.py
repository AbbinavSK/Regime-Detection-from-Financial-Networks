import argparse

import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from regime_networks import config
from regime_networks.networks import load_networks
from regime_networks.plotting import (set_publication_style, style_axis, save_to_subfolder, add_panel_label, FIGURE_WIDTH, PANEL_LETTERS, SMALL_FONT_SIZE, REGIME_PALETTE, NEUTRAL_COLOURS, SECTOR_NAMES,
                                      GRAPH_LABELS, MARKET_NAMES)

FOLDER = "network_topology"
REGIME_FIGURE_STATISTICS = {"density": r"Density $\rho$", "avg_clustering": "Average clustering", "degree_assortativity": "Degree assortativity", "sector_modularity": "Sector modularity"}
SPLIT_HATCHES = {"train": "", "test": "////"}
CENTREPIECE_REGIMES = ["Calm", "Crisis"]


# The drawn market and window length's rows of the topology windows.
def _drawn_cell(windows):
    return windows[(windows["index_code"] == config.TOPOLOGY_INDEX) & (windows["window_length"] == config.TOPOLOGY_WINDOW_LENGTH)]


# One Calm and one Crisis window (rows) under each construction (columns): the headline construction's representative window of each regime (mu nearest the regime's median mu), so every row shows one window three ways on one Kamada-Kawai layout of its MST-threshold graph (connected, so no isolated stock is placed arbitrarily), nodes coloured by sector.
def network_structure_figure(data, representatives):
    stock_names = data["stock_names"]
    sector_of = {s: s.split("_")[0] for s in stock_names}
    sector_names = SECTOR_NAMES.get(config.TOPOLOGY_INDEX, {})
    unique_sectors = sorted(set(sector_of.values()))
    sector_colour = dict(zip(unique_sectors, sns.color_palette("husl", len(unique_sectors))))
    node_colour = [sector_colour[sector_of[s]] for s in stock_names]
    chosen = representatives[representatives["graph_type"] == config.HEADLINE_GRAPH_TYPE].set_index("regime")
    fig, axes = plt.subplots(len(CENTREPIECE_REGIMES), len(config.GRAPH_TYPES), figsize=(FIGURE_WIDTH, 4.2), squeeze=False)
    letters = iter(PANEL_LETTERS)

    for row, regime in enumerate(CENTREPIECE_REGIMES):
        window = chosen.loc[regime]
        pos = nx.kamada_kawai_layout(data[config.GRAPH_KEYS["mst"]][int(window["window"])], weight="weight")

        for col, graph_type in enumerate(config.GRAPH_TYPES):
            ax = axes[row, col]
            G = data[config.GRAPH_KEYS[graph_type]][int(window["window"])]
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=NEUTRAL_COLOURS["reference"], width=0.15, alpha=0.35)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colour, node_size=2.5, linewidths=0)
            ax.axis("off")
            add_panel_label(ax, next(letters))
            ax.set_title(f"{GRAPH_LABELS[graph_type]}, {regime}")
            ax.text(0.5, -0.01, f"{pd.Timestamp(window['window_date']).date()} ({window['split']}), $\\mu$ = {window['mu']:.3f}, $\\rho$ = {nx.density(G):.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=SMALL_FONT_SIZE - 1)

    fig.legend(handles=[Patch(facecolor=sector_colour[s], label=sector_names.get(s, s)) for s in unique_sectors], loc="outside lower center", ncol=4)
    return fig


# Distribution of each regime-profile statistic by regime (box: interquartile range and median; whiskers: 1.5 IQR) in the drawn market and window length, training and test windows side by side, one row per statistic and one column per construction, each construction grouped by its own model's regimes.
def regime_comparison_figure(windows):
    cell = _drawn_cell(windows)
    fig, axes = plt.subplots(len(REGIME_FIGURE_STATISTICS), len(config.GRAPH_TYPES), figsize=(FIGURE_WIDTH, 1.45 * len(REGIME_FIGURE_STATISTICS) + 0.3), squeeze=False)

    for row, (statistic, label) in enumerate(REGIME_FIGURE_STATISTICS.items()):
        for col, graph_type in enumerate(config.GRAPH_TYPES):
            ax = axes[row][col]

            for rank, regime in enumerate(config.REGIME_NAMES):
                for offset, split in enumerate(SPLIT_HATCHES):
                    values = cell.loc[(cell["graph_type"] == graph_type) & (cell["regime"] == regime) & (cell["split"] == split), statistic].dropna().to_numpy()

                    if values.size:
                        box = ax.boxplot(values, positions=[3 * rank + offset], widths=0.8, patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 0.8},
                                         whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6}, boxprops={"linewidth": 0.6})
                        box["boxes"][0].set(facecolor=REGIME_PALETTE[rank], alpha=0.6, hatch=SPLIT_HATCHES[split])

            ax.set_xlim(-1, 3 * len(config.REGIME_NAMES) - 1)
            ax.set_xticks([3 * rank + 0.5 for rank in range(len(config.REGIME_NAMES))], config.REGIME_NAMES if row == len(REGIME_FIGURE_STATISTICS) - 1 else [])
            style_axis(ax, grid=True)

            if row == 0:
                ax.set_title(GRAPH_LABELS[graph_type])

            if col == 0:
                ax.set_ylabel(label)
                add_panel_label(ax, PANEL_LETTERS[row])

    fig.legend(handles=[Patch(facecolor="white", edgecolor="black", linewidth=0.6, hatch=hatch, label=f"{'Training' if split == 'train' else 'Test'} windows") for split, hatch in SPLIT_HATCHES.items()], loc="outside upper center", ncol=2)
    return fig


# Threshold-network density against mu for the (market, window length) panels of a layout (a list of rows), each window coloured by the threshold construction's own regime and each panel giving the Spearman rank correlation.
def density_mu_figure(windows, layout):
    threshold = windows[windows["graph_type"] == "threshold"]
    fig, axes = plt.subplots(len(layout), len(layout[0]), figsize=(FIGURE_WIDTH, min(2.05 * len(layout), 6.4) + 0.3), squeeze=False)

    for row, panels in enumerate(layout):
        for col, (index_code, window_length) in enumerate(panels):
            ax = axes[row][col]
            cell = threshold[(threshold["index_code"] == index_code) & (threshold["window_length"] == window_length)]

            for rank, regime in enumerate(config.REGIME_NAMES):
                part = cell[cell["regime"] == regime]
                ax.scatter(part["mu"], part["density"], s=3, alpha=0.6, color=REGIME_PALETTE[rank], label=regime, linewidths=0)

            ax.text(0.03, 0.95, f"Spearman {cell['mu'].corr(cell['density'], method='spearman'):.3f}", transform=ax.transAxes, va="top", fontsize=SMALL_FONT_SIZE)
            ax.set_title(f"{MARKET_NAMES[index_code]}, T = {window_length}")
            add_panel_label(ax, PANEL_LETTERS[row * len(panels) + col])
            style_axis(ax, grid=True)

            if row == len(layout) - 1:
                ax.set_xlabel(r"Mean correlation $\mu$")

            if col == 0:
                ax.set_ylabel(r"Threshold density $\rho$")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(config.REGIME_NAMES), markerscale=3)
    return fig


# Draw the network-topology figures from outputs/network_topology/ (and the drawn cell's network data for the layouts) into results/figures/network_topology/: the centrepiece, the regime-profile box plots, and the density-against-mu figure for the headline window length and for the other window lengths.
def main(dpi=config.FIGURE_DPI):
    set_publication_style()
    folder = config.OUTPUTS_DIR / "network_topology"
    windows = pd.read_csv(folder / "window_topology.csv")
    data = load_networks(config.TOPOLOGY_INDEX, config.TOPOLOGY_WINDOW_LENGTH)
    save_to_subfolder(network_structure_figure(data, pd.read_csv(folder / "representative_windows.csv")), FOLDER, "network_structure_by_regime.png", dpi=dpi)
    save_to_subfolder(regime_comparison_figure(windows), FOLDER, "topology_by_regime_main.png", dpi=dpi)
    headline = config.HEADLINE_WINDOW[0]
    other_windows = [t for t in config.WINDOW_LENGTHS if t != headline]
    half = len(config.INDEX_CODES) // 2
    layouts = {f"threshold_density_vs_mu_{headline}d.png": [[(m, headline) for m in config.INDEX_CODES[:half]], [(m, headline) for m in config.INDEX_CODES[half:]]],
               "threshold_density_vs_mu_other_windows.png": [[(m, t) for t in other_windows] for m in config.INDEX_CODES]}

    for name, layout in layouts.items():
        save_to_subfolder(density_mu_figure(windows, layout), FOLDER, name, dpi=dpi)

    print(f"drew network-topology figures into {config.FIGURES_DIR / FOLDER}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the network-topology-by-regime figures from outputs/network_topology/.")
    parser.add_argument("--dpi", type=int, default=config.FIGURE_DPI, help=f"Export resolution (default: {config.FIGURE_DPI})")
    main(parser.parse_args().dpi)
