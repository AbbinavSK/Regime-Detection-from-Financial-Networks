import argparse

import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch

from regime_networks import config
from regime_networks.networks import load_networks
from regime_networks.plotting import set_publication_style, SECTOR_NAMES, GRAPH_LABELS

FOLDER = "animations"


# Animate one index's windows as a GIF: each construction's network on one fixed layout, nodes coloured by sector, under the window's correlations on that network's edges.
def build_animation(index_code, window_length):
    data = load_networks(index_code, window_length)
    window_shift = dict(config.WINDOW_CONFIGS)[window_length]
    correlations, mu_values, dates = data["correlation_matrices"], data["mu_values"], data["window_dates"]
    graphs = {graph_type: data[config.GRAPH_KEYS[graph_type]] for graph_type in config.GRAPH_TYPES}
    stock_sector = {s: s.split("_")[0] for s in data["stock_names"]}
    unique_sectors = sorted(set(stock_sector.values()))
    sector_colour = dict(zip(unique_sectors, sns.color_palette("husl", len(unique_sectors))))
    node_colours = {s: sector_colour[stock_sector[s]] for s in data["stock_names"]}

    set_publication_style()
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])
    heatmap_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    network_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    fig.subplots_adjust(hspace=0.12, wspace=0.05, right=0.92)
    pos = nx.kamada_kawai_layout(graphs["tmfg"][0], weight="weight")
    fig.colorbar(plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=-1, vmax=1)), cax=fig.add_axes([0.94, 0.55, 0.015, 0.33]), label="correlation")
    sector_names = SECTOR_NAMES.get(index_code, {})
    fig.legend(handles=[Patch(facecolor=sector_colour[s], label=sector_names.get(s, s)) for s in unique_sectors], loc="lower center", ncol=min(len(unique_sectors), 8), fontsize=8, frameon=False)

    # Draw one network on the fixed layout, annotated with the window's mean correlation and the graph's edge density.
    def draw_network(G, ax, mu_value):
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="black", width=0.6, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[node_colours[n] for n in G.nodes()], node_size=35, alpha=0.9, linewidths=0)

        for y, text in ((0.98, f"μ = {mu_value:.3f}"), (0.88, f"ρ = {nx.density(G):.3f}")):
            ax.text(0.02, y, text, transform=ax.transAxes, va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"))

        ax.axis("off")

    # Redraw every panel for one window.
    def update(frame):
        for ax, graph_type in zip(heatmap_axes, config.GRAPH_TYPES):
            ax.clear()
            edges = nx.to_numpy_array(graphs[graph_type][frame], nodelist=data["stock_names"], weight=None) > 0
            ax.imshow(np.where(edges, correlations[frame], np.nan), cmap="jet", vmin=-1, vmax=1)
            ax.set_title(f"{GRAPH_LABELS[graph_type]}: correlations on edges")
            ax.set_xticks([])
            ax.set_yticks([])

        for ax, graph_type in zip(network_axes, config.GRAPH_TYPES):
            ax.clear()
            draw_network(graphs[graph_type][frame], ax, mu_values[frame])

        fig.suptitle(f"{index_code} — Threshold vs MST-Threshold vs TMFG-Threshold (T={window_length}, S={window_shift})\n{dates[frame].date()}", fontsize=15, fontweight="bold", y=0.98)

    frames = len(correlations)
    folder = config.FIGURES_DIR / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{index_code}_Threshold_vs_MST_vs_TMFG_{window_length}d.gif"
    animation.FuncAnimation(fig, update, frames=frames).save(path, writer="pillow", fps=config.ANIMATION_FPS, dpi=config.ANIMATION_DPI)
    plt.close(fig)
    print(f"saved {path} ({frames} frames)")


# Render the animation for each index at the configured window length into results/figures/animations/.
def main(index_codes=None, window_length=None):
    for index_code in config.INDEX_CODES if index_codes is None else index_codes:
        build_animation(index_code, config.ANIMATION_WINDOW_LENGTH if window_length is None else window_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render presentation GIFs of each window's correlation heatmap and its three network constructions.")
    parser.add_argument("--index", choices=config.INDEX_CODES + ["all"], default="all", help="Index to animate (default: all)")
    parser.add_argument("--window-length", type=int, choices=config.WINDOW_LENGTHS, default=config.ANIMATION_WINDOW_LENGTH, help=f"Window length (default: {config.ANIMATION_WINDOW_LENGTH})")
    args = parser.parse_args()
    main(config.INDEX_CODES if args.index == "all" else [args.index], args.window_length)
