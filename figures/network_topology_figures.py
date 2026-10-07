import argparse

import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from regime_networks import config
from regime_networks.networks import load_networks
from regime_networks.plotting import set_publication_style, style_axis, save_to_subfolder, format_date_axis, annotate_events, add_panel_label, REGIME_PALETTE, NEUTRAL_COLOURS, SECTOR_NAMES, GRAPH_LABELS, GRAPH_TYPE_COLOURS

FOLDER = "network_topology"
REGIME_FIGURE_STATISTICS = {"density": r"Edge density $\rho$", "avg_clustering": "Average clustering", "degree_assortativity": "Degree assortativity",
                            "avg_shortest_path_lcc": "Mean shortest path (LCC)", "nmi_sector": "NMI (communities vs. sector)"}
SPLIT_HATCHES = {"train": "", "test": "///"}


# 3x3 grid (construction x regime) of each regime's representative window (mu nearest the regime's median mu under that construction's own model), Kamada-Kawai layouts computed per panel, nodes coloured by sector.
def network_structure_figure(data, representatives):
    stock_names = data["stock_names"]
    sector_of = {s: s.split("_")[0] for s in stock_names}
    sector_names = SECTOR_NAMES.get(config.TOPOLOGY_INDEX, {})
    unique_sectors = sorted(set(sector_of.values()))
    sector_colour = dict(zip(unique_sectors, sns.color_palette("husl", len(unique_sectors))))
    node_colour = [sector_colour[sector_of[s]] for s in stock_names]
    chosen = representatives.set_index(["graph_type", "regime"])

    fig, axes = plt.subplots(len(config.GRAPH_TYPES), len(config.REGIME_NAMES), figsize=(5.2 * len(config.REGIME_NAMES), 5.2 * len(config.GRAPH_TYPES)))
    panel_letters = iter("ABCDEFGHI")

    for row, graph_type in enumerate(config.GRAPH_TYPES):
        for col, regime in enumerate(config.REGIME_NAMES):
            ax = axes[row, col]
            ax.axis("off")
            add_panel_label(ax, next(panel_letters), x=-0.02, y=1.03)
            ax.set_title(f"{GRAPH_LABELS[graph_type]}: {regime}", fontsize=18)

            if (graph_type, regime) not in chosen.index:
                ax.text(0.5, 0.5, "no windows in this regime", transform=ax.transAxes, ha="center", va="center", fontsize=13)
                continue

            window = chosen.loc[(graph_type, regime)]
            G = data[config.GRAPH_KEYS[graph_type]][int(window["window"])]
            pos = nx.kamada_kawai_layout(G, weight="weight")
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=NEUTRAL_COLOURS["reference"], width=0.4, alpha=0.4)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colour, node_size=20, linewidths=0)
            ax.text(0.02, 0.98, f"{pd.Timestamp(window['window_date']).date()} ({window['split']})\n$\\mu$={window['mu']:.3f}\n$\\rho$={nx.density(G):.3f}", transform=ax.transAxes, va="top", fontsize=13,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor=NEUTRAL_COLOURS["ci"]))

    legend = [Patch(facecolor=sector_colour[s], label=sector_names.get(s, s)) for s in unique_sectors]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# Edge density and community-vs-sector NMI over time, one line per construction, with crisis events marked.
def topology_timeseries_figure(windows):
    metrics = [("density", r"Edge density $\rho$"), ("nmi_sector", "NMI (communities vs. sector)")]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 3.1 * len(metrics)), sharex=True)
    dates = pd.to_datetime(windows["window_date"])

    for row, (ax, (column, label)) in enumerate(zip(axes, metrics)):
        for graph_type in config.GRAPH_TYPES:
            sub = windows["graph_type"] == graph_type
            ax.plot(dates[sub], windows.loc[sub, column], color=GRAPH_TYPE_COLOURS[graph_type], linewidth=1.5, label=GRAPH_LABELS[graph_type])

        if row == 0:
            annotate_events(ax, config.CRISIS_EVENTS, dates.min(), dates.max())
        else:
            for event_date in [pd.to_datetime(date_str) for date_str in config.CRISIS_EVENTS if dates.min() <= pd.to_datetime(date_str) <= dates.max()]:
                ax.axvline(event_date, color="#444444", linestyle=":", linewidth=1.3, alpha=0.85, zorder=4)

        ax.set_ylabel(label)
        format_date_axis(ax)
        style_axis(ax, grid=True)
        ax.grid(True, alpha=0.08)

    axes[0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=8)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return fig


# Distribution of each compared statistic by regime (box: interquartile range and median; whiskers: 1.5 IQR), training and test windows side by side, one row per statistic and one column per construction, each construction grouped by its own model's regimes.
def regime_comparison_figure(windows):
    fig, axes = plt.subplots(len(REGIME_FIGURE_STATISTICS), len(config.GRAPH_TYPES), figsize=(4.4 * len(config.GRAPH_TYPES), 2.6 * len(REGIME_FIGURE_STATISTICS)), squeeze=False)

    for row, (statistic, label) in enumerate(REGIME_FIGURE_STATISTICS.items()):
        for col, graph_type in enumerate(config.GRAPH_TYPES):
            ax = axes[row][col]

            for rank, regime in enumerate(config.REGIME_NAMES):
                for offset, split in enumerate(SPLIT_HATCHES):
                    values = windows.loc[(windows["graph_type"] == graph_type) & (windows["regime"] == regime) & (windows["split"] == split), statistic].dropna().to_numpy()

                    if values.size:
                        box = ax.boxplot(values, positions=[3 * rank + offset], widths=0.8, patch_artist=True, showfliers=False, medianprops={"color": "black"})
                        box["boxes"][0].set(facecolor=REGIME_PALETTE[rank], alpha=0.6, hatch=SPLIT_HATCHES[split])

            ax.set_xlim(-1, 3 * len(config.REGIME_NAMES) - 1)
            ax.set_xticks([3 * rank + 0.5 for rank in range(len(config.REGIME_NAMES))])
            ax.set_xticklabels(config.REGIME_NAMES, fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
            style_axis(ax, grid=True)

            if row == 0:
                ax.set_title(GRAPH_LABELS[graph_type], fontweight="bold")

            if col == 0:
                ax.set_ylabel(label, fontsize=8)

    fig.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch=hatch, label=f"{split} windows") for split, hatch in SPLIT_HATCHES.items()], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    return fig


# Degree distribution per construction, pooled over every window of each regime (log density, log1p-spaced bins per panel).
def degree_distribution_figure(degrees):
    fig, axes = plt.subplots(1, len(config.GRAPH_TYPES), figsize=(5.2 * len(config.GRAPH_TYPES), 4.2), sharey=True)

    for ax, graph_type in zip(axes, config.GRAPH_TYPES):
        panel = degrees[degrees["graph_type"] == graph_type]
        bins = np.expm1(np.linspace(0, np.log1p(panel["degree"].max()), 31))

        for rank, regime in enumerate(config.REGIME_NAMES):
            counts = panel[panel["regime"] == regime]

            if not counts.empty:
                ax.hist(counts["degree"], bins=bins, weights=counts["count"], histtype="step", linewidth=2.0, color=REGIME_PALETTE[rank], label=regime, density=True)

        ax.set_xlabel("Degree")
        ax.set_yscale("log")
        ax.set_title(GRAPH_LABELS[graph_type], fontweight="bold")
        style_axis(ax, grid=True)
        ax.grid(True, alpha=0.08)

    axes[0].set_ylabel("Density (log scale)")
    axes[0].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=8)
    plt.tight_layout()
    return fig


# Median largest clique size per construction and regime over all windows, with interquartile bars and, above each bar, the share of windows whose clique enumeration hit CLIQUE_ENUM_CAP (a lower bound there).
def clique_statistics_figure(windows):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width, x = 0.25, np.arange(len(config.REGIME_NAMES))

    for i, graph_type in enumerate(config.GRAPH_TYPES):
        groups = windows[windows["graph_type"] == graph_type].groupby("regime")
        medians = groups["largest_clique"].median().reindex(config.REGIME_NAMES)
        low, high = groups["largest_clique"].quantile(0.25).reindex(config.REGIME_NAMES), groups["largest_clique"].quantile(0.75).reindex(config.REGIME_NAMES)
        capped = groups["clique_enumeration_capped"].mean().reindex(config.REGIME_NAMES)
        bars = ax.bar(x + (i - 1) * width, medians, width, yerr=[medians - low, high - medians], capsize=2, color=GRAPH_TYPE_COLOURS[graph_type], label=GRAPH_LABELS[graph_type])

        for bar, top, share in zip(bars, high, capped):
            if share > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, top + 0.5, f"{share:.0%} capped", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(config.REGIME_NAMES)
    ax.set_ylabel("Largest clique size (median, IQR)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=8)
    style_axis(ax, grid=True)
    plt.tight_layout()
    return fig


# Draw the five network-topology figures from outputs/network_topology/ (and the network data for the layout grid) into results/figures/network_topology/.
def main(dpi=config.FIGURE_DPI):
    set_publication_style()
    folder = config.OUTPUTS_DIR / "network_topology"
    windows = pd.read_csv(folder / "window_topology.csv")
    data = load_networks(config.TOPOLOGY_INDEX, config.TOPOLOGY_WINDOW_LENGTH)
    save_to_subfolder(network_structure_figure(data, pd.read_csv(folder / "representative_windows.csv")), FOLDER, "network_structure_by_regime.png", dpi=dpi)
    save_to_subfolder(topology_timeseries_figure(windows), FOLDER, "topology_timeseries.png", dpi=dpi)
    save_to_subfolder(regime_comparison_figure(windows), FOLDER, "topology_by_regime.png", dpi=dpi)
    save_to_subfolder(degree_distribution_figure(pd.read_csv(folder / "degree_counts.csv")), FOLDER, "degree_distribution_by_regime.png", dpi=dpi)
    save_to_subfolder(clique_statistics_figure(windows), FOLDER, "clique_statistics.png", dpi=dpi)
    print(f"drew network-topology figures into {config.FIGURES_DIR / FOLDER}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the network-topology-by-regime figures from outputs/network_topology/.")
    parser.add_argument("--dpi", type=int, default=config.FIGURE_DPI, help=f"Export resolution (default: {config.FIGURE_DPI})")
    main(parser.parse_args().dpi)
