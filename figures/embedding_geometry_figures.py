import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regime_networks import config
from regime_networks.grid import grid_tag
from regime_networks.geometry import geometry_path
from regime_networks.plotting import set_publication_style, save_to_subfolder, style_axis, mark_split, REGIME_PALETTE, GRAPH_LABELS

MDS_COLOURINGS = {"regime": None, "edgecount": "edge count", "oov": "OOV rate"}


# Create a fresh 1xn row of panels at the shared dashboard size.
def _new_axes_row(n):
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False, constrained_layout=True)
    return fig, axes[0]


# Set year ticks every second year on an imshow axis's window-index axes (both x and y), at the first window of each such year.
def _set_index_date_ticks(ax, dates):
    years = pd.to_datetime(list(dates)).year
    positions, labels = [], []
    last_year = None

    for i, year in enumerate(years):
        if year != last_year and year % 2 == 0:
            positions.append(i)
            labels.append(str(year))
            last_year = year

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)


# Draw one self-similarity heatmap panel over cosine distance's fixed [0, 2] range, with date ticks and the train/test split marked on both axes.
def _plot_selfsimilarity_panel(ax, D, dates, n_train):
    im = ax.imshow(D, cmap="turbo", vmin=0.0, vmax=2.0, aspect="auto")
    _set_index_date_ticks(ax, dates)
    style_axis(ax)

    if 0 < n_train < len(dates):
        split_line = mark_split(ax, n_train - 0.5)
        ax.axhline(n_train - 0.5, color=split_line.get_color(), linestyle=split_line.get_linestyle(),
                   linewidth=split_line.get_linewidth(), zorder=split_line.get_zorder())

    return im


# Draw the self-similarity row: one panel per graph_type, sharing one colour scale and one colourbar.
def _plot_selfsimilarity_figure(distances_by_graph_type, dates_by_graph_type, n_train_by_graph_type, graph_types):
    fig, axes = _new_axes_row(len(graph_types))

    for col, graph_type in enumerate(graph_types):
        image = _plot_selfsimilarity_panel(axes[col], distances_by_graph_type[graph_type], dates_by_graph_type[graph_type], n_train_by_graph_type[graph_type])
        axes[col].set_title(GRAPH_LABELS[graph_type], fontweight="bold")

    fig.colorbar(image, ax=list(axes), shrink=0.8, label="distance")
    return fig


# Draw one 2-D MDS scatter panel coloured by the given colouring, titled with just the graph_type name.
def _plot_mds_panel(ax, coords, colour_values, colour_kind, cmap, vmin, vmax, graph_type):
    if colour_kind == "regime":
        for regime_id, name, hexcolour in zip(range(config.N_REGIMES), config.REGIME_NAMES, REGIME_PALETTE):
            mask = colour_values == regime_id
            ax.scatter(coords[mask, 0], coords[mask, 1], color=hexcolour, s=20, alpha=0.8, label=name, zorder=2)

        sc = None
    else:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=colour_values, cmap=cmap, vmin=vmin, vmax=vmax, s=20, alpha=0.8, zorder=2)

    ax.set_xlabel("MDS Dimension 1")
    ax.set_ylabel("MDS Dimension 2")
    ax.set_title(GRAPH_LABELS[graph_type], fontweight="bold")
    style_axis(ax, grid=True)
    return sc


# Draw one 2-D MDS figure for one colouring: one graph_type panel per column, sharing one colour scale.
def _plot_mds_figure(coords_by_graph_type, colour_values_by_graph_type, colour_kind, graph_types, colourbar_label):
    fig, axes = _new_axes_row(len(graph_types))

    if colour_kind == "regime":
        vmin = vmax = cmap = None
    else:
        all_values = np.concatenate([colour_values_by_graph_type[gt] for gt in graph_types])
        vmin, vmax, cmap = float(all_values.min()), float(all_values.max()), "viridis"

    for ax, graph_type in zip(axes, graph_types):
        sc = _plot_mds_panel(ax, coords_by_graph_type[graph_type], colour_values_by_graph_type[graph_type], colour_kind, cmap, vmin, vmax, graph_type)

    if colour_kind == "regime":
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True, facecolor="white", edgecolor="lightgray")
    else:
        fig.colorbar(sc, ax=list(axes), shrink=0.7, label=colourbar_label)

    return fig


# Draw one window length's self-similarity figure and its three MDS figures from the saved geometry files of every available graph construction.
def draw_window_length(index_code, window_length, dpi):
    entries = {}

    for graph_type in config.GRAPH_TYPES:
        path = geometry_path(index_code, window_length, graph_type)

        if path.exists():
            with np.load(path) as saved:
                entries[graph_type] = {key: saved[key] for key in saved.files}

    if not entries:
        return False

    graph_types = list(entries)
    folder = f"geometry/{grid_tag(index_code, window_length)}"
    fig = _plot_selfsimilarity_figure({gt: entries[gt]["distance"] for gt in graph_types}, {gt: pd.to_datetime(entries[gt]["dates"]) for gt in graph_types},
                                      {gt: int(entries[gt]["n_train"]) for gt in graph_types}, graph_types)
    save_to_subfolder(fig, folder, "geometry_selfsimilarity.png", dpi=dpi)
    colour_arrays = {"regime": "regime_labels", "edgecount": "edge_counts", "oov": "oov"}

    for colour_kind, label in MDS_COLOURINGS.items():
        fig = _plot_mds_figure({gt: entries[gt]["coords"] for gt in graph_types}, {gt: entries[gt][colour_arrays[colour_kind]] for gt in graph_types}, colour_kind, graph_types, colourbar_label=label)
        save_to_subfolder(fig, folder, f"geometry_mds_graph2vec_{colour_kind}.png", dpi=dpi)

    return True


# Draw the embedding-geometry figures of GEOMETRY_INDEX for every window length that has saved geometry files.
def main(dpi=config.FIGURE_DPI):
    set_publication_style()
    drawn = [t for t in config.WINDOW_LENGTHS if draw_window_length(config.GEOMETRY_INDEX, t, dpi)]
    print(f"drew embedding-geometry figures for window lengths {drawn} into {config.FIGURES_DIR / 'geometry'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the Graph2Vec embedding-geometry figures (self-similarity heatmaps, MDS maps) from outputs/geometry/.")
    parser.add_argument("--dpi", type=int, default=config.FIGURE_DPI, help=f"Export resolution (default: {config.FIGURE_DPI})")
    main(parser.parse_args().dpi)
