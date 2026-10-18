import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from regime_networks import config
from regime_networks.grid import grid_tag
from regime_networks.geometry import geometry_path
from regime_networks.plotting import set_publication_style, save_to_subfolder, add_panel_label, legend_above, FIGURE_WIDTH, PANEL_LETTERS, REGIME_PALETTE, GRAPH_LABELS, SMALL_FONT_SIZE

COLOUR_BAR_LABELS = {"selfsimilarity": "Cosine distance", "date": "Window end (year)", "edgecount": "Edges", "oov": "OOV rate"}
ROW_HEIGHT = 1.55
MARKER_SIZE = 4
STORED_KEYS = {"date": "dates", "edgecount": "edge_counts", "oov": "oov"}


# Year ticks every fourth year on an imshow axis's window-index axes (both x and y), at the first window of each such year.
def _set_index_date_ticks(ax, dates):
    years = pd.to_datetime(list(dates)).year
    positions = [i for i in range(len(years)) if years[i] % 4 == 0 and (i == 0 or years[i - 1] != years[i])]
    ax.set_xticks(positions, [str(years[i]) for i in positions], rotation=90)
    ax.set_yticks(positions, [str(years[i]) for i in positions])


# Draw one self-similarity heatmap over cosine distance's fixed [0, 2] range with the train/test split dashed on both axes; returns the image for the row's colour bar.
def _selfsimilarity_panel(ax, entry):
    image = ax.imshow(entry["distance"], cmap="turbo", vmin=0.0, vmax=2.0, aspect="equal")
    _set_index_date_ticks(ax, entry["dates"])
    split = int(entry["n_train"]) - 0.5

    for line in (ax.axvline, ax.axhline):
        line(split, color="black", linestyle="--", linewidth=0.8)

    ax.tick_params(length=2)
    return image


# Draw one frameless MDS map (its axes carry no meaning) coloured by regime (discrete) or by a continuous value on a shared scale, with the projection's stress beneath it; returns the scatter for the row's colour bar (None for regimes).
def _mds_panel(ax, entry, kind, scale):
    coords = entry["coords"]

    if kind == "regime":
        for rank in range(config.N_REGIMES):
            mask = entry["regime_labels"] == rank
            ax.scatter(coords[mask, 0], coords[mask, 1], color=REGIME_PALETTE[rank], s=MARKER_SIZE, alpha=0.8, linewidths=0)

        scatter = None
    else:
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=entry[kind], cmap="viridis", vmin=scale[0], vmax=scale[1], s=MARKER_SIZE, alpha=0.8, linewidths=0)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.text(0.5, -0.02, f"stress {float(entry['stress']):.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=SMALL_FONT_SIZE - 1)
    return scatter


# Window end dates as decimal years, the date colouring of the MDS maps.
def _decimal_years(dates):
    dates = pd.to_datetime(list(dates))
    return np.asarray(dates.year + (dates.dayofyear - 1) / 365.25, dtype=float)


# One window length's geometry figure: a row per requested kind (self-similarity heatmaps, or MDS maps coloured by regime, date, edge count or OOV rate), one column per construction, each row lettered and with its own colour bar, or a legend above it.
def geometry_figure(entries, rows):
    graph_types = list(entries)
    fig, axes = plt.subplots(len(rows), len(graph_types), figsize=(FIGURE_WIDTH, ROW_HEIGHT * len(rows) + 0.2), squeeze=False)
    values = {kind: {g: _decimal_years(entries[g]["dates"]) if kind == "date" else entries[g][STORED_KEYS[kind]] for g in graph_types} for kind in rows if kind in STORED_KEYS}

    for row, kind in enumerate(rows):
        scale = (min(float(v.min()) for v in values[kind].values()), max(float(v.max()) for v in values[kind].values())) if kind in values else None

        for col, graph_type in enumerate(graph_types):
            ax = axes[row, col]
            entry = {**entries[graph_type], **({kind: values[kind][graph_type]} if kind in values else {})}
            mappable = _selfsimilarity_panel(ax, entry) if kind == "selfsimilarity" else _mds_panel(ax, entry, kind, scale)

            if row == 0:
                ax.set_title(GRAPH_LABELS[graph_type], loc="center")

            if col == 0:
                add_panel_label(ax, PANEL_LETTERS[row])

        if kind == "regime":
            legend_above(axes[row, -1], [Patch(color=REGIME_PALETTE[r], label=name) for r, name in enumerate(config.REGIME_NAMES)])
            axes[row, -1].get_legend().set_in_layout(False)
        else:
            fig.colorbar(mappable, ax=list(axes[row]), shrink=0.9, label=COLOUR_BAR_LABELS[kind], pad=0.02)

    return fig


# Draw each geometry figure of config.GEOMETRY_FIGURES from the saved geometry files of GEOMETRY_INDEX; returns the window lengths drawn.
def main(dpi=config.FIGURE_DPI):
    set_publication_style()
    drawn = []

    for window_length, rows in config.GEOMETRY_FIGURES:
        paths = {g: geometry_path(config.GEOMETRY_INDEX, window_length, g) for g in config.GRAPH_TYPES}
        entries = {}

        for graph_type, path in paths.items():
            with np.load(path) as saved:
                entries[graph_type] = {key: saved[key] for key in saved.files}

        save_to_subfolder(geometry_figure(entries, rows), f"geometry/{grid_tag(config.GEOMETRY_INDEX, window_length)}", f"geometry_{'_'.join(rows)}.png", dpi=dpi)
        drawn.append(window_length)

    print(f"drew {len(drawn)} embedding-geometry figures (window lengths {drawn}) into {config.FIGURES_DIR / 'geometry'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the Graph2Vec embedding-geometry figures (self-similarity heatmaps, MDS maps) from outputs/geometry/.")
    parser.add_argument("--dpi", type=int, default=config.FIGURE_DPI, help=f"Export resolution (default: {config.FIGURE_DPI})")
    main(parser.parse_args().dpi)
