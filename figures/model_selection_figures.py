import argparse

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from regime_networks import config
from regime_networks.grid import output_stem, headline_grid_point
from regime_networks.regimes import aic_minimum_location
from regime_networks.plotting import set_publication_style, style_axis, save_to_subfolder, NEUTRAL_COLOURS, GRAPH_LABELS, GRAPH_TYPE_COLOURS

FOLDER = "model_selection"
CRITERIA = {"kmeans_inertia": ("Inertia", "kmeans_elbow"), "hmm_aic": ("AIC", "hmm_aic")}
AIC_MINIMUM_MARKERS = {True: {"marker": "*", "markersize": 11}, False: {"marker": "^", "markersize": 8, "markerfacecolor": "none", "markeredgewidth": 1.5}}


# Mark a line's lowest AIC in the given colour: starred if interior to MODEL_SELECTION_K_RANGE, an open triangle if at either end of it.
def _mark_aic_minimum(ax, line, colour):
    if line["hmm_aic"].notna().any():
        best = line.loc[line["hmm_aic"].idxmin()]
        ax.plot(best["k"], best["hmm_aic"], color=colour, linestyle="none", zorder=4, **AIC_MINIMUM_MARKERS[aic_minimum_location(best["k"]) == "interior"])


# Legend entries explaining the two AIC-minimum markers.
def _aic_marker_handles():
    return [Line2D([0], [0], color=NEUTRAL_COLOURS["primary"], linestyle="none", label="lowest AIC", **AIC_MINIMUM_MARKERS[True]),
            Line2D([0], [0], color=NEUTRAL_COLOURS["primary"], linestyle="none", label="lowest AIC at the edge of the K range", **AIC_MINIMUM_MARKERS[False])]


# One model-selection criterion against the candidate number of regimes for the headline grid point, with the chosen K dashed and, for AIC, the minimum marked as interior or at the edge of the range.
def criterion_figure(results, column, ylabel, title):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(results["k"], results[column], marker="o", color=NEUTRAL_COLOURS["primary"], linewidth=2)
    ax.axvline(config.N_REGIMES, color=NEUTRAL_COLOURS["reference"], linestyle="--", linewidth=1.2)

    if column == "hmm_aic":
        _mark_aic_minimum(ax, results, NEUTRAL_COLOURS["primary"])
        ax.legend(handles=_aic_marker_handles(), loc="best", frameon=False, fontsize=8)

    ax.set_xlabel("k (number of regimes)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax, grid=True)
    plt.tight_layout()
    return fig


# One criterion for every grid point of a descriptor: a panel per market (rows) and window length (columns), one line per construction, the chosen K dashed and, for AIC, each line's minimum starred when interior to the K range and an open triangle when at its edge.
def criterion_grid_figure(results, column, ylabel, descriptor):
    markets = [code for code in config.INDEX_CODES if code in set(results["index_code"])]
    lengths = [length for length in config.WINDOW_LENGTHS if length in set(results["window_length"])]
    fig, axes = plt.subplots(len(markets), len(lengths), figsize=(3.6 * len(lengths), 2.6 * len(markets)), sharex=True, squeeze=False)

    for (row, index_code), (col, window_length) in [(r, c) for r in enumerate(markets) for c in enumerate(lengths)]:
        ax = axes[row][col]
        panel = results[(results["index_code"] == index_code) & (results["window_length"] == window_length)]
        constructions = panel["graph_type"].fillna("none")

        for graph_type, line in [(g, panel[constructions == g]) for g in config.GRAPH_TYPES + ["none"] if (constructions == g).any()]:
            colour = GRAPH_TYPE_COLOURS.get(graph_type, NEUTRAL_COLOURS["primary"])
            ax.plot(line["k"], line[column], marker="o", markersize=3, linewidth=1.4, color=colour, label=GRAPH_LABELS.get(graph_type, descriptor))

            if column == "hmm_aic":
                _mark_aic_minimum(ax, line, colour)

        ax.axvline(config.N_REGIMES, color=NEUTRAL_COLOURS["reference"], linestyle="--", linewidth=1.0)
        ax.set_title(f"{index_code}, T={window_length}", fontsize=9)
        ax.tick_params(labelsize=7)
        style_axis(ax, grid=True)

        if col == 0:
            ax.set_ylabel(ylabel, fontsize=8)

        if row == len(markets) - 1:
            ax.set_xlabel("k", fontsize=8)

    handles = axes[0][0].get_legend_handles_labels()[0] + (_aic_marker_handles() if column == "hmm_aic" else [])

    if len(handles) > 1:
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(f"{descriptor}: {ylabel} by number of regimes (train)", fontsize=11, y=1.03)
    plt.tight_layout()
    return fig


# Draw the headline K-means elbow and HMM AIC curves and, per descriptor, the grids over every market, window length and construction from outputs/model_selection.csv into results/figures/model_selection/.
def main(dpi=config.FIGURE_DPI):
    set_publication_style()
    results = pd.read_csv(config.OUTPUTS_DIR / "model_selection.csv")
    headline = results[results["tag"] == output_stem(*headline_grid_point())]
    save_to_subfolder(criterion_figure(headline, "kmeans_inertia", "Inertia", "K-Means Elbow Plot (train)"), FOLDER, "kmeans_elbow.png", dpi=dpi)
    save_to_subfolder(criterion_figure(headline, "hmm_aic", "AIC", "Gaussian HMM AIC (train)"), FOLDER, "hmm_aic.png", dpi=dpi)

    for descriptor in [d for d in config.DESCRIPTORS if d in set(results["descriptor"])]:
        for column, (ylabel, stem) in CRITERIA.items():
            save_to_subfolder(criterion_grid_figure(results[results["descriptor"] == descriptor], column, ylabel, descriptor), FOLDER, f"{stem}_{descriptor}.png", dpi=dpi)

    print(f"drew model-selection figures into {config.FIGURES_DIR / FOLDER}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the K-means elbow and HMM AIC model-selection figures from outputs/model_selection.csv.")
    parser.add_argument("--dpi", type=int, default=config.FIGURE_DPI, help=f"Export resolution (default: {config.FIGURE_DPI})")
    main(parser.parse_args().dpi)
