import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from regime_networks import config
from regime_networks.grid import grid_tag, parse_output_stem
from regime_networks.regimes import empirical_transitions
from regime_networks.plotting import set_publication_style, save_to_subfolder, plot_regime_timeline, plot_vix_overlay, plot_transition_comparison

METHODS = ("kmeans", "hmm")
RANK_ORDER = np.arange(config.N_REGIMES)


# Draw one grid point's regime timeline (HMM training windows with their full-sequence Viterbi labels), VIX overlay (when the labels carry VIX) and train/test transition comparison (causal labels on both sides) for each clustering method, into a folder named by the grid point's tag.
def draw_grid_point(labels, grid_point, dpi):
    tag, descriptor, window_length = grid_tag(grid_point["index_code"], grid_point["window_length"], grid_point["graph_type"]), grid_point["descriptor"], grid_point["window_length"]
    train, test = labels[labels["split"] == "train"], labels[labels["split"] == "test"]

    for method in METHODS:
        column = f"{method}_regime"
        train_display = train[f"{column}_smoothed"].astype(int).to_numpy() if f"{column}_smoothed" in labels else train[column].to_numpy()
        fig = plot_regime_timeline(list(train["window_date"]), train["mu"].to_numpy(), train_display, list(test["window_date"]), test["mu"].to_numpy(), test[column].to_numpy(), RANK_ORDER)
        save_to_subfolder(fig, f"regimes/{tag}", f"{descriptor}_{method}_timeline.png", dpi=dpi)

        if "vix" in labels:
            crisis = {split: {"series_eod": part["vix"].to_numpy(), "pred": (part[column] == config.N_REGIMES - 1).astype(int).to_numpy()} for split, part in (("train", train), ("test", test))}
            fig = plot_vix_overlay(pd.DatetimeIndex(train["window_date"]), crisis["train"], pd.DatetimeIndex(test["window_date"]), crisis["test"], window_length)
            save_to_subfolder(fig, f"regimes/{tag}", f"{descriptor}_{method}_vix_overlay.png", dpi=dpi)

        (train_transitions, train_occupancy), (test_transitions, test_occupancy) = (empirical_transitions(part[column].to_numpy(), config.N_REGIMES, RANK_ORDER) for part in (train, test))
        fig = plot_transition_comparison(train_transitions, test_transitions, train_occupancy, test_occupancy, f"{tag} ({method})")
        save_to_subfolder(fig, f"regimes/{tag}", f"{descriptor}_{method}_transition_comparison.png", dpi=dpi)


# Render every grid point's regime figures from outputs/labels/, optionally only files whose stem contains a filter string.
def main(dpi=config.FIGURE_DPI, contains=""):
    set_publication_style()
    paths = sorted(p for p in config.LABELS_DIR.glob("*.csv") if contains in p.stem)

    for path in paths:
        draw_grid_point(pd.read_csv(path, parse_dates=["window_date"]), parse_output_stem(path.stem), dpi)

    print(f"drew regime figures for {len(paths)} grid points into {config.FIGURES_DIR / 'regimes'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw regime timelines, VIX overlays and transition comparisons from the sweeps' saved per-window labels.")
    parser.add_argument("--dpi", type=int, default=config.FIGURE_DPI, help=f"Export resolution (default: {config.FIGURE_DPI}; {config.DASHBOARD_DPI} is enough for browsing the full grid)")
    parser.add_argument("--contains", default="", help="Only grid points whose labels file name contains this text, e.g. SP500_132d_mst_graph2vec")
    args = parser.parse_args()
    main(args.dpi, args.contains)
