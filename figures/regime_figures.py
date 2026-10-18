import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from regime_networks import config
from regime_networks.grid import labels_path
from regime_networks.regimes import empirical_transitions
from regime_networks.plotting import (set_publication_style, save_to_subfolder, add_panel_label, legend_above, draw_regime_timeline, draw_vix_overlay, draw_transition_matrix,
                                      FIGURE_WIDTH, PANEL_LETTERS, MARKET_NAMES, DESCRIPTOR_LABELS)

RANK_ORDER = np.arange(config.N_REGIMES)
PANEL_HEIGHT = 1.75


# One panel's training and test windows from its sweep labels, with the regime column of its clustering method.
def _panel_windows(index_code, graph_type, descriptor, method):
    labels = pd.read_csv(labels_path(index_code, config.HEADLINE_WINDOW[0], graph_type, descriptor), parse_dates=["window_date"])
    return labels[labels["split"] == "train"], labels[labels["split"] == "test"], f"{method}_regime"


# Draw a timeline or VIX overlay panel on an axis (HMM training windows with their full-sequence Viterbi labels, every other window with its causal labels); returns the legend handles.
def draw_panel(ax, index_code, graph_type, descriptor, method, kind):
    train, test, column = _panel_windows(index_code, graph_type, descriptor, method)

    if kind == "timeline":
        train_display = train[f"{column}_smoothed"].astype(int).to_numpy() if f"{column}_smoothed" in train else train[column].to_numpy()
        return draw_regime_timeline(ax, list(train["window_date"]), train["mu"].to_numpy(), train_display, list(test["window_date"]), test["mu"].to_numpy(), test[column].to_numpy())

    crisis = {split: {"series_eod": part["vix"].to_numpy(), "pred": (part[column] == config.N_REGIMES - 1).astype(int).to_numpy()} for split, part in (("train", train), ("test", test))}
    return draw_vix_overlay(ax, pd.DatetimeIndex(train["window_date"]), crisis["train"], pd.DatetimeIndex(test["window_date"]), crisis["test"])


# A figure of stacked timeline and VIX panels sharing the date axis: panels of one kind share one legend above the figure and are titled by what differs between them (market or descriptor); panels of different kinds each carry their own legend.
def stacked_figure(panels):
    fig, axes = plt.subplots(len(panels), 1, figsize=(FIGURE_WIDTH, PANEL_HEIGHT * len(panels) + 0.25), sharex=True, squeeze=False)
    one_kind = len({panel[4] for panel in panels}) == 1
    varying = next((i for i in (0, 2) if len({panel[i] for panel in panels}) > 1), None)

    for ax, letter, panel in zip(axes[:, 0], PANEL_LETTERS, panels):
        handles = draw_panel(ax, *panel)

        if len(panels) > 1:
            add_panel_label(ax, letter)

        if varying is not None:
            ax.set_title((MARKET_NAMES if varying == 0 else DESCRIPTOR_LABELS)[panel[varying]])

        if not one_kind:
            legend_above(ax, handles)

    if one_kind:
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))

    axes[-1, 0].set_xlabel("Date")
    return fig


# The training and test transition matrices of one grid point and method side by side, sharing one colour bar; causal labels on both sides.
def transitions_figure(index_code, graph_type, descriptor, method):
    train, test, column = _panel_windows(index_code, graph_type, descriptor, method)
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.35))

    for ax, letter, (period, part) in zip(axes, PANEL_LETTERS, (("Training", train), ("Test", test))):
        P, occupancy = empirical_transitions(part[column].to_numpy(), config.N_REGIMES, RANK_ORDER)
        image = draw_transition_matrix(ax, P, occupancy, f"{period} windows (n = {occupancy.sum()})")
        add_panel_label(ax, letter)

    axes[0].set_ylabel("Current window")
    fig.colorbar(image, ax=list(axes), shrink=0.85, label="Transition probability")
    return fig


# Draw the regime figures the paper and its supplement show (config.REGIME_FIGURES) from the sweeps' saved per-window labels in outputs/labels/.
def main(dpi=config.FIGURE_DPI):
    set_publication_style()

    for name, panels in config.REGIME_FIGURES.items():
        fig = transitions_figure(*panels[0][:4]) if panels[0][4] == "transitions" else stacked_figure(panels)
        save_to_subfolder(fig, "regimes", name, dpi=dpi)

    print(f"drew {len(config.REGIME_FIGURES)} regime figures into {config.FIGURES_DIR / 'regimes'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the paper's regime timelines, VIX overlays and transition matrices from the sweeps' saved per-window labels.")
    parser.add_argument("--dpi", type=int, default=config.FIGURE_DPI, help=f"Export resolution (default: {config.FIGURE_DPI})")
    main(parser.parse_args().dpi)
