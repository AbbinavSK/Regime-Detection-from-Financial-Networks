import numpy as np
import pandas as pd
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from regime_networks import config

REGIME_PALETTE = ["#2ca02c", "#1f77b4", "#d62728"]
NEUTRAL_COLOURS = {"primary": "#000000", "reference": "#7f7f7f", "ci": "#d9d9d9"}
WINDOW_LENGTH_COLOURS = {378: "blue", 132: "green", 63: "purple"}
SECTOR_NAMES = {
    "SP500": {
        "IT": "Information Technology",
        "HC": "Health Care",
        "FIN": "Financials",
        "CD": "Consumer Discretionary",
        "CS": "Communication Services",
        "IND": "Industrials",
        "CST": "Consumer Staples",
        "ENE": "Energy",
        "UTL": "Utilities",
        "RE": "Real Estate",
        "MAT": "Materials",
    },
    "Nikkei225": {
        "PHARMA": "Pharmaceuticals",
        "ELECMC": "Electric Machinery",
        "AUTO": "Automobiles & Auto Parts",
        "PRECIS": "Precision Instruments",
        "COMMS": "Communications",
        "BANK": "Banking",
        "OTHFIN": "Other Financial Services",
        "SECUR": "Securities",
        "INSUR": "Insurance",
        "FISH": "Fishery, Agriculture & Forestry",
        "FOODS": "Foods",
        "RETAIL": "Retail Trade",
        "SERVIC": "Services",
        "MINING": "Mining",
        "TEXTIL": "Textiles & Apparel",
        "PULP": "Pulp & Paper",
        "CHEMIC": "Chemicals",
        "PETROL": "Oil & Coal Products",
        "RUBBER": "Rubber Products",
        "GLASS": "Glass & Ceramics",
        "STEEL": "Iron & Steel",
        "NONFER": "Nonferrous Metals",
        "TRADCO": "Trading Companies",
        "CONSTR": "Construction",
        "MACHIN": "Machinery",
        "SHIPBD": "Shipbuilding",
        "OTHMFG": "Other Manufacturing",
        "REALES": "Real Estate",
        "RAILBS": "Railroad & Bus",
        "LANDTR": "Land Transportation",
        "MARINE": "Marine Transportation",
        "AIRTR": "Air Transportation",
        "ELECPW": "Electric Power",
        "GAS": "Gas",
    },
    "FTSE350": {
        "IND": "Industrials",
        "ENE": "Energy",
        "CD": "Consumer Discretionary",
        "FIN": "Financials",
        "MAT": "Materials",
        "CST": "Consumer Staples",
        "UTL": "Utilities",
        "HC": "Health Care",
        "TECH": "Technology",
        "TEL": "Telecommunications",
        "RE": "Real Estate",
    },
    "CSI300": {
        "IT": "Information Technology",
        "IND": "Industrials",
        "FIN": "Financials",
        "MAT": "Materials",
        "HC": "Health Care",
        "CST": "Consumer Staples",
        "CD": "Consumer Discretionary",
        "UTL": "Utilities",
        "ENE": "Energy",
        "CS": "Communication",
        "RE": "Real Estate",
    },
}
GRAPH_LABELS = {"threshold": "Threshold", "mst": "MST-Threshold", "tmfg": "TMFG-Threshold"}
GRAPH_TYPE_COLOURS = {"threshold": "#7570b3", "mst": "#1b9e77", "tmfg": "#d95f02"}
STRATEGY_COLOURS = {"Fully invested": "#7570b3", "Regime-timed": "#1b9e77", config.REGIME_FORECAST_STRATEGY: "#d95f02", config.TRAILING_VOL_STRATEGY: "#e7298a", config.MU_ONLY_STRATEGY: "#66a61e"}


# Set global rcParams so every manuscript figure shares one visual language.
def set_publication_style():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.15,
        "grid.color": "#000000",
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": config.FIGURE_DPI,
        "savefig.bbox": "tight",
    })


# Apply the manuscript's spine/tick/grid convention to a single axis.
def style_axis(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(direction="out", length=4, width=1.0)

    if grid:
        ax.grid(True, alpha=0.15, color="black", linewidth=0.6)
    else:
        ax.grid(False)

    return ax


# Add a bold panel label (e.g. "A") outside an axis's top-left corner, journal-style.
def add_panel_label(ax, label, x=-0.15, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="right")


# Shade a list of (start, end) x-intervals on an axis with one consistent colour/alpha.
def shade_periods(ax, intervals, colour, alpha=0.2, zorder=1):
    for start, end in intervals:
        ax.axvspan(start, end, color=colour, alpha=alpha, linewidth=0, zorder=zorder)


# Format an axis's x-axis as evenly-spaced year ticks, matching every date-indexed figure.
def format_date_axis(ax, interval_years=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(interval_years))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# Draw the dashed train/test split marker at split_date and return its legend handle.
def mark_split(ax, split_date):
    ax.axvline(split_date, color="#333333", linestyle="--", linewidth=1.5, zorder=5)
    return Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.5, label="Train/test split")


# Annotate a {date_str: label} dict (e.g. config.CRISIS_EVENTS) as labelled dotted vertical lines within [x_min, x_max].
def annotate_events(ax, events, x_min, x_max):
    y_top = ax.get_ylim()[1]

    for date_str, label in events.items():
        event_date = pd.to_datetime(date_str)

        if x_min <= event_date <= x_max:
            ax.axvline(event_date, color="#444444", linestyle=":", linewidth=1.3, alpha=0.85, zorder=4)
            ax.text(event_date, y_top * 0.97, label, rotation=90, fontsize=9, color="#444444", va="top", ha="center")


# Save a figure into results/figures/{subfolder}/ at the given resolution with the manuscript's export settings, and close it.
def save_to_subfolder(fig, subfolder, name, dpi=config.FIGURE_DPI):
    folder = config.FIGURES_DIR / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=dpi, bbox_inches="tight", transparent=False)
    plt.close(fig)


# The time span each dated window covers when shaded, from its own date to the next window's, the last one extended by the step before it so it stays visible.
def _date_spans(dates):
    dates = list(dates)
    last_step = dates[-1] - dates[-2] if len(dates) > 1 else pd.Timedelta(days=1)
    return list(zip(dates, dates[1:] + [dates[-1] + last_step]))


# Plot a regime timeline shaded by rank, train and test on one continuous axis.
def plot_regime_timeline(train_dates, train_mu, train_labels, test_dates, test_mu, test_labels, rank_of_cluster):
    fig, ax = plt.subplots(figsize=(12, 4))
    dates = list(train_dates) + list(test_dates)
    ranks = [rank_of_cluster[label] for label in list(train_labels) + list(test_labels)]
    spans = _date_spans(dates)

    for r in range(config.N_REGIMES):
        for alpha, windows in ((0.5, range(len(train_dates))), (0.25, range(len(train_dates), len(dates)))):
            shade_periods(ax, [spans[i] for i in windows if ranks[i] == r], REGIME_PALETTE[r], alpha=alpha)

    ax.plot(train_dates, train_mu, color="black", linewidth=2)
    ax.plot(test_dates, test_mu, color="black", linewidth=2)

    handles = [Patch(color=REGIME_PALETTE[r], alpha=0.5, label=config.REGIME_NAMES[r]) for r in range(config.N_REGIMES)]

    if len(test_dates) > 0:
        handles.append(mark_split(ax, pd.Timestamp(config.TRAIN_TEST_CUTOFF)))

    ax.set_xlabel("Date")
    ax.set_ylabel(r"Average Correlation ($\mu$)")
    format_date_axis(ax)
    style_axis(ax, grid=True)
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray")
    plt.tight_layout()
    return fig


# Plot VIX with predicted-crisis periods shaded, train and test on one continuous axis. VIX curve colour marks the window length.
def plot_vix_overlay(train_dates, train_validation, test_dates, test_validation, window_length):
    vix_colour = WINDOW_LENGTH_COLOURS[window_length]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train_dates, train_validation["series_eod"], color=vix_colour, linewidth=2.5, zorder=3)
    ax.plot(test_dates, test_validation["series_eod"], color=vix_colour, linewidth=2.5, zorder=3)
    ax.axhline(config.VIX_CRISIS_THRESHOLD, color=NEUTRAL_COLOURS["reference"], linestyle="--", linewidth=1.2, zorder=2)

    dates = train_dates.append(test_dates)
    crisis = np.concatenate([train_validation["pred"], test_validation["pred"]]).astype(bool)
    spans = _date_spans(dates)

    for alpha, windows in ((0.5, range(len(train_dates))), (0.25, range(len(train_dates), len(dates)))):
        shade_periods(ax, [spans[i] for i in windows if crisis[i]], REGIME_PALETTE[2], alpha=alpha)

    annotate_events(ax, config.CRISIS_EVENTS, dates[0], dates[-1])

    handles = [
        Line2D([0], [0], color=vix_colour, lw=2.5, label=f"VIX (T={window_length})"),
        Line2D([0], [0], color=NEUTRAL_COLOURS["reference"], lw=1.2, linestyle="--", label="VIX Threshold"),
        Patch(facecolor=REGIME_PALETTE[2], alpha=0.5, label="Predicted Crisis Regime"),
    ]

    if len(test_dates) > 0:
        handles.append(mark_split(ax, pd.Timestamp(config.TRAIN_TEST_CUTOFF)))

    ax.set_xlabel("Date")
    ax.set_ylabel("VIX")
    ax.set_xlim(dates[0], spans[-1][1])
    format_date_axis(ax)
    style_axis(ax, grid=True)
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray")
    plt.tight_layout()
    return fig


# Plot train vs test empirical transition matrices side by side, annotated with cell probabilities and occupancy.
def plot_transition_comparison(train_transitions, test_transitions, train_occupancy, test_occupancy, title_tag):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    for i, (ax, P, occ, title) in enumerate(((axes[0], train_transitions, train_occupancy, "Train"), (axes[1], test_transitions, test_occupancy, "Test"))):
        im = ax.imshow(P, cmap="viridis", vmin=0, vmax=1)
        style_axis(ax)
        ax.set_xticks(range(config.N_REGIMES))
        ax.set_yticks(range(config.N_REGIMES))
        ax.set_xticklabels(config.REGIME_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(config.REGIME_NAMES if i == 0 else [])
        ax.set_title(f"{title} (n={occ.sum()})", fontweight="bold")

        for r in range(config.N_REGIMES):
            for c in range(config.N_REGIMES):
                val = P[r, c]
                label = "NaN" if np.isnan(val) else f"{val:.2f}"
                ax.text(c, r, f"{label}\n(occ={occ[r]})", ha="center", va="center", fontsize=8,
                        color="white" if (not np.isnan(val) and val > 0.5) else "black")

    fig.colorbar(im, ax=axes, shrink=0.8, label="transition probability")
    plt.suptitle(f"Empirical transitions: {title_tag}", fontweight="bold")
    return fig
