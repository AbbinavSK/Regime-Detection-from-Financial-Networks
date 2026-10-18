import numpy as np
import pandas as pd
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from regime_networks import config

FIGURE_WIDTH = 372 / 72.27
PANEL_LETTERS = "abcdefghijkl"
FONT_SIZE = 8
SMALL_FONT_SIZE = 7
PANEL_LABEL_SIZE = 9
REGIME_PALETTE = ["#2ca02c", "#1f77b4", "#d62728"]
TRANSITION_CMAP = "YlGnBu"
NEUTRAL_COLOURS = {"primary": "#000000", "reference": "#7f7f7f", "ci": "#d9d9d9"}
VIX_COLOUR = "green"
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
GRAPH_LABELS = {"threshold": "Threshold", "mst": "MST–threshold", "tmfg": "TMFG–threshold"}
MARKET_NAMES = {"SP500": "S&P 500", "Nikkei225": "Nikkei 225", "FTSE350": "FTSE 350", "CSI300": "CSI 300"}
DESCRIPTOR_LABELS = {"graph2vec": "Graph2Vec", "eigencentrality": "Eigenvector centrality", "spectral": "Spectral descriptor"}
GRAPH_TYPE_COLOURS = {"threshold": "#7570b3", "mst": "#1b9e77", "tmfg": "#d95f02"}
STRATEGY_COLOURS = {config.BUY_AND_HOLD: "#7570b3", "Regime-timed": "#1b9e77", config.REGIME_FORECAST_STRATEGY: "#d95f02",
                    **{constant: colour for constant, colour in zip(config.CONSTANT_EXPOSURE_STRATEGIES.values(), ("#1b9e77", "#d95f02"))}}
STRATEGY_LINESTYLES = {name: "--" if name in config.CONSTANT_EXPOSURE_STRATEGIES.values() else "-" for name in config.PORTFOLIO_STRATEGY_NAMES}


# Set global rcParams so every manuscript figure shares one visual language: drawn at its printed width in a sans-serif face (Arial, as Springer's artwork guidance asks, falling back to similar faces where it is not installed), text at 7 to 9 pt at that size, and laid out by constrained layout so the saved file is exactly the figure size.
def set_publication_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.titleweight": "normal",
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": SMALL_FONT_SIZE,
        "ytick.labelsize": SMALL_FONT_SIZE,
        "legend.fontsize": SMALL_FONT_SIZE,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "axes.linewidth": 0.6,
        "axes.titlepad": 4,
        "axes.labelpad": 2,
        "lines.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": config.FIGURE_DPI,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.04,
        "figure.constrained_layout.w_pad": 0.04,
    })


# Turn an axis's light grid on or off (spines and ticks follow set_publication_style).
def style_axis(ax, grid=False):
    if grid:
        ax.grid(True, alpha=0.15, color="black", linewidth=0.4)
    else:
        ax.grid(False)

    return ax


# Label a panel with a bold lower-case letter at its top-left corner, as the captions refer to it.
def add_panel_label(ax, letter):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=PANEL_LABEL_SIZE)


# Shade a list of (start, end) x-intervals on an axis with one consistent colour and alpha.
def shade_periods(ax, intervals, colour, alpha=0.2, zorder=1):
    for start, end in intervals:
        ax.axvspan(start, end, color=colour, alpha=alpha, linewidth=0, zorder=zorder)


# Format an axis's x-axis as year ticks every second year, matching every date-indexed figure.
def format_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# Draw the dashed train/test split marker at split_date and return its legend handle.
def mark_split(ax, split_date):
    ax.axvline(split_date, color="#333333", linestyle="--", linewidth=0.9, zorder=5)
    return Line2D([0], [0], color="#333333", linestyle="--", linewidth=0.9, label="Train/test split")


# Annotate a {date_str: label} dict (e.g. config.CRISIS_EVENTS) as labelled dotted vertical lines within [x_min, x_max].
def annotate_events(ax, events, x_min, x_max):
    y_top = ax.get_ylim()[1]

    for date_str, label in events.items():
        event_date = pd.to_datetime(date_str)

        if x_min <= event_date <= x_max:
            ax.axvline(event_date, color="#444444", linestyle=":", linewidth=0.8, alpha=0.85, zorder=4)
            ax.text(event_date, y_top * 0.97, f" {label}", rotation=90, fontsize=SMALL_FONT_SIZE - 1, color="#444444", va="top", ha="right")


# Save a figure into results/figures/{subfolder}/ (name may add a subfolder of its own) at the given resolution, exactly at its figure size, and close it.
def save_to_subfolder(fig, subfolder, name, dpi=config.FIGURE_DPI):
    path = config.FIGURES_DIR / subfolder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, transparent=False)
    plt.close(fig)


# A legend with its entries in a row above the right end of an axis, clear of the data and of the panel letter.
def legend_above(ax, handles):
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles), borderaxespad=0.6)


# The time span each dated window covers when shaded, from its own date to the next window's, the last one extended by the step before it so it stays visible.
def _date_spans(dates):
    dates = list(dates)
    last_step = dates[-1] - dates[-2] if len(dates) > 1 else pd.Timedelta(days=1)
    return list(zip(dates, dates[1:] + [dates[-1] + last_step]))


# Regime legend entries: one shaded patch per regime, plus the train/test split line.
def regime_handles():
    return [Patch(color=REGIME_PALETTE[r], alpha=0.5, label=config.REGIME_NAMES[r]) for r in range(config.N_REGIMES)] + [Line2D([0], [0], color="#333333", linestyle="--", linewidth=0.9, label="Train/test split")]


# Draw a regime timeline on an axis: each window shaded by its regime rank (lighter in the test period) behind its mean correlation, train and test on one continuous date axis; returns the legend handles.
def draw_regime_timeline(ax, train_dates, train_mu, train_labels, test_dates, test_mu, test_labels):
    dates = list(train_dates) + list(test_dates)
    ranks = list(train_labels) + list(test_labels)
    spans = _date_spans(dates)

    for r in range(config.N_REGIMES):
        for alpha, windows in ((0.5, range(len(train_dates))), (0.25, range(len(train_dates), len(dates)))):
            shade_periods(ax, [spans[i] for i in windows if ranks[i] == r], REGIME_PALETTE[r], alpha=alpha)

    ax.plot(train_dates, train_mu, color="black", linewidth=1.0)
    ax.plot(test_dates, test_mu, color="black", linewidth=1.0)
    mark_split(ax, pd.Timestamp(config.TRAIN_TEST_CUTOFF))
    ax.set_ylabel(r"Mean correlation $\mu$")
    ax.set_xlim(dates[0], spans[-1][1])
    format_date_axis(ax)
    style_axis(ax, grid=True)
    return regime_handles()


# Draw the VIX on an axis with the predicted Crisis windows shaded (lighter in the test period), the VIX threshold dashed and the named stress events dotted; returns the legend handles.
def draw_vix_overlay(ax, train_dates, train_validation, test_dates, test_validation):
    ax.plot(train_dates, train_validation["series_eod"], color=VIX_COLOUR, linewidth=1.0, zorder=3)
    ax.plot(test_dates, test_validation["series_eod"], color=VIX_COLOUR, linewidth=1.0, zorder=3)
    ax.axhline(config.VIX_CRISIS_THRESHOLD, color=NEUTRAL_COLOURS["reference"], linestyle="--", linewidth=0.8, zorder=2)
    dates = train_dates.append(test_dates)
    crisis = np.concatenate([train_validation["pred"], test_validation["pred"]]).astype(bool)
    spans = _date_spans(dates)

    for alpha, windows in ((0.5, range(len(train_dates))), (0.25, range(len(train_dates), len(dates)))):
        shade_periods(ax, [spans[i] for i in windows if crisis[i]], REGIME_PALETTE[2], alpha=alpha)

    ax.set_ylabel("VIX")
    ax.set_xlim(dates[0], spans[-1][1])
    annotate_events(ax, config.CRISIS_EVENTS, dates[0], dates[-1])
    mark_split(ax, pd.Timestamp(config.TRAIN_TEST_CUTOFF))
    format_date_axis(ax)
    style_axis(ax, grid=True)
    return [Line2D([0], [0], color=VIX_COLOUR, linewidth=1.0, label="VIX at window end"), Line2D([0], [0], color=NEUTRAL_COLOURS["reference"], linewidth=0.8, linestyle="--", label=f"VIX = {config.VIX_CRISIS_THRESHOLD:g}"),
            Patch(facecolor=REGIME_PALETTE[2], alpha=0.5, label="Crisis regime"), Line2D([0], [0], color="#333333", linestyle="--", linewidth=0.9, label="Train/test split")]


# Draw one empirical transition matrix on an axis, each cell's probability printed in a colour that reads against its shade and each row labelled with its regime's number of windows; returns the image for a shared colour bar.
def draw_transition_matrix(ax, P, occupancy, title):
    cmap = plt.get_cmap(TRANSITION_CMAP)
    image = ax.imshow(P, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(config.N_REGIMES), config.REGIME_NAMES)
    ax.set_yticks(range(config.N_REGIMES), [f"{name}\n(n = {occupancy[r]})" for r, name in enumerate(config.REGIME_NAMES)])
    ax.tick_params(length=0)
    ax.set_xlabel("Next window")
    ax.set_title(title)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for r in range(config.N_REGIMES):
        for c in range(config.N_REGIMES):
            value = P[r, c]
            red, green, blue = cmap(0.0 if np.isnan(value) else value)[:3]
            ax.text(c, r, "--" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", fontsize=FONT_SIZE, color="white" if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5 else "black")

    return image
