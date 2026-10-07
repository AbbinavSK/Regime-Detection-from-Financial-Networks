import argparse

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from regime_networks import config
from regime_networks.portfolio import drawdown_series, regime_spans
from regime_networks.plotting import (set_publication_style, style_axis, format_date_axis, add_panel_label, shade_periods, save_to_subfolder,
                                      REGIME_PALETTE, NEUTRAL_COLOURS, STRATEGY_COLOURS)

FOLDER = "portfolio"
LINEWIDTH = 2.0
GRID_ALPHA = 0.08
REGIME_SHADE_ALPHA = 0.25


# Spans of one regime in force on the daily index, each stretched to the next trading day so a one-day span stays visible.
def _visible_spans(daily, regime_name):
    days = daily.index
    return [(start, days[days.get_loc(end) + 1] if end != days[-1] else end + pd.Timedelta(days=1)) for start, end in regime_spans(daily["regime"], regime_name)]


# Cumulative return (log scale) and drawdown of every strategy in one market on a shared date axis, with Transitional and Crisis periods shaded.
def equity_drawdown_figure(daily):
    strategies = {name: daily[f"{name} return"] for name in config.PORTFOLIO_STRATEGY_NAMES}
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    handles = [axes[0].plot((1 + ret).cumprod(), label=name, color=STRATEGY_COLOURS[name], linewidth=LINEWIDTH, zorder=3)[0] for name, ret in strategies.items()]
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Cumulative return")

    for name, ret in strategies.items():
        drawdown = drawdown_series(ret)
        axes[1].fill_between(drawdown.index, drawdown.values, 0, color=STRATEGY_COLOURS[name], alpha=0.25, linewidth=0, zorder=1)
        axes[1].plot(drawdown, color=STRATEGY_COLOURS[name], linewidth=LINEWIDTH, zorder=3)

    axes[1].axhline(0.0, color=NEUTRAL_COLOURS["reference"], linewidth=0.8, zorder=2)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    axes[1].set_ylabel("Drawdown")

    for ax, letter in zip(axes, "AB"):
        for rank in (1, 2):
            shade_periods(ax, _visible_spans(daily, config.REGIME_NAMES[rank]), REGIME_PALETTE[rank], alpha=REGIME_SHADE_ALPHA, zorder=0)

        add_panel_label(ax, letter, x=-0.09, y=1.03)
        format_date_axis(ax)
        style_axis(ax, grid=True)
        ax.grid(True, alpha=GRID_ALPHA)

    axes[0].legend(handles=handles, loc="upper left", frameon=True, facecolor="white", edgecolor="lightgray", fontsize=8, ncol=1)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return fig


# Draw each market's portfolio figure from outputs/portfolio/{INDEX}_daily.csv into results/figures/portfolio/.
def main(dpi=config.FIGURE_DPI):
    set_publication_style()

    for index_code in config.INDEX_CODES:
        daily = pd.read_csv(config.OUTPUTS_DIR / "portfolio" / f"{index_code}_daily.csv", index_col="date", parse_dates=["date"])
        save_to_subfolder(equity_drawdown_figure(daily), FOLDER, f"portfolio_equity_drawdown_{index_code}.png", dpi=dpi)

    print(f"drew the portfolio figures of {len(config.INDEX_CODES)} markets into {config.FIGURES_DIR / FOLDER}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw each market's portfolio equity and drawdown figure from outputs/portfolio/{INDEX}_daily.csv.")
    parser.add_argument("--dpi", type=int, default=config.FIGURE_DPI, help=f"Export resolution (default: {config.FIGURE_DPI})")
    main(parser.parse_args().dpi)
