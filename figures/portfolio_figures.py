import argparse

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

from regime_networks import config
from regime_networks.grid import grid_tag
from regime_networks.portfolio import drawdown_series, regime_spans
from regime_networks.plotting import (set_publication_style, style_axis, format_date_axis, add_panel_label, shade_periods, save_to_subfolder, FIGURE_WIDTH, PANEL_LETTERS,
                                      REGIME_PALETTE, NEUTRAL_COLOURS, STRATEGY_COLOURS, STRATEGY_LINESTYLES)

FOLDER = "portfolio"
LINEWIDTH = 0.9
REGIME_SHADE_ALPHA = 0.25


# Spans of one regime in force on the daily index, each stretched to the next trading day so a one-day span stays visible.
def _visible_spans(daily, regime_name):
    days = daily.index
    return [(start, days[days.get_loc(end) + 1] if end != days[-1] else end + pd.Timedelta(days=1)) for start, end in regime_spans(daily["regime"], regime_name)]


# Cumulative return (log scale) and drawdown, net of transaction costs, of every strategy in one market and construction on a shared date axis, the constant-exposure benchmarks dashed in their timed strategy's colour, with Transitional and Crisis periods shaded and one legend above both panels.
def equity_drawdown_figure(daily):
    strategies = {name: daily[f"{name} return"] for name in config.PORTFOLIO_STRATEGY_NAMES}
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 4.1), sharex=True)
    handles = [axes[0].plot((1 + ret).cumprod(), label=name, color=STRATEGY_COLOURS[name], linestyle=STRATEGY_LINESTYLES[name], linewidth=LINEWIDTH, zorder=3)[0] for name, ret in strategies.items()]
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 1.5, 2.0, 3.0, 6.0, 8.0)))
    axes[0].yaxis.set_minor_locator(mticker.NullLocator())
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axes[0].set_ylabel("Cumulative return")

    for name, ret in strategies.items():
        axes[1].plot(drawdown_series(ret), color=STRATEGY_COLOURS[name], linestyle=STRATEGY_LINESTYLES[name], linewidth=LINEWIDTH, zorder=3)

    axes[1].axhline(0.0, color=NEUTRAL_COLOURS["reference"], linewidth=0.6, zorder=2)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    axes[1].set_ylabel("Drawdown")

    for ax, letter in zip(axes, PANEL_LETTERS):
        for rank in (1, 2):
            shade_periods(ax, _visible_spans(daily, config.REGIME_NAMES[rank]), REGIME_PALETTE[rank], alpha=REGIME_SHADE_ALPHA, zorder=0)

        add_panel_label(ax, letter)
        format_date_axis(ax)
        style_axis(ax, grid=True)

    shading = [Patch(color=REGIME_PALETTE[rank], alpha=REGIME_SHADE_ALPHA, label=f"{config.REGIME_NAMES[rank]} regime") for rank in (1, 2)]
    fig.legend(handles=handles + shading, loc="outside upper center", ncol=3)
    axes[-1].set_xlabel("Date")
    return fig


# Draw each market's portfolio figure under the headline construction from outputs/portfolio/{tag}_daily.csv into results/figures/portfolio/; the tables cover every construction.
def main(dpi=config.FIGURE_DPI):
    set_publication_style()
    tags = [grid_tag(index_code, config.HEADLINE_WINDOW[0], config.HEADLINE_GRAPH_TYPE) for index_code in config.INDEX_CODES]

    for tag in tags:
        daily = pd.read_csv(config.OUTPUTS_DIR / "portfolio" / f"{tag}_daily.csv", index_col="date", parse_dates=["date"])
        save_to_subfolder(equity_drawdown_figure(daily), FOLDER, f"portfolio_equity_drawdown_{tag}.png", dpi=dpi)

    print(f"drew {len(tags)} portfolio figures into {config.FIGURES_DIR / FOLDER}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw each market's portfolio equity and drawdown figure under the headline construction from outputs/portfolio/{tag}_daily.csv.")
    parser.add_argument("--dpi", type=int, default=config.FIGURE_DPI, help=f"Export resolution (default: {config.FIGURE_DPI})")
    main(parser.parse_args().dpi)
