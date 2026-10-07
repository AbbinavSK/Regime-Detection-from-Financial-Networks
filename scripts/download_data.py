import argparse

from regime_networks import config
from regime_networks.constituents import fetch_constituents
from regime_networks.data import run_index, combine_index_panels, download_vix

TARGETS = config.INDEX_CODES + ["VIX"]


# Download, screen and save one target: an index panel (a composite such as FTSE350 is built from its component indices, screened again after intersecting their dates) or the VIX series.
def download(target):
    if target == "VIX":
        download_vix(config.START_DATE, config.END_DATE_INCLUSIVE)
    elif target in config.INDEX_COMPONENTS:
        panels = [run_index(code, fetch_constituents(code), config.START_DATE, config.END_DATE_INCLUSIVE) for code in config.INDEX_COMPONENTS[target]]
        combine_index_panels(target, panels, config.START_DATE, config.END_DATE_INCLUSIVE)
    else:
        run_index(target, fetch_constituents(target), config.START_DATE, config.END_DATE_INCLUSIVE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download constituent prices from Yahoo Finance, screen them, and save the panels to data/.")
    parser.add_argument("--target", choices=TARGETS + ["all"], default="all", help="Index (or VIX) to download (default: all)")
    args = parser.parse_args()

    for target in TARGETS if args.target == "all" else [args.target]:
        download(target)
