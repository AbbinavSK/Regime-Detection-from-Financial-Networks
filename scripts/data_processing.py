import argparse

from regime_networks import config
from regime_networks.data import check_data_files, cleaned_prices_path
from regime_networks.returns import load_log_returns, rolling_correlations
from regime_networks.networks import build_networks, save_networks


# Build and save every index's rolling-correlation network data across all configured window configs, after checking every cleaned panel is present.
def main():
    check_data_files([cleaned_prices_path(index_code) for index_code in config.INDEX_CODES])

    for index_code in config.INDEX_CODES:
        log_returns = load_log_returns(cleaned_prices_path(index_code))

        for window_length, window_shift in config.WINDOW_CONFIGS:
            correlation_matrices, window_dates, stock_names = rolling_correlations(log_returns, window_length, window_shift)
            print(f"{index_code} T={window_length}: {correlation_matrices[0].shape} correlation matrix, {len(correlation_matrices)} windows")

            save_networks(build_networks(correlation_matrices, stock_names, window_length, window_dates, index_code))


if __name__ == "__main__":
    argparse.ArgumentParser(description="Build the correlation matrices and threshold, MST-threshold and TMFG-threshold networks for every index and window length into outputs/ (slow).").parse_args()
    main()
