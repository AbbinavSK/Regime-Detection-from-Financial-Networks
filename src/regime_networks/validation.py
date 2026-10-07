import numpy as np
import pandas as pd

from regime_networks import config
from regime_networks.data import check_data_files

PER_WINDOW_KEYS = ["window_dates", "correlation_matrices", "mu_values", *config.GRAPH_KEYS.values()]


# Split a network-data dict into train and test windows by window end date.
def split_train_test(data, cutoff=config.TRAIN_TEST_CUTOFF):
    cutoff = pd.Timestamp(cutoff)
    is_train = pd.DatetimeIndex(pd.to_datetime(data["window_dates"])) <= cutoff
    train = {k: v for k, v in data.items() if k not in PER_WINDOW_KEYS}
    test = dict(train)

    for key in PER_WINDOW_KEYS:
        values = data[key]
        train[key] = [v for v, m in zip(values, is_train) if m]
        test[key] = [v for v, m in zip(values, is_train) if not m]

    return train, test


# Load the VIX close series, failing with guidance on obtaining the data if the file is missing.
def load_vix(path=None):
    path = config.VIX_PATH if path is None else path
    check_data_files([path])
    return pd.read_csv(path, index_col=0, parse_dates=True).squeeze().sort_index()


# Compute precision, recall, and F1 for a binary prediction.
def binary_f1(pred, actual):
    pred, actual = np.asarray(pred), np.asarray(actual)
    tp = int(((pred == 1) & (actual == 1)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# Value of a daily series (e.g. VIX) as of each date, forward-filling from the latest observation on or before it.
def align_to_dates(series, dates):
    dates_idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return series.reindex(series.index.union(dates_idx)).sort_index().ffill().reindex(dates_idx)


# Score predicted crisis labels against a validation series like VIX.
def validate_regime(labels, dates, series, threshold, crisis_id):
    series_eod = align_to_dates(series, dates).values
    crisis_point = (series_eod > threshold).astype(int)
    pred = (np.asarray(labels) == crisis_id).astype(int)
    return {"series_eod": series_eod, "pred": pred, "point": binary_f1(pred, crisis_point)}
