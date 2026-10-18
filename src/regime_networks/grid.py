import re
from pathlib import Path

import pandas as pd

from regime_networks import config

GRID_TAG_PATTERN = re.compile(r"(?P<index_code>[A-Za-z0-9]+)_(?P<window_length>\d+)d(?:_(?P<graph_type>[a-z]+))?")


# Name of one sweep grid point, "{INDEX}_{T}d" plus "_{graph_type}" for descriptors built on a network.
def grid_tag(index_code, window_length, graph_type=None):
    return f"{index_code}_{window_length}d" + (f"_{graph_type}" if graph_type else "")


# Stem of one grid point's per-descriptor outputs (labels file and model bundle), "{grid_tag}_{descriptor}".
def output_stem(index_code, window_length, graph_type, descriptor):
    return f"{grid_tag(index_code, window_length, graph_type)}_{descriptor}"


# Split an output stem into descriptor, index code, window length and graph construction (None for the spectral descriptor), refusing any stem the naming scheme could not have produced for a configured descriptor.
def parse_output_stem(stem):
    descriptor = next((d for d in config.DESCRIPTORS if stem.endswith(f"_{d}")), None)
    match = GRID_TAG_PATTERN.fullmatch(stem[: -len(descriptor) - 1]) if descriptor else None

    if match is None or match["graph_type"] not in config.DESCRIPTOR_GRAPH_TYPES[descriptor]:
        raise ValueError(f"{stem!r} is not '{{INDEX}}_{{T}}d[_{{graph_type}}]_{{descriptor}}' for a configured descriptor and one of its constructions ({config.DESCRIPTOR_GRAPH_TYPES})")

    return {"descriptor": descriptor, "index_code": match["index_code"], "window_length": int(match["window_length"]), "graph_type": match["graph_type"]}


# Grid points (index_code, window_length, graph_type) a sweep of one descriptor visits, in sweep order: markets as in INDEX_CODES, window lengths ascending, constructions as in DESCRIPTOR_GRAPH_TYPES.
def descriptor_grid(descriptor, index_codes=None, window_lengths=None):
    index_codes = config.INDEX_CODES if index_codes is None else index_codes
    window_lengths = config.WINDOW_LENGTHS if window_lengths is None else window_lengths
    return [(index_code, window_length, graph_type) for index_code in index_codes for window_length in window_lengths for graph_type in config.DESCRIPTOR_GRAPH_TYPES[descriptor]]


# Position of a grid point in the order a full run writes it (descriptors as in DESCRIPTORS, markets as in INDEX_CODES, window lengths ascending, constructions as in GRAPH_TYPES; an unset or unknown value sorts last), so merged outputs keep one order however they were run.
def grid_position(index_code=None, window_length=None, graph_type=None, descriptor=None):
    return tuple(order.index(value) if value in order else len(order) for value, order in ((descriptor, config.DESCRIPTORS), (index_code, config.INDEX_CODES), (window_length, config.WINDOW_LENGTHS), (graph_type, config.GRAPH_TYPES)))


# Write a results table of grid-point rows after replacing any earlier rows for the same grid points, in full-run grid order and with the new rows' columns, so a partial re-run keeps every other grid point's rows and writes the same file as a clean run.
def merge_grid_results(path, rows):
    table = pd.DataFrame(rows)
    keys = [column for column in ("index_code", "window_length", "graph_type") if column in table]

    if path.exists():
        previous = pd.read_csv(path, float_precision="round_trip").reindex(columns=table.columns)
        table = pd.concat([previous[~previous.set_index(keys).index.isin(table.set_index(keys).index)], table], ignore_index=True)

    table.iloc[sorted(range(len(table)), key=lambda i: grid_position(**table.iloc[i][keys].to_dict()))].to_csv(path, index=False, lineterminator="\n")


# The headline configuration's grid point (index_code, window_length, graph_type, descriptor) in one market, the primary market by default.
def headline_grid_point(index_code=None):
    return (config.PRIMARY_INDEX if index_code is None else index_code, config.HEADLINE_WINDOW[0], config.HEADLINE_GRAPH_TYPE, config.HEADLINE_DESCRIPTOR)


# Per-window labels file of one grid point and descriptor, in directory (outputs/labels/ by default).
def labels_path(index_code, window_length, graph_type, descriptor, directory=None):
    return Path(config.LABELS_DIR if directory is None else directory) / f"{output_stem(index_code, window_length, graph_type, descriptor)}.csv"


# Model bundle of one grid point and descriptor, in directory (outputs/models/ by default).
def bundle_path(index_code, window_length, graph_type, descriptor, directory=None):
    return Path(config.MODELS_DIR if directory is None else directory) / f"{output_stem(index_code, window_length, graph_type, descriptor)}.joblib"


# Spectral feature table of one market and window length (every window, train then test), written by the sweep and read back by model selection.
def spectral_features_path(index_code, window_length, directory=None):
    return Path(config.OUTPUTS_DIR if directory is None else directory) / f"{index_code}_spectral_features_{window_length}d.pkl"
