import hashlib
import time
from pathlib import Path
from typing import Mapping, Sequence

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf

from regime_networks import config

NO_PRICES_REASON = "no_close_prices_returned"
DOWNLOAD_ERROR_REASON = "download_error"
MANIFEST_NAME = "MANIFEST.csv"


# Reason code recorded in the exclusions files for each screening rule, keyed by rule; the return-jump code carries the configured threshold.
def exclusion_reasons() -> dict[str, str]:
    return {"nonpositive": "nonpositive_price", "incomplete": "incomplete_price_history", "jump": f"absolute_log_return_above_{config.MAX_ABS_LOG_RETURN}", "flat_run": "flat_price_run"}


# Path of one index's cleaned price panel, the file every analysis reads, in directory (data/ by default).
def cleaned_prices_path(index_code: str, directory: Path | str | None = None) -> Path:
    return Path(config.DATA_DIR if directory is None else directory) / f"{index_code}_AdjClose_Cleaned.csv"


# Order sector-prefixed stock columns deterministically, refusing any ticker code that appears twice (under one sector label or two).
def order_columns_by_sector(frame: pd.DataFrame) -> pd.DataFrame:
    codes = pd.Index([name.split("_", 1)[-1] for name in frame.columns])
    duplicate_codes = codes[codes.duplicated()].unique().tolist()

    if duplicate_codes:
        raise ValueError(f"The same ticker code appears more than once among the stock columns: {duplicate_codes}. Two tickers mapped to one '{{sector}}_{{code}}' label, or one company is listed in two combined panels, possibly under different sector names.")

    return frame.reindex(columns=sorted(frame.columns, key=lambda name: (name.split("_", 1)[0], name)))


# Return the official exchange sessions within the requested period (get_calendar already bounds .sessions to [start, end]).
def market_sessions(index_code: str, start_date: str, end_date_inclusive: str) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar(config.MARKET_CALENDARS[index_code], start=start_date, end=end_date_inclusive)
    return calendar.sessions.tz_localize(None).normalize()


# Return the exclusive end date for yfinance downloads, given an inclusive end date.
def _exclusive_end_date(end_date_inclusive: str) -> str:
    return (pd.Timestamp(end_date_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


# Return one Close price column for each requested ticker.
def _extract_close_prices(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=tickers)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        elif "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", axis=1, level=-1)
        else:
            return pd.DataFrame(index=raw.index, columns=tickers)
    elif "Close" in raw.columns:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        return pd.DataFrame(index=raw.index, columns=tickers)

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    return close.reindex(columns=tickers)


# Per-ticker error messages from yfinance's most recent download, stopping if this yfinance version no longer records them (a rate limit would otherwise look like a delisting).
def _provider_errors() -> dict[str, str]:
    if not hasattr(yf.shared, "_ERRORS"):
        raise RuntimeError(f"yfinance {getattr(yf, '__version__', '?')} no longer records per-ticker download errors in yf.shared._ERRORS, which the download needs to tell a rate limit from a delisted ticker; install the version pinned in environment.yml.")

    return {str(ticker).upper(): str(message) for ticker, message in yf.shared._ERRORS.items()}


# Download one batch of tickers, retrying immediately if the call itself raises; returns (close_prices, per-ticker provider errors, None) or (None, {}, last_exception).
def _download_batch(tickers: Sequence[str], start_date: str, end_date_inclusive: str) -> tuple[pd.DataFrame | None, dict[str, str], Exception | None]:
    last_exc = None

    for _ in range(config.DOWNLOAD_BATCH_ATTEMPTS):
        try:
            raw = yf.download(tickers=tickers, start=start_date, end=_exclusive_end_date(end_date_inclusive), auto_adjust=True, progress=False, threads=True, timeout=config.DOWNLOAD_TIMEOUT_SECONDS)
            return _extract_close_prices(raw, tickers), _provider_errors(), None
        except Exception as exc:
            last_exc = exc

    return None, {}, last_exc


# Move every ticker with at least one price into resolved and return the others with the reason they are missing (including the provider's error message when it recorded one).
def _collect_resolved(close: pd.DataFrame, tickers: Sequence[str], errors: Mapping[str, str], resolved: dict[str, pd.Series]) -> dict[str, str]:
    missing = {}

    for ticker in tickers:
        series = close[ticker] if ticker in close else pd.Series(dtype=float)

        if series.notna().any():
            resolved[ticker] = series
        else:
            missing[ticker] = f"{NO_PRICES_REASON}: {errors[ticker.upper()]}" if ticker.upper() in errors else NO_PRICES_REASON

    return missing


# Download each still-missing ticker on its own, waiting between rounds so rate limits can clear; returns the tickers that are still missing with their latest reason.
def _retry_missing(missing: dict[str, str], start_date: str, end_date_inclusive: str, resolved: dict[str, pd.Series]) -> dict[str, str]:
    missing = dict(missing)

    for attempt in range(1, config.DOWNLOAD_TICKER_ATTEMPTS + 1):
        if not missing:
            break

        print(f"Retrying {len(missing)} tickers one at a time (round {attempt}/{config.DOWNLOAD_TICKER_ATTEMPTS})")
        time.sleep(config.DOWNLOAD_RETRY_WAIT_SECONDS)

        for ticker in list(missing):
            close, errors, exc = _download_batch([ticker], start_date, end_date_inclusive)
            still_missing = {ticker: f"{DOWNLOAD_ERROR_REASON}: {type(exc).__name__}"} if close is None else _collect_resolved(close, [ticker], errors, resolved)

            if ticker in still_missing:
                missing[ticker] = still_missing[ticker]
            else:
                del missing[ticker]

    return missing


# Download prices in batches of DOWNLOAD_BATCH_SIZE, retry every ticker that returned nothing on its own, and refuse to continue if any still failed for a network or rate-limit reason (which would silently shrink the universe); returns the Close-price panel and the excluded tickers.
def download_prices(ticker_map: Mapping[str, str], start_date: str, end_date_inclusive: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = list(ticker_map)
    resolved: dict[str, pd.Series] = {}
    missing: dict[str, str] = {}
    batches = [tickers[offset : offset + config.DOWNLOAD_BATCH_SIZE] for offset in range(0, len(tickers), config.DOWNLOAD_BATCH_SIZE)]

    for number, batch in enumerate(batches, start=1):
        print(f"Downloading batch {number}/{len(batches)} ({len(batch)} tickers)")
        close, errors, batch_exc = _download_batch(batch, start_date, end_date_inclusive)
        missing.update({ticker: f"{DOWNLOAD_ERROR_REASON}: {type(batch_exc).__name__}" for ticker in batch} if close is None else _collect_resolved(close, batch, errors, resolved))

    missing = _retry_missing(missing, start_date, end_date_inclusive, resolved)
    transient = {ticker: reason for ticker, reason in missing.items() if reason.startswith(DOWNLOAD_ERROR_REASON) or any(marker in reason for marker in config.DOWNLOAD_TRANSIENT_ERRORS)}

    if transient:
        raise RuntimeError(f"{len(transient)} tickers still failed with network or rate-limit errors after {config.DOWNLOAD_TICKER_ATTEMPTS} single-ticker retries; wait and download again rather than freeze a smaller universe: {transient}")

    if not resolved:
        raise ValueError("The price provider returned no usable Close-price series.")

    exclusions = [{"ticker": ticker, "stock": ticker_map[ticker], "reason": reason} for ticker, reason in missing.items()]

    panel = pd.concat([resolved[ticker] for ticker in tickers if ticker in resolved], axis=1)
    panel.columns = [ticker_map[ticker] for ticker in tickers if ticker in resolved]
    panel.index.name = "Date"
    panel = panel.sort_index()
    return order_columns_by_sector(panel), pd.DataFrame(exclusions, columns=["ticker", "stock", "reason"])


# Length of the longest run of consecutive True values in each column of a boolean frame.
def _longest_true_run(flags: pd.DataFrame) -> pd.Series:
    return flags.apply(lambda column: column.groupby((~column).cumsum()).sum().max())


# Keep official exchange sessions and stocks with complete, positive, non-stale prices: drop market-wide stale dates, then stocks with an implausible one-day jump or a flat-price run as long as the shortest rolling window (whose correlation would be undefined).
def screen_price_panel(raw_prices: pd.DataFrame, index_code: str, start_date: str,
                       end_date_inclusive: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = raw_prices.copy()
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index, errors="coerce")).tz_localize(None).normalize()

    if prices.index.hasnans:
        raise ValueError(f"{index_code}: the provider returned an unparseable date.")

    if prices.index.has_duplicates:
        raise ValueError(f"{index_code}: the provider returned duplicate dates.")

    prices = prices.sort_index().loc[start_date:end_date_inclusive]

    if prices.empty:
        raise ValueError(f"{index_code}: no provider dates fall within the study period.")

    sessions = market_sessions(index_code, start_date, end_date_inclusive)
    session_rows = prices.index.isin(sessions)
    excluded_non_session_rows = int((~session_rows).sum())
    prices = prices.loc[session_rows]

    reasons = exclusion_reasons()
    prices = prices.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    nonpositive = (prices <= 0).any(axis=0)
    incomplete = prices.isna().any(axis=0) & ~nonpositive
    exclusions = pd.concat([pd.DataFrame({"ticker": "", "stock": prices.columns[nonpositive], "reason": reasons["nonpositive"]}),
                            pd.DataFrame({"ticker": "", "stock": prices.columns[incomplete], "reason": reasons["incomplete"]})], ignore_index=True)
    cleaned = prices.loc[:, ~(nonpositive | incomplete)]

    returns = np.log(cleaned).diff()
    zero_return_share = (returns.abs() < config.ZERO_RETURN_TOLERANCE).mean(axis=1)
    stale_rows = zero_return_share.ge(config.STALE_ROW_THRESHOLD)
    excluded_stale_rows = int(stale_rows.sum())
    cleaned = cleaned.loc[~stale_rows]

    returns = np.log(cleaned).diff().iloc[1:]

    if len(returns) < config.MIN_RETURN_ROWS:
        raise ValueError(f"{index_code}: need at least {config.MIN_RETURN_ROWS} log-return rows, found {len(returns)}.")

    valid_jump = returns.abs().le(config.MAX_ABS_LOG_RETURN).all(axis=0)
    jump_exclusions = pd.DataFrame({"ticker": "", "stock": cleaned.columns[~valid_jump], "reason": reasons["jump"]})
    exclusions = pd.concat([exclusions, jump_exclusions], ignore_index=True)
    cleaned = cleaned.loc[:, valid_jump]

    returns = np.log(cleaned).diff().iloc[1:]
    max_flat_run = min(window_length for window_length, _ in config.WINDOW_CONFIGS)
    valid_run = _longest_true_run(returns.abs().lt(config.ZERO_RETURN_TOLERANCE)).lt(max_flat_run)
    run_exclusions = pd.DataFrame({"ticker": "", "stock": cleaned.columns[~valid_run], "reason": reasons["flat_run"]})
    exclusions = pd.concat([exclusions, run_exclusions], ignore_index=True)
    cleaned = cleaned.loc[:, valid_run]

    cleaned = order_columns_by_sector(cleaned)
    return_rows = len(cleaned) - 1

    if cleaned.shape[1] < config.MIN_STOCKS:
        raise ValueError(f"{index_code}: need at least {config.MIN_STOCKS} retained stocks, found {cleaned.shape[1]}.")

    summary = pd.DataFrame(
        [
            {
                "index_code": index_code,
                "requested_start": start_date,
                "requested_end": end_date_inclusive,
                "observed_start": cleaned.index.min().strftime("%Y-%m-%d"),
                "observed_end": cleaned.index.max().strftime("%Y-%m-%d"),
                "price_rows": len(cleaned),
                "return_rows": return_rows,
                "retained_stocks": cleaned.shape[1],
                "excluded_non_session_rows": excluded_non_session_rows,
                "excluded_stale_rows": excluded_stale_rows,
            }
        ]
    )
    return cleaned, exclusions, summary


# Download, validate and report one index price panel, saving raw and cleaned prices plus the constituent snapshot, per-stock exclusion reasons and screening summary.
def run_index(index_code: str, ticker_map: Mapping[str, str], start_date: str, end_date_inclusive: str,
              directory: Path | str | None = None) -> pd.DataFrame:
    directory = Path(config.DATA_DIR if directory is None else directory)
    directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"ticker": list(ticker_map), "stock": list(ticker_map.values())}).to_csv(directory / f"{index_code}_constituents.csv", index=False, lineterminator="\n")
    raw, download_exclusions = download_prices(ticker_map, start_date, end_date_inclusive)
    raw.to_csv(directory / f"{index_code}_AdjClose_Raw.csv", index=True, lineterminator="\n")

    cleaned, quality_exclusions, summary = screen_price_panel(raw, index_code, start_date, end_date_inclusive)
    cleaned.to_csv(cleaned_prices_path(index_code, directory), index=True, lineterminator="\n")

    exclusions = pd.concat([download_exclusions, quality_exclusions], ignore_index=True)
    exclusions.to_csv(directory / f"{index_code}_exclusions.csv", index=False, lineterminator="\n")
    summary.to_csv(directory / f"{index_code}_screening_summary.csv", index=False, lineterminator="\n")

    details = summary.iloc[0]
    print(
        f"{index_code}: {details.retained_stocks} stocks, {details.price_rows} prices, "
        f"{details.observed_start} to {details.observed_end}"
    )
    print(f"Excluded non-session dates: {details.excluded_non_session_rows}")
    print(f"Excluded stale-price dates: {details.excluded_stale_rows}")
    print(f"Excluded tickers: {len(exclusions)}")
    return cleaned


# Intersect and validate already-clean sub-index panels as one final panel, saving its cleaned prices, exclusion reasons and screening summary.
def combine_index_panels(index_code: str, panels: Sequence[pd.DataFrame], start_date: str,
                         end_date_inclusive: str, directory: Path | str | None = None) -> pd.DataFrame:
    if len(panels) < 2:
        raise ValueError("At least two sub-index panels are required for combination.")

    combined = pd.concat(panels, axis=1, join="inner")
    cleaned, exclusions, summary = screen_price_panel(combined, index_code, start_date, end_date_inclusive)

    directory = Path(config.DATA_DIR if directory is None else directory)
    directory.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(cleaned_prices_path(index_code, directory), index=True, lineterminator="\n")
    exclusions.to_csv(directory / f"{index_code}_exclusions.csv", index=False, lineterminator="\n")
    summary.to_csv(directory / f"{index_code}_screening_summary.csv", index=False, lineterminator="\n")
    print(f"{index_code}: {cleaned.shape[1]} stocks after date intersection")
    print(f"Excluded non-session dates: {summary.loc[0, 'excluded_non_session_rows']}")
    print(f"Excluded stale-price dates: {summary.loc[0, 'excluded_stale_rows']}")
    print(f"Excluded tickers: {len(exclusions)}")
    return cleaned


# Download and save a finite, positive VIX close series on provider dates.
def download_vix(start_date: str, end_date_inclusive: str, directory: Path | str | None = None) -> pd.Series:
    raw = yf.download(tickers="^VIX", start=start_date, end=_exclusive_end_date(end_date_inclusive),
                      auto_adjust=False, progress=False, threads=False, timeout=config.DOWNLOAD_TIMEOUT_SECONDS)
    close = _extract_close_prices(raw, ["^VIX"])

    if close.empty:
        raise ValueError("VIX download returned no Close prices.")

    vix = close["^VIX"].copy()
    vix.index = pd.DatetimeIndex(pd.to_datetime(vix.index, errors="coerce")).tz_localize(None).normalize()
    vix = vix.sort_index().loc[start_date:end_date_inclusive]
    vix = pd.to_numeric(vix, errors="coerce").replace([np.inf, -np.inf], np.nan)
    vix = vix.dropna()
    vix = vix[vix > 0]

    if len(vix) < 2:
        raise ValueError("VIX download did not provide two finite, positive Close prices.")

    vix.index.name = "Date"
    vix.name = "^VIX"

    directory = Path(config.DATA_DIR if directory is None else directory)
    directory.mkdir(parents=True, exist_ok=True)

    vix.to_csv(directory / "VIX_Close.csv", header=True, lineterminator="\n")
    print(f"VIX: {len(vix)} prices, {vix.index.min():%Y-%m-%d} to {vix.index.max():%Y-%m-%d}")
    return vix


# SHA-256 digest of a file, read in chunks so large price panels need not fit in memory at once.
def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


# How a data file compares with the data manifest in DATA_DIR: its checksum matches or differs, it is not listed, or there is no manifest.
def manifest_status(path: Path | str) -> str:
    manifest = Path(config.DATA_DIR) / MANIFEST_NAME

    if not manifest.exists():
        return f"no {MANIFEST_NAME} in {config.DATA_DIR}"

    expected = pd.read_csv(manifest).set_index("file")["sha256"]

    if Path(path).name not in expected:
        return f"not listed in {MANIFEST_NAME}"

    return f"matches {MANIFEST_NAME}" if file_sha256(path) == expected[Path(path).name] else f"differs from {MANIFEST_NAME}, so it is not the published file"


# Data files the analyses read: each market's cleaned price panel and the VIX series.
def required_data_files() -> list[Path]:
    return [cleaned_prices_path(index_code) for index_code in config.INDEX_CODES] + [Path(config.VIX_PATH)]


# Guidance for missing data files: which files, where they belong, how to restore and verify the paper's data, and the fresh-download alternative.
def missing_data_message(missing: Sequence[Path]) -> str:
    return (f"{len(missing)} data files are missing: {', '.join(str(path) for path in missing)}. The paper's data files are part of the repository, in {config.DATA_DIR}: "
            f"restore them with 'git checkout -- data' and check them with 'python scripts/data_manifest.py --verify' against {MANIFEST_NAME} (README, Data). "
            "Alternatively, 'python run_all.py --download' downloads today's constituent lists and prices and screens them, which will not reproduce the paper's numbers exactly because Yahoo Finance revises adjusted prices and index memberships change.")


# Raise FileNotFoundError, with guidance on obtaining the data, if any of the given data files is missing.
def check_data_files(paths: Sequence[Path | str]) -> None:
    missing = [Path(path) for path in paths if not Path(path).exists()]

    if missing:
        raise FileNotFoundError(missing_data_message(missing))
