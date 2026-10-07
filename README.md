# Regime Detection in Financial Markets using Machine Learning

**Authors:** Abbinav Sankar Kailasam (University College London), Blaz Zlicar (Abu Dhabi Investment Authority) and Fabio Caccioli (University College London).

This repository holds the code and data behind the paper. It asks whether the structure of a stock market's correlation network can identify the market's regime (Calm, Transitional or Crisis) using only past data, out of sample, in agreement with an external stress measure (the VIX), across four equity markets: the S&P 500, Nikkei 225, FTSE 350 and CSI 300. One command regenerates every figure and table in the paper from the published price data.

## Approach

1. **Networks.** Rolling windows of 63, 132 and 378 trading days give correlation matrices, each filtered into three sparse networks: a correlation threshold, a minimum spanning tree plus threshold edges, and a TMFG plus threshold edges.
2. **Descriptors.** Each window becomes a vector: a Graph2Vec embedding of its network, its eigenvector centralities, or nine spectral features of its correlation matrix.
3. **Regimes.** The vectors are standardised, reduced with PCA and clustered into three regimes with K-means and with a Gaussian hidden Markov model (decoded causally). Everything is fitted on windows ending by 31 December 2019 and applied unchanged afterwards. Regimes are named by their mean correlation, from Calm (lowest) to Crisis (highest).
4. **Evaluation.** The Crisis regime is validated against VIX > 30 (S&P 500). Regimes are compared across markets and descriptors, network topology is compared between regimes, and the regimes drive an out-of-sample portfolio backtest: a fixed exposure per regime, and volatility targeting from a one-step regime forecast, against trailing-volatility and mean-correlation benchmarks.

The whole grid is run and reported: 4 markets, 3 window lengths and each descriptor's network constructions, 72 configurations in all. One of them, the **headline configuration** (S&P 500, 132-day windows, MST-threshold network, Graph2Vec, K-means), is used wherever an analysis needs a single configuration. It was chosen after results covering the full sample had been seen, which is why every configuration's results are reported.

## Quick start

```bash
conda env create -f environment.yml        # Python 3.13.5 and pinned packages (Intel MKL)
conda activate regime-networks
python scripts/data_manifest.py --verify   # check data/ against its checksums
python run_all.py                          # every output, figure and table: 2 to 3.5 hours
```

- `python run_all.py --list` shows the steps; `--from STEP` resumes, `--only STEP ...` runs a selection, `--download` fetches and screens the data afresh, and `--animations` adds GIFs of the networks.
- Every step is also a script with `--help`. `python scripts/run_sweeps.py --quick` is a smoke test of a few minutes.
- `python scripts/seed_robustness.py` (optional, a few minutes) measures how much the headline regimes depend on Graph2Vec's random seed.
- The environment installs this checkout in editable mode and resolves every path from it, so use one environment per checkout.

## Pipeline at a glance

| Step | Script | Produces |
|---|---|---|
| verify (or download + manifest) | `scripts/data_manifest.py` (`scripts/download_data.py`) | checked (or fresh) price panels in `data/` |
| networks | `scripts/data_processing.py` | correlation matrices and the three networks of every window |
| sweeps | `scripts/run_sweeps.py` | descriptors, regimes, VIX scores, per-window labels and fitted models for every configuration |
| geometry | `scripts/embedding_geometry_sweep.py` | Graph2Vec self-similarity and MDS maps |
| topology | `scripts/network_topology.py` | network statistics grouped by regime |
| model selection | `scripts/model_selection.py` | K-means inertia and HMM AIC for 2 to 9 regimes |
| portfolio | `scripts/portfolio_analysis.py` | backtests of five strategies and forecast scores, every market |
| figures, tables | `figures/*.py`, `tables/paper_tables.py` | the paper's figures and tables |

Intermediate outputs go to `outputs/` (not tracked; several gigabytes). The paper's figures and tables go to `results/figures/` and `results/tables/` (CSV and LaTeX), which are tracked, so they can be inspected without running anything.

## Repository layout

```
src/regime_networks/   the package: every computation (parameters in config.py)
scripts/               compute steps, data/ and outputs/ -> outputs/
figures/, tables/      figure and table steps -> results/
run_all.py             runs every step in order
notebooks/             a step-by-step Graph2Vec tutorial; the evidence for the correlation threshold
data/                  the paper's data and MANIFEST.csv
results/               the paper's figures and tables
environment.yml        the pinned environment
```

## Data

- **Sources.** Daily adjusted closes from Yahoo Finance (via `yfinance`), 1 January 2001 to 30 September 2026, and the VIX close. Constituent lists come from Wikipedia, except the Nikkei 225, a snapshot of the index provider's list dated 21 July 2026 in `src/regime_networks/constituents.py`. The FTSE 350 combines the FTSE 100 and FTSE 250. No credentials are needed.
- **Screen.** Official exchange sessions only; stocks with a complete, positive price history over the whole period; market-wide stale dates removed; stocks removed for an absolute daily log return above 0.8 or a flat price over 63 or more returns. This keeps 348, 184, 158 and 36 stocks. Every exclusion and its reason is saved in `data/`.
- **The paper's data.** `data/` holds the files the results were computed from, downloaded on 7 October 2026, with SHA-256 checksums in `data/MANIFEST.csv`; `run_all.py` verifies them first and stops with restore instructions if one is missing. `run_all.py --download` overwrites them with a fresh download, which does not reproduce the paper's numbers exactly (Yahoo Finance revises adjusted prices and index memberships change); `git checkout -- data` restores them.
- **Terms.** The data are third-party and not covered by the code's MIT licence; their use is subject to the providers' terms.

## Reproducibility

- All randomness is seeded. Graph2Vec trains single-threaded and its inference is seeded per graph, so with the pinned environment a run is bitwise identical on the same platform, and a partial re-run writes the same files as a full one.
- On another CPU, BLAS library or operating system, results agree to floating-point precision; Graph2Vec trains in 32-bit, so a few windows near a cluster boundary can change regime.
- The package switches off pandas' use of numexpr, whose multithreaded evaluation intermittently corrupted large-frame arithmetic in the reference environment.
- There is no test suite; `python -m pyflakes src scripts figures tables run_all.py` checks for unused imports and undefined names.

## Citation and licence

- **Citing.** See `CITATION.cff`. If you use the data, cite this repository and acknowledge Yahoo Finance, Wikipedia and the Nikkei index provider.
- **Licence.** The code is released under the MIT licence (`LICENSE`). It covers the code, not the third-party data in `data/`.
