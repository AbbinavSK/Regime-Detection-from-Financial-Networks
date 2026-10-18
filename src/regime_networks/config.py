from pathlib import Path

# ---- Paths, resolved from the repository root so every script runs from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
LABELS_DIR = OUTPUTS_DIR / "labels"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
VIX_PATH = DATA_DIR / "VIX_Close.csv"

# ---- Study period and markets.
START_DATE = "2001-01-01"
END_DATE_INCLUSIVE = "2026-09-30"
INDEX_CODES = ["SP500", "Nikkei225", "FTSE350", "CSI300"]
# The market every hyperparameter was chosen on, and the only one with an external stress indicator (the VIX).
PRIMARY_INDEX = "SP500"
VIX_INDEX = PRIMARY_INDEX
# Composite indices built by downloading and screening each component, then intersecting their dates and screening again.
INDEX_COMPONENTS = {"FTSE350": ["FTSE100", "FTSE250"]}
MARKET_CALENDARS = {"SP500": "XNYS", "Nikkei225": "XTKS", "FTSE100": "XLON", "FTSE250": "XLON", "FTSE350": "XLON", "CSI300": "XSHG"}

# Named market events (date -> name), annotated on the regime figures.
CRISIS_EVENTS = {"2008-09-15": "GFC", "2011-08-01": "Euro Debt", "2020-03-01": "COVID", "2022-06-01": "Rate Shock"}

# ---- Price download (prices and constituent lists). Tickers that return no prices are re-downloaded one at a time, waiting between rounds; a failure whose provider message contains a transient marker stops the download instead of dropping the stock.
DOWNLOAD_BATCH_SIZE = 50
DOWNLOAD_BATCH_ATTEMPTS = 2
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_TICKER_ATTEMPTS = 3
DOWNLOAD_RETRY_WAIT_SECONDS = 10
DOWNLOAD_TRANSIENT_ERRORS = ("RateLimit", "Too Many Requests", "Timeout", "timed out", "Connection")

# ---- Price screening.
MIN_STOCKS = 4
MIN_RETURN_ROWS = 378
MAX_ABS_LOG_RETURN = 0.8
STALE_ROW_THRESHOLD = 0.95
ZERO_RETURN_TOLERANCE = 1e-12

# ---- Rolling windows (length T, shift in trading days) and network construction.
WINDOW_CONFIGS = ((378, 22), (132, 10), (63, 5))
WINDOW_LENGTHS = sorted(length for length, _ in WINDOW_CONFIGS)
# The smallest threshold on a 0.005 grid that at most 5% of the pairwise correlations pooled over the S&P 500 training windows (T=132, ending by 2019) reach; it is 0.66 on all windows. The evidence is in notebooks/threshold_diagnostic.ipynb, which no pipeline step reads.
CORRELATION_THRESHOLD = 0.65
GRAPH_TYPES = ["threshold", "mst", "tmfg"]
GRAPH_KEYS = {"threshold": "threshold_graphs", "mst": "mst_threshold_graphs", "tmfg": "tmfg_threshold_graphs"}

# ---- Sweep grid: the descriptors and the network constructions each is built on (None means the correlation matrix itself).
DESCRIPTORS = ["graph2vec", "eigencentrality", "spectral"]
DESCRIPTOR_GRAPH_TYPES = {"graph2vec": ["threshold", "mst", "tmfg"], "eigencentrality": ["mst", "tmfg"], "spectral": [None]}

# ---- Graph2Vec (WL relabelling + Doc2Vec PV-DBOW with negative sampling) and the random seed used throughout. The learning rate decays linearly to G2V_TRAIN_MIN_ALPHA in training and to G2V_INFER_MIN_ALPHA in inference; G2V_TRAIN_MIN_ALPHA, G2V_NEGATIVE and G2V_NS_EXPONENT are gensim's defaults, passed explicitly so every setting is visible here.
SEED = 42
G2V_DIMENSIONS = 128
G2V_WL_ITERATIONS = 3
G2V_EPOCHS = 100
G2V_LEARNING_RATE = 0.025
G2V_TRAIN_MIN_ALPHA = 0.0001
G2V_INFER_MIN_ALPHA = 1e-5
G2V_MIN_COUNT = 5
G2V_DOWN_SAMPLING = 0.0001
G2V_NEGATIVE = 5
G2V_NS_EXPONENT = 0.75
G2V_WORKERS = 1

# ---- Dimensionality reduction and regime models.
PCA_VARIANCE = 0.95
N_REGIMES = 3
REGIME_NAMES = ["Calm", "Transitional", "Crisis"]
KMEANS_N_INIT = 20
HMM_N_SEEDS = 20
HMM_COVARIANCE_TYPE = "diag"
HMM_N_ITER = 200
HMM_TOL = 1e-4

# ---- Out-of-sample protocol and VIX validation. The lag of the Crisis regime behind VIX stress is the shift, in windows, that maximises its F1 over all windows within VIX_LAG_WINDOWS (positive: the regime trails the VIX).
TRAIN_TEST_CUTOFF = "2019-12-31"
VIX_CRISIS_THRESHOLD = 30.0
VIX_LAG_WINDOWS = range(-15, 31)
PR_LOCALISATION_FLAG = 10

# ---- Headline configuration (primary market, T=132, MST-threshold, Graph2Vec, K-means), the one the analyses below single out; it was chosen after full-sample results (test period included) had been seen, which is why every grid point is reported. Its grid point and file names come from regime_networks.grid (headline_grid_point).
HEADLINE_WINDOW = (132, 10)
HEADLINE_GRAPH_TYPE = "mst"
HEADLINE_DESCRIPTOR = "graph2vec"
HEADLINE_METHOD = "kmeans"
# The single grid point and HMM restart cap of run_sweeps.py --quick, a smoke test written under outputs/quick/.
QUICK_SWEEP_POINT = (PRIMARY_INDEX, HEADLINE_WINDOW[0], HEADLINE_GRAPH_TYPE)
QUICK_HMM_SEEDS = 3

# ---- Network topology by detected regime: every window's three constructions, each grouped by the regimes its own headline-descriptor, headline-method model detected, summarised per regime over train, test and all windows; the window whose mu is nearest its regime's median mu is the one drawn.
TOPOLOGY_INDEX = PRIMARY_INDEX
TOPOLOGY_WINDOW_LENGTH = HEADLINE_WINDOW[0]

# ---- Embedding geometry: the market it is computed for and the MDS restarts.
GEOMETRY_INDEX = PRIMARY_INDEX
MDS_N_INIT = 8

# ---- Model selection for the number of regimes (K-means elbow, HMM AIC) on every grid point's train embedding; the headline grid point gets its own figure pair. A lowest AIC at either end of MODEL_SELECTION_K_RANGE is reported as an edge minimum, since the true minimum may lie outside the range. The AIC comparison used 10 HMM restarts in the dissertation, not HMM_N_SEEDS. Fits run over MODEL_SELECTION_N_JOBS processes; every fit is seeded, so results agree across settings up to floating-point rounding (about 1e-7 relative in K-means inertia).
MODEL_SELECTION_K_RANGE = range(2, 10)
MODEL_SELECTION_HMM_SEEDS = 10
MODEL_SELECTION_N_JOBS = 4

# ---- Doc2Vec seed robustness (scripts/seed_robustness.py, a pipeline step after model selection): every market's headline grid point refitted with Graph2Vec seeded by SEED (checked to reproduce the saved regimes) and by each of these seeds, the scaler, PCA and K-means unchanged, and each seed's K-means regimes compared with SEED's. Fits run over SEED_ROBUSTNESS_N_JOBS processes; each is single-threaded and seeded, so results do not depend on the number.
SEED_ROBUSTNESS_SEEDS = (1, 2, 3, 4, 5)
SEED_ROBUSTNESS_N_JOBS = 4

# ---- Portfolio backtest, run for every market and every construction of PORTFOLIO_GRAPH_TYPES at the headline window length (each with its own Graph2Vec K-means model) over the full test period: each strategy's per-window equity exposure applies from the next trading day to an equal-weight, never-rebalanced sleeve of that market's cleaned constituents, in local currency, idle capital earning a constant cash rate. PORTFOLIO_STRATEGIES hold a fixed exposure per regime rank (Calm, Transitional, Crisis); buy-and-hold is the fully invested benchmark.
PORTFOLIO_GRAPH_TYPES = GRAPH_TYPES
BUY_AND_HOLD = "Buy-and-hold"
PORTFOLIO_STRATEGIES = {BUY_AND_HOLD: (1.0, 1.0, 1.0), "Regime-timed": (1.0, 0.5, 0.0)}
# The regime-forecast strategy holds 1 minus the forecast probability that the next window is Crisis (the soft K-means state times the training transition matrix).
REGIME_FORECAST_STRATEGY = "Regime-forecast"
# Each timed strategy's benchmark holds that strategy's average daily exposure over the test period throughout, separating the value of timing from that of holding less equity.
TIMED_STRATEGIES = ["Regime-timed", REGIME_FORECAST_STRATEGY]
CONSTANT_EXPOSURE_STRATEGIES = {name: f"Constant exposure ({name})" for name in TIMED_STRATEGIES}
# Every strategy in output order, the one list downstream steps iterate over.
PORTFOLIO_STRATEGY_NAMES = [*PORTFOLIO_STRATEGIES, REGIME_FORECAST_STRATEGY, *CONSTANT_EXPOSURE_STRATEGIES.values()]
# Transaction cost per unit of exposure change (10 basis points, one-way), charged on the day the new exposure applies; the first day pays for the purchase from cash. Metrics are reported gross and net of it.
TRANSACTION_COST = 0.001
# Soft-state regime forecast (the soft K-means state times the training transition matrix): the temperature is searched over np.logspace(*FORECAST_TEMPERATURE_MULTIPLES) multiples of the median squared distance to the nearest centroid, and log scores floor a probability at FORECAST_PROBABILITY_FLOOR.
FORECAST_TEMPERATURE_MULTIPLES = (-3, 2, 101)
FORECAST_PROBABILITY_FLOOR = 1e-6
# Annual cash rate per market: the mean of the monthly short-term rate over the test period, January 2020 to September 2026 (retrieved 4 October 2026). S&P 500: US 3-month Treasury bill, secondary market (Federal Reserve H.15 RIFSGFSM03_N.M, FRED TB3MS). Nikkei 225 and CSI 300: OECD 3-month interbank rates (FRED IR3TIB01JPM156N, IR3TIB01CNM156N), to July 2026, the latest month published. FTSE 350: SONIA monthly average (Bank of England IUMSOIA), as no UK 3-month series covers the period.
RISK_FREE_RATES_ANNUAL = {"SP500": 0.028572, "Nikkei225": 0.002465, "FTSE350": 0.027276, "CSI300": 0.025147}
# Floor on the daily standard deviation of a market's equity-sleeve returns (real equity returns sit near 0.01); below it the returns are degenerate and the backtest stops.
MIN_EQUITY_RETURN_STD = 0.001
# Annualisation uses each market's observed sessions per year (portfolio.sessions_per_year), measured against this calendar-year length in days.
CALENDAR_DAYS_PER_YEAR = 365.25

# ---- Paper tables: the decimals printed in the LaTeX fragments.
TABLE_DECIMALS = 3

# ---- Figures: export resolution of the manuscript figures.
FIGURE_DPI = 600
# The regime figures the paper and its supplement show, one image file per figure under results/figures/regimes/, each a list of its panels from top to bottom: (market, construction or None for the spectral descriptor, descriptor, clustering method, panel kind), all at the headline window length. The kinds are a timeline against mu, the VIX overlay (S&P 500 only) and the pair of training and test transition matrices.
REGIME_FIGURES = {f"{PRIMARY_INDEX}_{HEADLINE_WINDOW[0]}d_{HEADLINE_GRAPH_TYPE}/{HEADLINE_DESCRIPTOR}_kmeans_timeline_vix.png": [(PRIMARY_INDEX, HEADLINE_GRAPH_TYPE, HEADLINE_DESCRIPTOR, "kmeans", kind) for kind in ("timeline", "vix_overlay")],
                  f"{PRIMARY_INDEX}_{HEADLINE_WINDOW[0]}d_{HEADLINE_GRAPH_TYPE}/{HEADLINE_DESCRIPTOR}_kmeans_transitions.png": [(PRIMARY_INDEX, HEADLINE_GRAPH_TYPE, HEADLINE_DESCRIPTOR, "kmeans", "transitions")],
                  f"{PRIMARY_INDEX}_{HEADLINE_WINDOW[0]}d_{HEADLINE_GRAPH_TYPE}/{HEADLINE_DESCRIPTOR}_hmm_timeline_vix.png": [(PRIMARY_INDEX, HEADLINE_GRAPH_TYPE, HEADLINE_DESCRIPTOR, "hmm", kind) for kind in ("timeline", "vix_overlay")],
                  f"{PRIMARY_INDEX}_{HEADLINE_WINDOW[0]}d/descriptor_timelines.png": [(PRIMARY_INDEX, HEADLINE_GRAPH_TYPE, HEADLINE_DESCRIPTOR, "kmeans", "timeline"), (PRIMARY_INDEX, HEADLINE_GRAPH_TYPE, "eigencentrality", "kmeans", "timeline"), (PRIMARY_INDEX, None, "spectral", "kmeans", "timeline")],
                  f"cross_market/timelines_{HEADLINE_WINDOW[0]}d_{HEADLINE_GRAPH_TYPE}.png": [(index_code, HEADLINE_GRAPH_TYPE, HEADLINE_DESCRIPTOR, "kmeans", "timeline") for index_code in INDEX_CODES if index_code != PRIMARY_INDEX]}
# The embedding-geometry figures of GEOMETRY_INDEX, each (window length, its rows from top to bottom), every row one panel per construction: self-similarity heatmaps, and MDS maps coloured by regime, by date, by edge count and by out-of-vocabulary rate (the share of a window's WL features absent from the trained vocabulary).
GEOMETRY_FIGURES = [(HEADLINE_WINDOW[0], ("selfsimilarity", "regime")), (HEADLINE_WINDOW[0], ("date", "edgecount", "oov")), *((length, ("selfsimilarity", "regime", "date", "edgecount")) for length in WINDOW_LENGTHS if length != HEADLINE_WINDOW[0])]
