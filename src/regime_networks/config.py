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

# Named market events (date -> name): episodes of the Crisis regime are named after the nearest event (portfolio analysis, where EXCLUDED_EPISODE is matched by name), and figures annotate them.
CRISIS_EVENTS = {"2000-03-01": "Dot-com", "2008-09-15": "GFC", "2011-08-01": "Euro Debt", "2020-03-01": "COVID", "2022-06-01": "Rate Shock"}

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

# ---- Out-of-sample protocol and VIX validation.
TRAIN_TEST_CUTOFF = "2019-12-31"
VIX_CRISIS_THRESHOLD = 30.0
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
CLIQUE_ENUM_CAP = 20000

# ---- Embedding geometry: the market it is computed for and the MDS restarts.
GEOMETRY_INDEX = PRIMARY_INDEX
MDS_N_INIT = 8

# ---- Model selection for the number of regimes (K-means elbow, HMM AIC) on every grid point's train embedding; the headline grid point gets its own figure pair. A lowest AIC at either end of MODEL_SELECTION_K_RANGE is reported as an edge minimum, since the true minimum may lie outside the range. The AIC comparison used 10 HMM restarts in the dissertation, not HMM_N_SEEDS. Fits run over MODEL_SELECTION_N_JOBS processes; every fit is seeded, so results agree across settings up to floating-point rounding (about 1e-7 relative in K-means inertia).
MODEL_SELECTION_K_RANGE = range(2, 10)
MODEL_SELECTION_HMM_SEEDS = 10
MODEL_SELECTION_N_JOBS = 4

# ---- Doc2Vec seed robustness (scripts/seed_robustness.py, run by hand, not a pipeline step): every market's headline grid point refitted with Graph2Vec seeded by SEED (checked to reproduce the saved regimes) and by each of these seeds, the scaler, PCA and K-means unchanged, and each seed's K-means regimes compared with SEED's. Fits run over SEED_ROBUSTNESS_N_JOBS processes; each is single-threaded and seeded, so results do not depend on the number.
SEED_ROBUSTNESS_SEEDS = (1, 2, 3, 4, 5)
SEED_ROBUSTNESS_N_JOBS = 4

# ---- Portfolio backtest, run for every market: each strategy's per-window equity exposure applies from the next trading day to an equal-weight, never-rebalanced sleeve of that market's cleaned constituents, in local currency, idle capital earning a constant cash rate; returns are gross of transaction costs. PORTFOLIO_STRATEGIES hold a fixed exposure per headline regime rank.
PORTFOLIO_STRATEGIES = {"Fully invested": (1.0, 1.0, 1.0), "Regime-timed": (1.0, 0.5, 0.0)}
# Volatility-targeting strategies, each holding min(VOL_TARGET_EXPOSURE_CAP, target / forecast volatility): the headline regime forecast (soft K-means state times the training transition matrix, times each regime's training volatility), the trailing volatility over TRAILING_VOL_SESSIONS, and the same regime forecast built from a K-means fit on mu alone.
REGIME_FORECAST_STRATEGY = "Regime-forecast vol target"
TRAILING_VOL_STRATEGY = "Trailing-vol target"
MU_ONLY_STRATEGY = "Mu-only vol target"
# Every strategy in output order, the one list downstream steps iterate over.
PORTFOLIO_STRATEGY_NAMES = [*PORTFOLIO_STRATEGIES, REGIME_FORECAST_STRATEGY, TRAILING_VOL_STRATEGY, MU_ONLY_STRATEGY]
# Target of the volatility-targeting strategies, from the risk proxy's training volatility: "calm" (in the headline model's Calm regime) or "long_run" (over all training sessions).
VOL_TARGET = "calm"
VOL_TARGET_EXPOSURE_CAP = 1.0
TRAILING_VOL_SESSIONS = 63
# Regime forecast: the soft-state temperature is searched over np.logspace(*FORECAST_TEMPERATURE_MULTIPLES) multiples of the median squared distance to the nearest centroid, and log scores floor a probability at FORECAST_PROBABILITY_FLOOR.
FORECAST_TEMPERATURE_MULTIPLES = (-3, 2, 101)
FORECAST_PROBABILITY_FLOOR = 1e-6
# Annual cash rate per market: the mean of the monthly short-term rate over the test period, January 2020 to September 2026 (retrieved 4 October 2026). S&P 500: US 3-month Treasury bill, secondary market (Federal Reserve H.15 RIFSGFSM03_N.M, FRED TB3MS). Nikkei 225 and CSI 300: OECD 3-month interbank rates (FRED IR3TIB01JPM156N, IR3TIB01CNM156N), to July 2026, the latest month published. FTSE 350: SONIA monthly average (Bank of England IUMSOIA), as no UK 3-month series covers the period.
RISK_FREE_RATES_ANNUAL = {"SP500": 0.028572, "Nikkei225": 0.002465, "FTSE350": 0.027276, "CSI300": 0.025147}
# Floor on the daily standard deviation of a market's equity-sleeve returns (real equity returns sit near 0.01); below it the returns are degenerate and the backtest stops.
MIN_EQUITY_RETURN_STD = 0.001
# Annualisation uses each market's observed sessions per year (portfolio.sessions_per_year), measured against this calendar-year length in days.
CALENDAR_DAYS_PER_YEAR = 365.25
EPISODE_EVENT_MAX_DAYS = 90
EXCLUDED_EPISODE = "COVID"
# Circular block bootstrap of each strategy's daily test-period returns (blocks of BOOTSTRAP_BLOCK_LENGTH sessions that wrap from the last day to the first, so every day is equally likely to be drawn); all strategies resample the same days, and the reported quantiles are an interquartile range, not a confidence interval.
BOOTSTRAP_BLOCK_LENGTH = 132
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_QUANTILES = (0.25, 0.75)

# ---- Paper tables: the decimals printed in the LaTeX fragments (the cross-market table compares the headline configuration's test-period persistence).
TABLE_DECIMALS = 3

# ---- Figures: presentation animations (one frame per window) and export resolution for manuscript figures and bulk per-grid-point dashboards.
ANIMATION_WINDOW_LENGTH = HEADLINE_WINDOW[0]
ANIMATION_FPS = 12
ANIMATION_DPI = 100
FIGURE_DPI = 600
DASHBOARD_DPI = 150
