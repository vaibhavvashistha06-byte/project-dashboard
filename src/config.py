#All settings in one place. Change a value here and re-run the scripts.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"          # cleaned prices from filtering.py
COMPANIES_FILE = ROOT / "data" / "filtered_companies.csv"
FEATURES_FILE = ROOT / "data" / "features" / "features.pkl"
DATASET_FILE = ROOT / "data" / "features" / "dataset.pkl"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

#target: "will the stock rise meaningfully over the next HORIZON days?"

HORIZON = 5            # trading days to hold
LABEL_MODE = "relative"   # "relative" = beat the market-average stock; "absolute" = raw return
ATR_MULT = 0.3         # label = 1 if return > ATR_MULT * daily-ATR% * sqrt(HORIZON)

#time split (never random!)
TRAIN_END = "2024-10-31"   # years 1-3 of the 5y download = train
VAL_END = "2025-10-31"     # year 4 = validation; test = year 5 (everything after this)
EMBARGO_DAYS = 10          # calendar days cut off before each boundary so labels don't leak

#evaluation
COST = 0.002               # 0.2% round-trip cost (brokerage + slippage)
TOP_QUANTILE = 0.90        # trade only the top 10% most confident days
MIN_SIGNALS = 30           # ignore pattern/cluster combos seen fewer times than this
RANDOM_STATE = 42