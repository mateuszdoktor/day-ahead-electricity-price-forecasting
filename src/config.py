from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DOWNLOAD_START_DATE = "2022-06-09"
DOWNLOAD_END_DATE = "2026-10-21"

# 226 weeks - 1582 days
FULL_DATASET_START_DATE = "2022-06-20"
FULL_DATASET_END_DATE = "2026-10-18"

# 156 weeks - 1092 days (3 years)
TRAIN_DATASET_START = "2022-06-20"
TRAIN_DATASET_END = "2025-06-15"

# 18 weeks - 126 days
VAL_DATASET_START = "2025-06-16"
VAL_DATASET_END = "2025-10-19"

# 52 weeks - 364 days (1 year)
TEST_DATASET_START = "2025-10-20"
TEST_DATASET_END = "2026-10-18"

LOCAL_TZ = "Europe/Warsaw"

ENTSOE_BIDDING_ZONE_CODES = [
    # "AT",
    "CZ",
    "DE_LU",
    # "FR",
    # "HU",
    "LT",
    # "NO",
    "PL",
    # "SE",
    "SE_4",
    "SK",
]

EPF_EXOG = ["PL_gen_fc_balance", "PL_load_fc"]
