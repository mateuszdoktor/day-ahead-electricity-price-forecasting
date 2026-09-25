from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DOWNLOAD_START_DATE = "2022-12-01"
DOWNLOAD_END_DATE = "2026-09-16"

# 194 weeks - 1358 days
FULL_DATASET_START_DATE = "2022-12-26"
FULL_DATASET_END_DATE = "2026-09-13"

# 105 weeks - 735 days
TRAIN_DATASET_START = "2022-12-26"
TRAIN_DATASET_END = "2024-12-29"

# 35 weeks - 245 days
VAL_DATASET_START = "2024-12-30"
VAL_DATASET_END = "2025-08-31"

# 54 weeks - 378 days
TEST_DATASET_START = "2025-09-01"
TEST_DATASET_END = "2026-09-13"

ENTOSE_BIDDING_ZONE_CODES = [
    "AT",
    "CZ",
    "DE_LU",
    "FR",
    "HU",
    "LT",
    "NO",
    "PL",
    "SE",
    "SE_4",
    "SK",
]
