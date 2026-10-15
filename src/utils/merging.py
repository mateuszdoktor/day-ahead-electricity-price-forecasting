import pandas as pd

from src.config import (
    DATA_DIR,
    FULL_DATASET_END_DATE,
    FULL_DATASET_START_DATE,
    LOCAL_TZ,
    TEST_DATASET_END,
    TEST_DATASET_START,
    TRAIN_DATASET_END,
    TRAIN_DATASET_START,
    VAL_DATASET_END,
    VAL_DATASET_START,
)
from src.utils.file_io import load_merged_data

MODELS = ("trees", "nn")
SCOPES = ("pl", "pl_foreign")

SPLITS = {
    "full": (FULL_DATASET_START_DATE, FULL_DATASET_END_DATE),
    "train": (TRAIN_DATASET_START, TRAIN_DATASET_END),
    "val": (VAL_DATASET_START, VAL_DATASET_END),
    "test": (TEST_DATASET_START, TEST_DATASET_END),
}


def merge_features(model: str, scope: str, date_range: str) -> pd.DataFrame:
    if model not in MODELS or scope not in SCOPES:
        raise KeyError((model, scope))

    entsoe = load_merged_data(
        DATA_DIR / "features" / "entsoe" / model / scope / f"{date_range}.parquet"
    )
    jao = load_merged_data(
        DATA_DIR / "features" / "jao" / model / scope / f"{date_range}.parquet"
    )
    investing = load_merged_data(
        DATA_DIR / "features" / "investing_com" / model / f"{date_range}.parquet"
    )
    price = load_merged_data(
        DATA_DIR / "preprocessed" / "entsoe" / date_range / "PL.parquet"
    )[["day_ahead_prices"]].rename(columns={"day_ahead_prices": "PL_price"})

    return pd.concat([price, entsoe, jao, investing], axis=1, join="inner").sort_index()


def to_local_naive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    index = df.index.tz_convert(LOCAL_TZ).tz_localize(None)

    duplicated = index.duplicated(keep="first")
    if duplicated.any():
        index = index + pd.to_timedelta(duplicated.astype("int64"), unit="s")

    df.index = index
    return df


def cut_to_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[start:end].copy()


def split_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: cut_to_dates(df, start, end) for name, (start, end) in SPLITS.items()}
