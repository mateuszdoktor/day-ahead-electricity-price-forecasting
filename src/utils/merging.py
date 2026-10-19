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

MODELS = ("trees", "nn", "epf")
SCOPES = ("pl", "pl_foreign")
DATASETS = [
    (model, scope)
    for model in MODELS
    for scope in SCOPES
    if not (model == "epf" and scope != "pl")
]

EPF_DROP = ["PL_price_lag24", "PL_price_lag48", "PL_price_lag168"]

SPLITS = {
    "full": (FULL_DATASET_START_DATE, FULL_DATASET_END_DATE),
    "train": (TRAIN_DATASET_START, TRAIN_DATASET_END),
    "val": (VAL_DATASET_START, VAL_DATASET_END),
    "test": (TEST_DATASET_START, TEST_DATASET_END),
}


def merge_features(model: str, scope: str, date_range: str) -> pd.DataFrame:
    if (model, scope) not in DATASETS:
        raise KeyError((model, scope))

    source = "nn" if model == "epf" else model
    entsoe = load_merged_data(
        DATA_DIR / "features" / "entsoe" / source / scope / f"{date_range}.parquet"
    )
    jao = load_merged_data(
        DATA_DIR / "features" / "jao" / source / scope / f"{date_range}.parquet"
    )
    investing = load_merged_data(
        DATA_DIR / "features" / "investing_com" / source / f"{date_range}.parquet"
    )
    price = load_merged_data(
        DATA_DIR / "preprocessed" / "entsoe" / date_range / "PL.parquet"
    )[["day_ahead_prices"]].rename(columns={"day_ahead_prices": "PL_price"})

    df = pd.concat([price, entsoe, jao, investing], axis=1, join="inner").sort_index()
    if model == "epf":
        df = df.drop(columns=EPF_DROP)
    return df


def cut_to_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.index.tz is None:
        return df.loc[start : f"{end} 23:00"].copy()

    local = df.index.tz_convert(LOCAL_TZ)
    start_local = pd.Timestamp(start, tz=LOCAL_TZ)
    end_local = pd.Timestamp(f"{end} 23:00", tz=LOCAL_TZ)
    return df.loc[(local >= start_local) & (local <= end_local)].copy()


def to_warsaw_hourly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.tz_convert(LOCAL_TZ).tz_localize(None)
    out = out.groupby(level=0).mean()
    full_index = pd.date_range(out.index.min(), out.index.max(), freq="h")
    return out.reindex(full_index).interpolate(method="time", limit=1)


def split_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: cut_to_dates(df, start, end) for name, (start, end) in SPLITS.items()}
