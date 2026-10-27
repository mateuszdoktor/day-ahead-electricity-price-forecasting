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
from src.utils.feature_engineering.fe_trees import build_tree_features, shift_unknown
from src.utils.file_io import load_merged_data

STEPS = ("d1", "d2", "d3")

STEP_BLOCKS = {
    "d1": [("entsoe", "d1"), ("investing_com", "d1")],
    "d2": [("jao", "d2")],
    "d3": [("entsoe", "d3"), ("jao", "d3")],
}


DATASETS = (
    [("tft", step) for step in STEPS]
    + [("trees", step) for step in STEPS]
    + [("epf", "pl")]
)

EPF_COLUMNS = [
    "PL_load_fc",
    "PL_load_fc_delta24",
    "PL_wind_fc_lag24",
    "PL_solar_fc_lag24",
    "PL_gen_fc_lag24",
]

SPLITS = {
    "full": (FULL_DATASET_START_DATE, FULL_DATASET_END_DATE),
    "train": (TRAIN_DATASET_START, TRAIN_DATASET_END),
    "val": (VAL_DATASET_START, VAL_DATASET_END),
    "test": (TEST_DATASET_START, TEST_DATASET_END),
}


def blocks_up_to(step: str) -> list[tuple[str, str]]:
    if step not in STEPS:
        raise KeyError(step)

    return [
        block for name in STEPS[: STEPS.index(step) + 1] for block in STEP_BLOCKS[name]
    ]


def merge_features(model: str, scope: str, date_range: str) -> pd.DataFrame:
    if (model, scope) not in DATASETS:
        raise KeyError((model, scope))

    step = "d1" if model == "epf" else scope
    price = load_merged_data(
        DATA_DIR / "preprocessed" / "entsoe" / date_range / "PL.parquet"
    )[["day_ahead_prices"]].rename(columns={"day_ahead_prices": "PL_price"})
    blocks = [
        load_merged_data(
            DATA_DIR
            / "features"
            / source
            / "tft"
            / block_step
            / f"{date_range}.parquet"
        )
        for source, block_step in blocks_up_to(step)
    ]

    df = pd.concat([price, *blocks], axis=1, join="inner").sort_index()
    if model == "trees":
        df = build_tree_features(df, step)
    if model == "epf":
        df = shift_unknown(df)[["PL_price", *EPF_COLUMNS]].copy()
        df["PL_gen_fc_balance"] = df["PL_gen_fc_lag24"] - (
            df["PL_solar_fc_lag24"] + df["PL_load_fc_delta24"]
        )
    return df


def cut_to_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.index.tz is None:
        return df.loc[start:f"{end} 23:00"].copy()

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
