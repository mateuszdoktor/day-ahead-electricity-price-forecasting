import pandas as pd

from src.utils.cleaning import convert_timezone

DataDict = dict[str, pd.DataFrame]

MAX_INTERPOLATION_HOURS = 3
WEATHER_DRIVEN_DATASETS = ["generation_wind_solar_forecast"]
# No longer used: the reservoir datasets (AT, NO, SE) are not downloaded anymore.
# FORWARD_FILL_DATASETS = ["water_reservoirs_hydro_storage"]


def interpolate_short_gaps(
    series: pd.Series, max_hours: int = MAX_INTERPOLATION_HOURS
) -> pd.Series:
    gap_length = series.isna().groupby(series.notna().cumsum()).transform("sum")
    short_gap = series.isna() & (gap_length <= max_hours)
    return series.where(~short_gap, series.interpolate(limit_area="inside"))


def fill_from_previous_days(
    df: pd.DataFrame, days: int, max_steps: int = 7
) -> pd.DataFrame:
    filled = df.copy()
    for step in range(1, max_steps + 1):
        shifted = df.shift(freq=pd.Timedelta(days=days * step))
        filled = filled.fillna(shifted.reindex(filled.index))
    return filled


def preprocess_dataset(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.resample("h").mean()
    if name == "generation_forecast":
        df = df.mask(df == 0)
    df = df.apply(interpolate_short_gaps)
    days = 1 if name in WEATHER_DRIVEN_DATASETS else 7
    return fill_from_previous_days(df, days=days)


def extend_forward_fill(
    df: pd.DataFrame, end: pd.Timestamp, freq: str = "h"
) -> pd.DataFrame:
    return df.reindex(pd.date_range(df.index.min(), end, freq=freq), method="ffill")


def preprocess_entsoe(data: dict[str, DataDict]) -> dict[str, DataDict]:
    data = {
        code: convert_timezone(datasets, timezone="UTC")
        for code, datasets in data.items()
    }
    # No longer used: only needed to extend the reservoir datasets (see above).
    # end = max(
    #     df.index.max()
    #     for datasets in data.values()
    #     for name, df in datasets.items()
    #     if name not in FORWARD_FILL_DATASETS
    # )
    return {
        code: {
            name: preprocess_dataset(name, df)
            # name: (
            #     extend_forward_fill(df, end)
            #     if name in FORWARD_FILL_DATASETS
            #     else preprocess_dataset(name, df)
            # )
            for name, df in datasets.items()
        }
        for code, datasets in data.items()
    }


def preprocess_jao(data: DataDict) -> DataDict:
    data = convert_timezone(data, timezone="UTC")
    return {name: preprocess_dataset(name, df) for name, df in data.items()}


def preprocess_investing_com(data: DataDict, end: pd.Timestamp) -> DataDict:
    return {
        name: extend_forward_fill(df.sort_index(), end, freq="h")
        for name, df in data.items()
    }


def _snake_case(value) -> str:
    return str(value).strip().replace(" ", "_").replace("/", "_").replace("-", "_")


def _rename_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = df.copy()
    if renamed.columns.equals(pd.RangeIndex(len(renamed.columns))) or list(
        renamed.columns
    ) == [0]:
        renamed.columns = [prefix]
        return renamed
    if len(renamed.columns) == 1:
        renamed.columns = [prefix]
        return renamed
    renamed.columns = [f"{prefix}_{_snake_case(col)}" for col in renamed.columns]
    return renamed


def merge_entsoe_zone(data: DataDict) -> pd.DataFrame:
    frames = [_rename_columns(df, name) for name, df in data.items()]
    return pd.concat(frames, axis=1, join="outer").sort_index()


def merge_entsoe_zones(data: dict[str, DataDict]) -> DataDict:
    return {zone: merge_entsoe_zone(datasets) for zone, datasets in data.items()}


def merge_entsoe_zone_frames(zones: DataDict) -> pd.DataFrame:
    frames = [df.add_prefix(f"{zone}_") for zone, df in zones.items()]
    return pd.concat(frames, axis=1, join="outer").sort_index()


def merge_jao(data: DataDict) -> pd.DataFrame:
    frames = [_rename_columns(df, name) for name, df in data.items()]
    return pd.concat(frames, axis=1, join="outer").sort_index()


def merge_investing_com(data: DataDict) -> pd.DataFrame:
    frames = [_rename_columns(df, name) for name, df in data.items()]
    return pd.concat(frames, axis=1, join="outer").sort_index()
