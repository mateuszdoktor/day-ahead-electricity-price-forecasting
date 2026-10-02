import pandas as pd

from src.utils.cleaning import convert_timezone

DataDict = dict[str, pd.DataFrame]

MAX_INTERPOLATION_HOURS = 3
WEATHER_DRIVEN_DATASETS = ["generation_wind_solar_forecast"]
FORWARD_FILL_DATASETS = ["water_reservoirs_hydro_storage"]


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
    end = max(
        df.index.max()
        for datasets in data.values()
        for name, df in datasets.items()
        if name not in FORWARD_FILL_DATASETS
    )
    return {
        code: {
            name: (
                extend_forward_fill(df, end)
                if name in FORWARD_FILL_DATASETS
                else preprocess_dataset(name, df)
            )
            for name, df in datasets.items()
        }
        for code, datasets in data.items()
    }


def preprocess_jao(data: DataDict) -> DataDict:
    data = convert_timezone(data, timezone="UTC")
    return {name: preprocess_dataset(name, df) for name, df in data.items()}


def preprocess_investing_com(data: DataDict, end: pd.Timestamp) -> DataDict:
    return {
        name: extend_forward_fill(df.sort_index(), end, freq="D")
        for name, df in data.items()
    }
