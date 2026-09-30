import logging

import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

# --- ENTSO-E ------------------------------

DataDict = dict[str, pd.DataFrame]

GENERATION_UNAVAILABILITY = "generation_units_unavailability"
PRODUCTION_UNAVAILABILITY = "production_units_unavailability"
UNAVAILABILITY_DATASETS = [GENERATION_UNAVAILABILITY, PRODUCTION_UNAVAILABILITY]

COUNTRY_CONFIG = {
    "PL": {
        "drop_gen_types": [
            "Other renewable",
            "Other",
            "Hydro Pumped Storage - Actual Consumption",
            "Wind Offshore",
        ],
        "drop_ws_forecast": ["Wind Offshore"],
    },
    "FR": {
        "drop_ws_forecast": ["Wind Offshore"],
    },
    "LT": {
        "datasets_to_remove": ["generation_forecast"],
    },
}


def _select_datasets(data: DataDict, datasets: list[str] | None) -> list[str]:
    if datasets is not None:
        return datasets
    return [name for name in data if name not in UNAVAILABILITY_DATASETS]


def drop_columns(
    data: DataDict, datasets: list[str], columns_list: list[list[str]]
) -> DataDict:
    for dataset, columns in zip(datasets, columns_list):
        if dataset in data:
            data[dataset] = data[dataset].drop(columns=columns, errors="ignore")
    return data


def drop_datasets(data: DataDict, datasets: list[str]) -> DataDict:
    for dataset in datasets:
        data.pop(dataset, None)
    return data


def drop_rows_missing_power_or_plant_type(
    data: DataDict, datasets: list[str] | None = None
) -> DataDict:
    required = ["nominal_power", "plant_type"]
    if datasets is None:
        datasets = UNAVAILABILITY_DATASETS

    for dataset in datasets:
        if dataset not in data:
            continue
        df = data[dataset]
        if set(required).issubset(df.columns):
            data[dataset] = df.dropna(subset=required)
    return data


def drop_duplicate_index(
    data: DataDict, datasets: list[str] | None = None, strategy: str = "first"
) -> DataDict:
    for dataset in _select_datasets(data, datasets):
        if dataset in data:
            df = data[dataset]
            data[dataset] = df[~df.index.duplicated(keep=strategy)]
    return data


def drop_duplicate_rows(
    data: DataDict, datasets: list[str] | None = None, strategy: str = "first"
) -> DataDict:
    for dataset in _select_datasets(data, datasets):
        if dataset in data:
            data[dataset].drop_duplicates(keep=strategy)
    return data


def drop_duplicate_subset(
    data: DataDict,
    datasets: list[str] | None = None,
    strategy: str = "first",
    subset: list[str] | None = None,
) -> DataDict:
    for dataset in _select_datasets(data, datasets):
        if dataset in data:
            data[dataset].drop_duplicates(subset=subset, keep=strategy)
    return data


def convert_timezone(data: DataDict, timezone: str = "UTC") -> DataDict:
    for dataset in _select_datasets(data, None):
        df = data[dataset]
        if not isinstance(df.index, pd.DatetimeIndex):
            continue
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(timezone)
        else:
            df.index = df.index.tz_convert(timezone)
    return data


def convert_column_types(
    data: DataDict, datasets: list[str], columns: list[str], types: list[str]
) -> DataDict:
    dtypes = dict(zip(columns, types))
    for dataset in datasets:
        data[dataset] = data[dataset].astype(dtypes)
    return data


def filter_businesstype(data: DataDict, dataset: str, keep: str) -> DataDict:
    df = data[dataset]
    data[dataset] = df[df["businesstype"] == keep]
    return data


def drop_invalid_units_unavailability(data: DataDict, dataset: str) -> DataDict:
    df = data[dataset].copy()

    df = df.drop_duplicates(keep="first")
    df = df.drop_duplicates(subset=["mrid", "revision"], keep="first")
    df = df.reset_index()

    start = pd.to_datetime(df["start"], utc=True)
    end = pd.to_datetime(df["end"], utc=True)
    created = pd.to_datetime(df["created_doc_time"], utc=True)

    created_over_3_years_before_start = (start - created).dt.days > 3 * 365
    created_on_or_after_end = created >= end

    data[dataset] = df[~(created_over_3_years_before_start | created_on_or_after_end)]
    return data


def prepare_units_unavailability(
    data: DataDict, dataset: str, businesstype: str | None = None
) -> DataDict:
    data = drop_columns(data, datasets=[dataset], columns_list=[["docstatus"]])
    if businesstype is not None:
        data = filter_businesstype(data, dataset=dataset, keep=businesstype)
    data = convert_column_types(
        data,
        datasets=[dataset],
        columns=["avail_qty", "pstn"],
        types=["float", "float"],
    )
    data = drop_rows_missing_power_or_plant_type(data, datasets=[dataset])
    data = drop_invalid_units_unavailability(data, dataset=dataset)
    return data


def clean_pl(data: DataDict, cfg: dict = COUNTRY_CONFIG["PL"]) -> DataDict:
    data = drop_columns(
        data,
        datasets=["generation", "generation_wind_solar_forecast"],
        columns_list=[cfg["drop_gen_types"], cfg["drop_ws_forecast"]],
    )
    data = drop_datasets(data, datasets=[PRODUCTION_UNAVAILABILITY])
    data = prepare_units_unavailability(
        data,
        dataset=GENERATION_UNAVAILABILITY,
        businesstype="Planned maintenance",
    )
    data = drop_duplicate_index(data, datasets=["day_ahead_prices"], strategy="first")
    return data


def clean_de_lu(data: DataDict) -> DataDict:
    data = drop_datasets(data, datasets=[PRODUCTION_UNAVAILABILITY])
    data = prepare_units_unavailability(data, dataset=GENERATION_UNAVAILABILITY)
    return data


def clean_lt(data: DataDict, cfg: dict = COUNTRY_CONFIG["LT"]) -> DataDict:
    return drop_datasets(data, datasets=cfg["datasets_to_remove"])


def clean_fr(data: DataDict, cfg: dict = COUNTRY_CONFIG["FR"]) -> DataDict:
    for dataset in UNAVAILABILITY_DATASETS:
        data = prepare_units_unavailability(data, dataset=dataset)
    data = drop_columns(
        data,
        datasets=["generation_wind_solar_forecast"],
        columns_list=[cfg["drop_ws_forecast"]],
    )
    return data


def save_data_entsoe(
    name: str,
    data: pd.Series | pd.DataFrame,
    country_code: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    data_dir = DATA_DIR / "clean" / "entsoe" / country_code / f"{start_date}_{end_date}"
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / f"{name}.parquet"
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data.to_parquet(file_path)

    logger.info(f"Saved {file_path.relative_to(DATA_DIR)}")
