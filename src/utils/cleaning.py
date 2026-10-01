import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# --- ENTSO-E ------------------------------

DataDict = dict[str, pd.DataFrame]

UNAVAILABILITY_DATASETS = [
    "generation_units_unavailability",
    "production_units_unavailability",
]

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


def clean_pl(data: DataDict, cfg: dict = COUNTRY_CONFIG["PL"]) -> DataDict:
    data = drop_columns(
        data,
        datasets=["generation", "generation_wind_solar_forecast"],
        columns_list=[cfg["drop_gen_types"], cfg["drop_ws_forecast"]],
    )
    data = drop_datasets(data, datasets=UNAVAILABILITY_DATASETS)
    data = drop_duplicate_index(data, datasets=["day_ahead_prices"], strategy="first")
    return data


def clean_de_lu(data: DataDict) -> DataDict:
    return drop_datasets(data, datasets=UNAVAILABILITY_DATASETS)


def clean_lt(data: DataDict, cfg: dict = COUNTRY_CONFIG["LT"]) -> DataDict:
    return drop_datasets(data, datasets=cfg["datasets_to_remove"])


def clean_fr(data: DataDict, cfg: dict = COUNTRY_CONFIG["FR"]) -> DataDict:
    data = drop_datasets(data, datasets=UNAVAILABILITY_DATASETS)
    data = drop_columns(
        data,
        datasets=["generation_wind_solar_forecast"],
        columns_list=[cfg["drop_ws_forecast"]],
    )
    return data


# --- Investing.com ------------------------------

INVESTING_COM_DATASETS = [
    "dutch_ttf_natural_gas_futures",
    "rotterdam_coal_futures",
    "german_power_baseload_futures",
    "eu_carbon_emissions_futures",
    "eur_pln",
]


def clean_investing_com_datasets(data: DataDict) -> DataDict:
    columns_to_drop = ["Date", "Open", "High", "Low", "Vol.", "Change %"]

    for dataset in INVESTING_COM_DATASETS:
        data = drop_columns(
            data,
            datasets=[dataset],
            columns_list=[columns_to_drop],
        )

    return data


# --- JAO -----------------------------

JAO_DATASETS = ["allocation_constraint", "d2cf", "minmax_np", "refprog"]


def clean_jao_datasets(data: DataDict) -> DataDict:
    data = drop_columns(
        data,
        datasets=["refprog"],
        columns_list=[
            [
                "border_DE_DE_DK1_VH",
                "border_DE_DK2_BigHub_DE",
                "border_NL_DK1_COBRA",
                "border_RO_RO_BG_VH",
                "border_NL_NL_NO2_NorNed",
                "border_DE_NO2_BigHub_DE",
                "border_DE_DE_SE4_Baltic",
                "border_PL_PL_SE4_SwePol",
                "border_PL_LT_BigHub_PL",
                "border_DK1_UK_Viking1",
                "border_DK1_UK_Viking2",
                "border_RO_MD",
                "border_UA_MD",
                "border_ES_MA_ESMA_link1",
                "border_ES_MA_ESMA_link2",
                "border_KS_ME",
                "border_SI_HU",
                "border_KS_AL",
                "border_MK_KS",
                "border_IT_MEMONITA2",
                "border_ME_ITMONITA2",
                "border_RS_KS",
            ]
        ],
    )

    data = drop_columns(
        data,
        datasets=["minmax_np"],
        columns_list=[
            [
                "minDE_NO2_BigHub",
                "maxDE_NO2_BigHub",
                "minNL_NO2_NorNed",
                "maxNL_NO2_NorNed",
                "minDE_DK2_BigHub",
                "maxDE_DK2_BigHub",
                "minDE_SE4_Baltic",
                "maxDE_SE4_Baltic",
                "minPL_LT_BigHub",
                "maxPL_LT_BigHub",
                "minPL_SE4_SwePol",
                "maxPL_SE4_SwePol",
                "minRO_BG_VH",
                "maxRO_BG_VH",
                "minNL_DK1_COBRA",
                "maxNL_DK1_COBRA",
                "minDE_DK1_VH",
                "maxDE_DK1_VH",
            ]
        ],
    )

    data = drop_columns(
        data,
        datasets=["allocation_constraint"],
        columns_list=[
            [
                "BE_export",
                "BE_import",
            ]
        ],
    )

    return data
