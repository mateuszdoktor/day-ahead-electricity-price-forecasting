import logging
import os

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

DATASETS = {
    "day_ahead_prices": ("query_day_ahead_prices", {}),
    "load": ("query_load", {}),
    "load_forecast": ("query_load_forecast", {}),
    "generation": ("query_generation", {}),
    "generation_forecast": ("query_generation_forecast", {}),
    "generation_wind_solar_forecast": ("query_wind_and_solar_forecast", {}),
    "generation_units_unavailability": ("query_unavailability_of_generation_units", {}),
    "production_units_unavailability": ("query_unavailability_of_production_units", {}),
    "water_reservoirs_hydro_storage": (
        "query_aggregate_water_reservoirs_and_hydro_storage",
        {},
    ),
}


def save_data(
    name: str,
    data: pd.Series | pd.DataFrame,
    country_code: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    data_dir = (
        DATA_DIR
        / "raw"
        / "entsoe"
        / country_code
        / f"{start_date:%Y-%m-%d}_{end_date:%Y-%m-%d}"
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / f"{name}.parquet"
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data.to_parquet(file_path)

    logger.info(f"Saved {file_path.relative_to(DATA_DIR)}")


class EntsoeClient:

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("ENTSOE_API_KEY")

        if api_key is None:
            raise AttributeError(
                "API Key for ENTSO-E not found in environmental variables"
            )

        self.pandas_client = EntsoePandasClient(api_key=api_key)

    def fetch(
        self,
        country_code: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        datasets: list[str] | None = None,
        fetch_all: bool = False,
        save_to_parquet: bool = False,
    ) -> dict[str, pd.Series | pd.DataFrame]:
        names = (
            [n for n in DATASETS if n != "water_reservoirs_hydro_storage"]
            if fetch_all
            else datasets
        )
        results = {}

        for i, name in enumerate(names, start=1):
            method, kwargs = DATASETS[name]
            query = getattr(self.pandas_client, method)
            results[name] = query(
                country_code=country_code, start=start_date, end=end_date, **kwargs
            )

            if save_to_parquet:
                save_data(name, results[name], country_code, start_date, end_date)

            logger.info(f"Processed {i}/{len(names)}: {name}")

        return results
