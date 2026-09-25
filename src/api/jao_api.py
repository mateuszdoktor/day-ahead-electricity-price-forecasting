import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from jao import (
    JaoPublicationToolPandasClient,
    JaoPublicationToolPandasNordics,
    JaoPublicationToolPandasParRun,
)

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

DATE_RANGE_DATASETS = {
    "allocation_constraint": "query_allocationconstraint",
    "refprog": "query_refprog",
    "d2cf": "query_d2cf",
}

DATASETS = {
    "allocation_constraint": "query_allocationconstraint",
    "refprog": "query_refprog",
    "d2cf": "query_d2cf",
    "minmax_np": "query_minmax_np",
}


def save_data(name: str, data: pd.Series | pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    file_path = output_dir / f"{name}.parquet"
    data.to_parquet(file_path)

    logger.info(f"Saved {file_path.relative_to(DATA_DIR)}")


class JaoClient:
    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("JAO_API_KEY")

        if api_key is None:
            raise ValueError("API Key for JAO not found in environmental variables")

        self.pandas_client = JaoPublicationToolPandasClient(api_key=api_key)
        self.pandas_client_nordics = JaoPublicationToolPandasNordics(api_key=api_key)

    def fetch(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        save_to_parquet: bool = False,
    ) -> dict[str, pd.DataFrame]:
        output_dir = (
            DATA_DIR / "raw" / "jao" / f"{start_date:%Y-%m-%d}_{end_date:%Y-%m-%d}"
        )

        batches = {name: [] for name in [*DATE_RANGE_DATASETS, "minmax_np"]}

        first_month = start_date.normalize().replace(day=1)

        for month in pd.date_range(first_month, end_date, freq="MS"):
            month_start = max(start_date, month)
            month_end = min(
                end_date, month + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1)
            )

            logger.info(f"Processing {month_start:%Y-%m}: {month_start} -> {month_end}")

            for name, method_name in DATE_RANGE_DATASETS.items():
                query = getattr(self.pandas_client, method_name)
                data = query(d_from=month_start, d_to=month_end)

                if isinstance(data, pd.Series):
                    data = data.to_frame()
                batches[name].append(data)

            days = pd.date_range(
                month_start.normalize(), month_end.normalize(), freq="D"
            )
            batches["minmax_np"].extend(
                self.pandas_client.query_minmax_np(day) for day in days
            )

        results = {}

        for name, parts in batches.items():
            parts = [part for part in parts if not part.empty]

            if not parts:
                logger.warning(f"No data returned for {name}")
                results[name] = pd.DataFrame()
                continue

            results[name] = pd.concat(parts)

            if save_to_parquet:
                save_data(name, results[name], output_dir)

        return results
