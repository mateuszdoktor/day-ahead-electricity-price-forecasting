from pathlib import Path
import logging

import pandas as pd

from src.config import ENTSOE_BIDDING_ZONE_CODES, DATA_DIR, BASE_DIR


logger = logging.getLogger(__name__)


DataDict = dict[str, pd.DataFrame]


def load_entsoe_data(
    data_dir: Path,
    date_range: str,
) -> dict[str, DataDict]:
    return {
        bz: {
            file.stem: pd.read_parquet(file)
            for file in (data_dir / bz / date_range).iterdir()
        }
        for bz in ENTSOE_BIDDING_ZONE_CODES
    }


def load_jao_data(
    data_dir: Path,
) -> DataDict:
    return {file.stem: pd.read_parquet(file) for file in data_dir.iterdir()}


def load_investing_data(
    data_dir: Path,
) -> DataDict:
    data = {}

    for file in data_dir.iterdir():
        df = pd.read_csv(file)
        df.index = pd.to_datetime(df["Date"], utc=True)
        data[file.stem] = df

    return data


def save_data(
    data: DataDict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, df in data.items():
        if isinstance(df, pd.Series):
            df = df.to_frame()

        file_path = output_dir / f"{dataset}.parquet"
        df.to_parquet(file_path)

        logger.info("Saved %s", file_path.relative_to(DATA_DIR))


def save_entsoe_data(
    data: dict[str, DataDict],
    output_dir: Path,
    date_range: str,
) -> None:
    for country_code, country_dataset in data.items():
        save_data(
            country_dataset,
            output_dir / country_code / date_range,
        )


def inspect_jao_files(
    output_dir: Path,
) -> None:
    file_paths = list(output_dir.iterdir())
    saved_count = len(file_paths)

    print(f"\n--- JAO ({saved_count} file/s) " + "-" * 30)

    for fp in file_paths:
        df = pd.read_parquet(fp)
        rel_path = fp.relative_to(BASE_DIR)

        print(
            f"  [FILE] {rel_path} | shape: {df.shape} | "
            f"idx: {df.index.min()}-{df.index.max()}"
        )


def inspect_investing_files(
    output_dir: Path,
) -> None:
    file_paths = list(output_dir.iterdir())
    saved_count = len(file_paths)

    print(f"\n--- Investing.com ({saved_count} file/s) " + "-" * 30)

    for fp in file_paths:
        df = pd.read_parquet(fp)
        rel_path = fp.relative_to(BASE_DIR)

        print(
            f"  [FILE] {rel_path} | shape: {df.shape} | "
            f"idx: {df.index.min()}-{df.index.max()}"
        )


def inspect_entsoe_files(
    output_dir: Path,
    date_range: str,
) -> None:
    for bidding_zone in ENTSOE_BIDDING_ZONE_CODES:
        data_dir = output_dir / bidding_zone / date_range
        file_paths = list(data_dir.iterdir())

        print(f"\n--- {bidding_zone} ({len(file_paths)} file/s) " + "-" * 30)

        for fp in file_paths:
            df = pd.read_parquet(fp)
            rel_path = fp.relative_to(BASE_DIR)

            print(
                f"  [FILE] {rel_path} | shape: {df.shape} | "
                f"idx: {df.index.min()}-{df.index.max()}"
            )
