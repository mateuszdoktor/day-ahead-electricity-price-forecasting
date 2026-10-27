from pathlib import Path

import numpy as np
import pandas as pd
from epftoolbox.models import LEAR

from src.config import BASE_DIR

DNN_DATASETS_DIR = BASE_DIR / "data" / "dnn" / "datasets"
DNN_HYPERPARAMS_DIR = BASE_DIR / "data" / "dnn" / "hyperparameters"


def forecast_path(model, split="val", run="screening", family="lear"):
    # results/<family>/<split>/<run>/<model>.csv

    path = BASE_DIR / "results" / family / split / run / f"{model}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def select_exogenous(df: pd.DataFrame, exog: list[str] | None = None) -> pd.DataFrame:
    exog = exog or []
    out = pd.DataFrame({"Price": df["PL_price"]}, index=df.index)
    for i, column in enumerate(exog, start=1):
        out[f"Exogenous {i}"] = df[column]
    return out


def evaluate_lear(
    dataset: pd.DataFrame,
    calibration_window: int,
    exog: list[str] | None,
    forecast_start,
    forecast_end,
    path: Path | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    df = select_exogenous(dataset, exog)
    start = pd.Timestamp(forecast_start).normalize()
    end = pd.Timestamp(forecast_end).normalize()
    days = pd.date_range(start, end, freq="D")
    preds = []
    for i, day in enumerate(days, start=1):
        if progress:
            print(f"[{i}/{len(days)}] {day.date()}")
        y_hat = LEAR().recalibrate_and_forecast_next_day(
            df=df,
            calibration_window=calibration_window,
            next_day_date=day,
        )
        preds.append(np.asarray(y_hat).reshape(-1))

    index = pd.date_range(start, periods=len(days) * 24, freq="h")
    pred_df = pd.DataFrame({"Pred": np.concatenate(preds)}, index=index)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(path)
    return pred_df


# def evaluate_lear_screening(
#     dataset: pd.DataFrame,
#     calibration_window: int,
#     exog: list[str] | None,
#     forecast_starts,
#     n_days: int = 7,
#     model: str | None = None,
#     path: Path | None = None,
# ) -> pd.DataFrame:

#     frames = []
#     starts = list(forecast_starts)
#     for i, start in enumerate(starts, start=1):
#         start_ts = pd.Timestamp(start).normalize()
#         end_ts = start_ts + pd.Timedelta(days=n_days - 1)
#         label = model or "LEAR"
#         print(f"[{i}/{len(starts)}] {label} {start_ts.date()} -> {end_ts.date()}")
#         frames.append(
#             evaluate_lear(
#                 dataset=dataset,
#                 calibration_window=calibration_window,
#                 exog=exog,
#                 forecast_start=start_ts,
#                 forecast_end=end_ts,
#                 progress=False,
#             )
#         )

#     preds = pd.concat(frames).sort_index()
#     if path is not None:
#         path.parent.mkdir(parents=True, exist_ok=True)
#         preds.to_csv(path)
#     return preds


def _save_dnn_dataset(dataset, exog, name):
    df = select_exogenous(dataset, exog)
    DNN_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DNN_DATASETS_DIR / f"{name}.csv")
    return df


def tune_dnn(
    dataset,
    exog,
    calibration_window,
    experiment_id,
    max_evals=1500,
    new_hyperopt=1,
    nlayers=2,
    years_test=2,
    shuffle_train=1,
    data_augmentation=0,
    test_days=14,
):
    from epftoolbox.models import hyperparameter_optimizer

    df = _save_dnn_dataset(dataset, exog, experiment_id)
    test_end = df.index.max().normalize()
    test_start = test_end - pd.Timedelta(days=test_days)
    print(f"{experiment_id} hyperopt, {max_evals} evals")
    hyperparameter_optimizer(
        path_datasets_folder=str(DNN_DATASETS_DIR),
        path_hyperparameters_folder=str(DNN_HYPERPARAMS_DIR),
        new_hyperopt=new_hyperopt,
        max_evals=max_evals,
        nlayers=nlayers,
        dataset=experiment_id,
        years_test=years_test,
        calibration_window=calibration_window,
        shuffle_train=shuffle_train,
        data_augmentation=data_augmentation,
        experiment_id=experiment_id,
        begin_test_date=test_start,
        end_test_date=test_end,
    )


def _dnn_model(
    experiment_id,
    calibration_window,
    nlayers=2,
    years_test=2,
    shuffle_train=1,
    data_augmentation=0,
):
    from epftoolbox.models import DNN

    return DNN(
        experiment_id=experiment_id,
        path_hyperparameter_folder=str(DNN_HYPERPARAMS_DIR),
        nlayers=nlayers,
        dataset=experiment_id,
        years_test=years_test,
        shuffle_train=shuffle_train,
        data_augmentation=data_augmentation,
        calibration_window=calibration_window,
    )


def _forecast_day_with_retry(model, df, day, retries: int = 3):
    from tensorflow.keras import backend as K

    for attempt in range(1, retries + 1):
        try:
            return model.recalibrate_and_forecast_next_day(df=df, next_day_date=day)
        except Exception as error:
            K.clear_session()
            if attempt == retries:
                raise
            print(
                f"{day.date()} failed ({type(error).__name__}), "
                f"retry {attempt}/{retries - 1}"
            )


def evaluate_dnn(
    dataset,
    calibration_window,
    exog,
    forecast_start,
    forecast_end,
    experiment_id,
    path: Path | None = None,
    progress: bool = True,
    nlayers=2,
    years_test=2,
    shuffle_train=1,
    data_augmentation=0,
    retries: int = 3,
) -> pd.DataFrame:
    df = select_exogenous(dataset, exog)
    model = _dnn_model(
        experiment_id,
        calibration_window,
        nlayers=nlayers,
        years_test=years_test,
        shuffle_train=shuffle_train,
        data_augmentation=data_augmentation,
    )
    start = pd.Timestamp(forecast_start).normalize()
    end = pd.Timestamp(forecast_end).normalize()
    days = pd.date_range(start, end, freq="D")
    preds = []
    for i, day in enumerate(days, start=1):
        if progress:
            print(f"[{i}/{len(days)}] {day.date()}")
        y_hat = _forecast_day_with_retry(model, df, day, retries=retries)
        preds.append(np.asarray(y_hat).reshape(-1))

    index = pd.date_range(start, periods=len(days) * 24, freq="h")
    pred_df = pd.DataFrame({"Pred": np.concatenate(preds)}, index=index)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(path)
    return pred_df


# def evaluate_dnn_screening(
#     dataset,
#     calibration_window,
#     exog,
#     forecast_starts,
#     experiment_id,
#     n_days: int = 7,
#     model: str | None = None,
#     path: Path | None = None,
#     nlayers=2,
#     years_test=2,
#     shuffle_train=1,
#     data_augmentation=0,
# ) -> pd.DataFrame:
#     frames = []
#     starts = list(forecast_starts)
#     label = model or experiment_id
#     for i, start in enumerate(starts, start=1):
#         start_ts = pd.Timestamp(start).normalize()
#         end_ts = start_ts + pd.Timedelta(days=n_days - 1)
#         print(f"[{i}/{len(starts)}] {label} {start_ts.date()} -> {end_ts.date()}")
#         frames.append(
#             evaluate_dnn(
#                 dataset=dataset,
#                 calibration_window=calibration_window,
#                 exog=exog,
#                 forecast_start=start_ts,
#                 forecast_end=end_ts,
#                 experiment_id=experiment_id,
#                 progress=False,
#                 nlayers=nlayers,
#                 years_test=years_test,
#                 shuffle_train=shuffle_train,
#                 data_augmentation=data_augmentation,
#             )
#         )

#     preds = pd.concat(frames).sort_index()
#     if path is not None:
#         path.parent.mkdir(parents=True, exist_ok=True)
#         preds.to_csv(path)
#     return preds
