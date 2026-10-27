from pathlib import Path

import numpy as np
import pandas as pd
from epftoolbox.evaluation import MAE, RMSE, rMAE

_METRICS = {
    "MAE": MAE,
    "RMSE": RMSE,
    "rMAE": lambda real, pred: rMAE(real, pred, m="W"),
}


def score_forecast(prices, forecast, metrics=("MAE", "RMSE")):
    real = prices.loc[forecast.index]
    if isinstance(real, pd.Series):
        real = real.to_frame()
    return pd.Series({name: _METRICS[name](real, forecast) for name in metrics})


def compare_models(forecasts, prices, metrics=("MAE", "RMSE")):
    scores = {
        model: score_forecast(prices, forecasts[[model]], metrics)
        for model in forecasts.columns
    }
    return pd.DataFrame(scores).T.rename_axis("model")


def compare_by_window(forecasts, prices, starts, n_days=7, metrics=("MAE", "RMSE")):
    rows = []
    for start in starts:
        start = pd.Timestamp(start).normalize()
        end = start + pd.Timedelta(hours=n_days * 24 - 1)
        scores = compare_models(forecasts.loc[start:end], prices, metrics)
        scores.insert(0, "window_start", start)
        rows.append(scores.reset_index())
    return pd.concat(rows, ignore_index=True)


def as_daily(values):
    return np.asarray(values).reshape(-1, 24)


def as_daily_inputs(prices, forecasts):
    real = as_daily(prices.loc[forecasts.index])
    preds = {model: as_daily(forecasts[model]) for model in forecasts.columns}
    p_real = pd.DataFrame(real.reshape(-1), columns=["Price"])
    pred_df = pd.DataFrame(
        {model: values.reshape(-1) for model, values in preds.items()}
    )
    return p_real, pred_df


def load_forecasts(paths):
    frames = []
    for path in paths:
        path = Path(path)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [path.stem]
        frames.append(df)
    return pd.concat(frames, axis=1).sort_index()
