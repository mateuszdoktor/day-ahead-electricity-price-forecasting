import pandas as pd

from src.config import LOCAL_TZ

CHANGE_SOURCES = ("ttf", "eua", "coal", "eurpln")
DELTA_COLUMNS = ("np_PL", "gen_PL", "vload_PL")
WINDOW = 24 * 7


def previous_day(daily: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    aligned = daily.shift(1).reindex(index.normalize())
    return pd.Series(aligned.to_numpy(), index=index)


def calculate_price_history(price: pd.Series) -> dict[str, pd.Series]:
    day = price.index.normalize()
    by_day = price.groupby(day)
    roll_std = price.rolling(WINDOW).std()

    return {
        "PL_price_d1_mean": previous_day(by_day.mean(), price.index),
        "PL_price_d1_min": previous_day(by_day.min(), price.index),
        "PL_price_d1_max": previous_day(by_day.max(), price.index),
        "PL_price_mean7d_same_hour": sum(price.shift(24 * k) for k in range(1, 8)) / 7,
        "PL_price_std7d": previous_day(roll_std.groupby(day).last(), price.index),
    }


def calculate_poland_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    f = calculate_price_history(df["PL_price"])
    f["PL_resload_persist"] = (
        df["PL_load_fc"] - df["PL_wind_fc_lag24"] - df["PL_solar_fc_lag24"]
    )

    for name in CHANGE_SOURCES:
        f[f"{name}_chg7d"] = df[f"{name}_lag48"].pct_change(WINDOW, fill_method=None)

    return f


def calculate_market_coupling_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    f = {
        "headroom_import_PL": df["np_PL"] - df["min_PL"],
        "headroom_export_PL": df["max_PL"] - df["np_PL"],
        "gen_vload_balance_PL": df["gen_PL"] - df["vload_PL"],
    }

    for name in DELTA_COLUMNS:
        f[f"{name}_delta24"] = df[name].diff(24)

    return f


def build_tree_features(tft: pd.DataFrame, step: str) -> pd.DataFrame:
    """Add tree-friendly columns to a TFT frame (UTC index, no calendar columns)."""
    df = tft.tz_convert(LOCAL_TZ)

    f = calculate_poland_features(df)
    if step != "d1":
        f.update(calculate_market_coupling_features(df))

    out = pd.concat([df, pd.DataFrame(f, index=df.index)], axis=1)
    out = out.replace([float("inf"), float("-inf")], float("nan"))

    return out.tz_convert("UTC")
