import pandas as pd

COAL_COLUMNS = [
    "generation_Fossil_Hard_coal",
    "generation_Fossil_Brown_coal_Lignite",
    "generation_Fossil_Coal_derived_gas",
]

DE_WIND_COLUMNS = [
    "generation_wind_solar_forecast_Wind_Onshore",
    "generation_wind_solar_forecast_Wind_Offshore",
]

LOAD_ZONES = {
    "DE_LU": "DE",
    "CZ": "CZ",
    "SK": "SK",
    "SE_4": "SE4",
    "LT": "LT",
}

SPREADS = {
    "DE_LU": "spread_PL_DE_LU_lag24",
    "CZ": "spread_PL_CZ_lag24",
    "SK": "spread_PL_SK_lag24",
    "SE_4": "spread_PL_SE4_lag24",
    "LT": "spread_PL_LT_lag24",
}

TREES_PL = [
    "PL_price_lag24",
    "PL_price_lag48",
    "PL_price_lag168",
    "PL_price_d1_mean",
    "PL_price_d1_min",
    "PL_price_d1_max",
    "PL_price_mean7d_same_hour",
    "PL_price_std7d",
    "PL_load_fc",
    "PL_load_fc_delta24",
    "PL_wind_fc_lag24",
    "PL_solar_fc_lag24",
    "PL_resload_persist",
    "PL_coal_gen_lag48",
]

TREES_FOREIGN = [
    "spread_PL_DE_LU_lag24",
    "spread_PL_CZ_lag24",
    "spread_PL_SK_lag24",
    "spread_PL_SE4_lag24",
    "spread_PL_LT_lag24",
    "DE_price_lag24",
    "DE_load_fc",
    "CZ_load_fc",
    "SK_load_fc",
    "SE4_load_fc",
    "LT_load_fc",
    "DE_wind_fc_lag24",
    "DE_solar_fc_lag24",
    "DE_resload_persist",
]

NN_PL = [
    "PL_price_lag24",
    "PL_price_lag48",
    "PL_price_lag168",
    "PL_load_fc",
    "PL_load_fc_delta24",
    "PL_wind_fc_lag24",
    "PL_solar_fc_lag24",
    "PL_coal_gen_lag48",
]

NN_FOREIGN = [
    "spread_PL_DE_LU_lag24",
    "spread_PL_CZ_lag24",
    "spread_PL_SK_lag24",
    "spread_PL_SE4_lag24",
    "spread_PL_LT_lag24",
    "DE_load_fc",
    "CZ_load_fc",
    "SK_load_fc",
    "SE4_load_fc",
    "LT_load_fc",
    "DE_wind_fc_lag24",
    "DE_solar_fc_lag24",
]

COLUMNS = {
    "trees": {
        "pl": TREES_PL,
        "pl_foreign": [*TREES_PL, *TREES_FOREIGN],
    },
    "nn": {
        "pl": NN_PL,
        "pl_foreign": [*NN_PL, *NN_FOREIGN],
    },
}


def previous_day(daily: pd.Series, index: pd.DatetimeIndex, n: int = 1) -> pd.Series:
    aligned = daily.shift(n).reindex(index.normalize())
    return pd.Series(aligned.to_numpy(), index=index)


def calculate_pl_price_features(price: pd.Series) -> dict[str, pd.Series]:
    day = price.index.normalize()
    by_day = price.groupby(day)

    f = {
        "PL_price_lag24": price.shift(24),
        "PL_price_lag48": price.shift(48),
        "PL_price_lag168": price.shift(168),
        "PL_price_d1_mean": previous_day(by_day.mean(), price.index),
        "PL_price_d1_min": previous_day(by_day.min(), price.index),
        "PL_price_d1_max": previous_day(by_day.max(), price.index),
        "PL_price_mean7d_same_hour": sum(price.shift(24 * k) for k in range(1, 8)) / 7,
    }

    roll_std = price.rolling(24 * 7).std()
    f["PL_price_std7d"] = previous_day(roll_std.groupby(day).last(), price.index)

    return f


def calculate_pl_load_and_generation(pl: pd.DataFrame) -> dict[str, pd.Series]:
    load_fc = pl["load_forecast"]
    wind = pl["generation_wind_solar_forecast_Wind_Onshore"].shift(24)
    solar = pl["generation_wind_solar_forecast_Solar"].shift(24)

    return {
        "PL_load_fc": load_fc,
        "PL_load_fc_delta24": load_fc - load_fc.shift(24),
        "PL_wind_fc_lag24": wind,
        "PL_solar_fc_lag24": solar,
        "PL_resload_persist": load_fc - wind - solar,
        "PL_coal_gen_lag48": pl[COAL_COLUMNS].sum(axis=1, min_count=1).shift(48),
    }


def calculate_foreign_features(
    data: dict[str, pd.DataFrame],
    pl_price_lag24: pd.Series,
) -> dict[str, pd.Series]:
    index = pl_price_lag24.index
    f = {}

    for zone, name in SPREADS.items():
        foreign_price = data[zone]["day_ahead_prices"].shift(24).reindex(index)
        f[name] = pl_price_lag24 - foreign_price

    de = data["DE_LU"]
    f["DE_price_lag24"] = de["day_ahead_prices"].shift(24).reindex(index)

    for zone, name in LOAD_ZONES.items():
        f[f"{name}_load_fc"] = data[zone]["load_forecast"].reindex(index)

    de_wind = de[DE_WIND_COLUMNS].sum(axis=1, min_count=1).shift(24).reindex(index)
    de_solar = de["generation_wind_solar_forecast_Solar"].shift(24).reindex(index)
    f["DE_wind_fc_lag24"] = de_wind
    f["DE_solar_fc_lag24"] = de_solar
    f["DE_resload_persist"] = f["DE_load_fc"] - de_wind - de_solar

    return f


def create_features(data_entsoe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    zones = {"PL", *LOAD_ZONES, *SPREADS}
    data = {
        zone: df.tz_convert("Europe/Warsaw")
        for zone, df in data_entsoe.items()
        if zone in zones
    }

    pl = data["PL"]
    f = {}
    f.update(calculate_pl_price_features(pl["day_ahead_prices"]))
    f.update(calculate_pl_load_and_generation(pl))
    f.update(calculate_foreign_features(data, f["PL_price_lag24"]))

    columns = COLUMNS["trees"]["pl_foreign"]
    features = pd.DataFrame(f, index=pl.index)
    features = features.replace([float("inf"), float("-inf")], float("nan"))

    return features[columns].copy().tz_convert("UTC")


def select_features(features: pd.DataFrame, model: str, scope: str) -> pd.DataFrame:
    if model not in COLUMNS or scope not in COLUMNS[model]:
        raise KeyError((model, scope))

    return features[COLUMNS[model][scope]].copy()
