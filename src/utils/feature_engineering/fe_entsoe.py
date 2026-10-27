import pandas as pd

COAL_COLUMNS = [
    "generation_Fossil_Hard_coal",
    "generation_Fossil_Brown_coal_Lignite",
    "generation_Fossil_Coal_derived_gas",
]

LOAD_ZONES = {
    "DE_LU": "DE",
    "CZ": "CZ",
    "SK": "SK",
    "SE_4": "SE4",
    "LT": "LT",
}

SPREADS = {
    "DE_LU": "spread_PL_DE_LU",
    "CZ": "spread_PL_CZ",
    "SK": "spread_PL_SK",
    "SE_4": "spread_PL_SE4",
    "LT": "spread_PL_LT",
}

TFT_D1 = [
    "PL_price_lag24",
    "PL_price_lag48",
    "PL_price_lag168",
    "PL_load_fc",
    "PL_load_fc_delta24",
    "PL_wind_fc",
    "PL_solar_fc",
    "PL_gen_fc",
    "PL_coal_gen_lag24",
]

TFT_D3 = [
    "spread_PL_DE_LU",
    "spread_PL_CZ",
    "spread_PL_SK",
    "spread_PL_SE4",
    "spread_PL_LT",
    "DE_load_fc",
    "CZ_load_fc",
    "SK_load_fc",
    "SE4_load_fc",
    "LT_load_fc",
]

COLUMNS = {
    "tft": {
        "d1": TFT_D1,
        "d3": TFT_D3,
    },
}


def calculate_pl_features(pl: pd.DataFrame) -> dict[str, pd.Series]:
    price = pl["day_ahead_prices"]
    load_fc = pl["load_forecast"]

    return {
        "PL_price_lag24": price.shift(24),
        "PL_price_lag48": price.shift(48),
        "PL_price_lag168": price.shift(168),
        "PL_load_fc": load_fc,
        "PL_load_fc_delta24": load_fc - load_fc.shift(24),
        "PL_wind_fc": pl["generation_wind_solar_forecast_Wind_Onshore"],
        "PL_solar_fc": pl["generation_wind_solar_forecast_Solar"],
        "PL_gen_fc": pl["generation_forecast"],
        "PL_coal_gen_lag24": pl[COAL_COLUMNS].sum(axis=1, min_count=1).shift(24),
    }


def calculate_foreign_features(
    data: dict[str, pd.DataFrame],
    pl_price: pd.Series,
) -> dict[str, pd.Series]:
    index = pl_price.index
    f = {}

    for zone, name in SPREADS.items():
        foreign_price = data[zone]["day_ahead_prices"].reindex(index)
        f[name] = pl_price - foreign_price

    for zone, name in LOAD_ZONES.items():
        f[f"{name}_load_fc"] = data[zone]["load_forecast"].reindex(index)

    return f


def create_features(data_entsoe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    zones = {"PL", *LOAD_ZONES, *SPREADS}
    data = {
        zone: df.tz_convert("Europe/Warsaw")
        for zone, df in data_entsoe.items()
        if zone in zones
    }

    pl = data["PL"]
    f = calculate_pl_features(pl)
    f.update(calculate_foreign_features(data, pl["day_ahead_prices"]))

    features = pd.DataFrame(f, index=pl.index)
    features = features.replace([float("inf"), float("-inf")], float("nan"))

    return features.tz_convert("UTC")


def select_features(features: pd.DataFrame, model: str, step: str) -> pd.DataFrame:
    if model not in COLUMNS or step not in COLUMNS[model]:
        raise KeyError((model, step))

    return features[COLUMNS[model][step]].copy()
