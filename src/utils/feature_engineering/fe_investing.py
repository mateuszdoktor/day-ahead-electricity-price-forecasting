import pandas as pd

SOURCES = {
    "ttf": "dutch_ttf_natural_gas_futures",
    "eua": "eu_carbon_emissions_futures",
    "coal": "rotterdam_coal_futures",
    "eurpln": "eur_pln",
    "de_baseload": "german_power_baseload_futures",
}

CHANGE_SOURCES = ("ttf", "eua", "coal", "de_baseload")
LAG = 48
CHANGE_WINDOW = 24 * 7

TREE_COLUMNS = [
    "ttf_lag48",
    "eua_lag48",
    "coal_lag48",
    "ttf_chg7d",
    "eua_chg7d",
    "coal_chg7d",
    "de_baseload_lag48",
    "de_baseload_chg7d",
]

NN_COLUMNS = [
    "ttf_lag48",
    "eua_lag48",
    "coal_lag48",
    "eurpln_lag48",
    "de_baseload_lag48",
]


def calculate_price_levels(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        f"{name}_lag48": df[source].shift(LAG) for name, source in SOURCES.items()
    }


def calculate_price_changes(levels: dict[str, pd.Series]) -> dict[str, pd.Series]:
    return {
        f"{name}_chg7d": levels[f"{name}_lag48"].pct_change(
            CHANGE_WINDOW, fill_method=None
        )
        for name in CHANGE_SOURCES
    }


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    levels = calculate_price_levels(df)
    changes = calculate_price_changes(levels)

    columns = list(dict.fromkeys([*TREE_COLUMNS, *NN_COLUMNS]))
    features = pd.DataFrame({**levels, **changes}, index=df.index)
    features = features.replace([float("inf"), float("-inf")], float("nan"))

    return features[columns].copy()


def select_tree_features(features: pd.DataFrame) -> pd.DataFrame:
    return features[TREE_COLUMNS].copy()


def select_nn_features(features: pd.DataFrame) -> pd.DataFrame:
    return features[NN_COLUMNS].copy()
