import pandas as pd

SOURCES = {
    "ttf": "dutch_ttf_natural_gas_futures",
    "eua": "eu_carbon_emissions_futures",
    "coal": "rotterdam_coal_futures",
    "eurpln": "eur_pln",
}

LAG = 48

TFT_D1 = [
    "ttf_lag48",
    "eua_lag48",
    "coal_lag48",
    "eurpln_lag48",
]

COLUMNS = {
    "tft": {
        "d1": TFT_D1,
    },
}


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(
        {f"{name}_lag48": df[source].shift(LAG) for name, source in SOURCES.items()},
        index=df.index,
    )

    return features.replace([float("inf"), float("-inf")], float("nan"))


def select_features(features: pd.DataFrame, model: str, step: str) -> pd.DataFrame:
    if model not in COLUMNS or step not in COLUMNS[model]:
        raise KeyError((model, step))

    return features[COLUMNS[model][step]].copy()
