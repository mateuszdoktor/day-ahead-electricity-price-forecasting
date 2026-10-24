import pandas as pd

ZONES = {
    "PL": "PL",
    "DE": "DE",
    "CZ": "CZ",
    "SK": "SK",
    "50Hz": "50Hertz",
}


TFT_D2 = [
    "np_PL",
    "gen_PL",
    "vload_PL",
    "ac_PL_import",
    "ac_PL_export",
    "flow_DE_PL",
    "flow_CZ_PL",
    "flow_SK_PL",
    "flow_UA_PL",
    "min_PL",
    "max_PL",
]

TFT_D3 = [
    "np_DE",
    "np_50Hz",
    "np_CZ",
    "np_SK",
    "flow_DE_CZ",
    "flow_CZ_SK",
]

COLUMNS = {
    "tft": {
        "d2": TFT_D2,
        "d3": TFT_D3,
    },
}


def calculate_net_positions(jao: pd.DataFrame) -> dict[str, pd.Series]:
    f = {}

    for name, zone in ZONES.items():
        f[f"np_{name}"] = jao[f"d2cf_coreNetPosition_{zone}"]

    f["min_PL"] = jao["minmax_np_minPL"]
    f["max_PL"] = jao["minmax_np_maxPL"]

    return f


def calculate_generation_and_load(jao: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "gen_PL": jao["d2cf_generation_PL"],
        "vload_PL": jao["d2cf_verticalLoad_PL"],
    }


def calculate_borders(jao: pd.DataFrame) -> dict[str, pd.Series]:
    def flow(name):
        return jao[f"refprog_border_{name}"]

    return {
        "flow_DE_PL": flow("DE_PL"),
        "flow_CZ_PL": -flow("PL_CZ"),
        "flow_SK_PL": -flow("PL_SK"),
        "flow_UA_PL": -flow("PL_UA"),
        "flow_DE_CZ": flow("DE_CZ"),
        "flow_CZ_SK": flow("CZ_SK"),
    }


def calculate_allocation_constraints(jao: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "ac_PL_import": jao["allocation_constraint_PL_import"],
        "ac_PL_export": jao["allocation_constraint_PL_export"],
    }


def create_all_jao_features(jao: pd.DataFrame) -> pd.DataFrame:
    f = {}
    f.update(calculate_net_positions(jao))
    f.update(calculate_generation_and_load(jao))
    f.update(calculate_borders(jao))
    f.update(calculate_allocation_constraints(jao))

    df = pd.DataFrame(f, index=jao.index)
    df = df.replace([float("inf"), float("-inf")], float("nan"))

    return df


def select_features(features: pd.DataFrame, model: str, step: str) -> pd.DataFrame:
    if model not in COLUMNS or step not in COLUMNS[model]:
        raise KeyError((model, step))

    return features[COLUMNS[model][step]].copy()
