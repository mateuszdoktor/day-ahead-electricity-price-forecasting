import pandas as pd

ZONES = {
    "PL": "PL",
    "DE": "DE",
    "CZ": "CZ",
    "SK": "SK",
    "50Hz": "50Hertz",
}

SECOND_RING = [
    "AT",
    "BE",
    "FR",
    "HR",
    "HU",
    "NL",
    "RO",
    "SI",
]

TREES_PL = [
    "np_PL",
    "gen_PL",
    "min_PL",
    "max_PL",
    "ac_PL_import",
    "ac_PL_export",
    "headroom_import_PL",
    "headroom_export_PL",
    "np_PL_delta24",
    "gen_PL_delta24",
    "flow_DE_PL",
    "flow_CZ_PL",
    "flow_SK_PL",
    "flow_UA_PL",
]

TREES_FOREIGN = [
    "np_DE",
    "gen_DE",
    "gen_DE_delta24",
    "np_50Hz",
    "gen_50Hz",
    "vload_50Hz",
    "gen_50Hz_delta24",
    "np_CZ",
    "gen_CZ",
    "np_SK",
    "gen_SK",
]

NN_PL = [
    "np_PL",
    "gen_PL",
    "min_PL",
    "max_PL",
    "ac_PL_import",
    "ac_PL_export",
    "flow_DE_PL",
    "flow_CZ_PL",
    "flow_SK_PL",
    "flow_UA_PL",
]

NN_FOREIGN = [
    "np_DE",
    "gen_DE",
    "np_50Hz",
    "gen_50Hz",
    "vload_50Hz",
    "np_CZ",
    "gen_CZ",
    "np_SK",
    "gen_SK",
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

PCA_GROUPS = [
    "pca_np",
    "pca_gen",
    "pca_load",
]


def calculate_net_positions(jao: pd.DataFrame) -> dict[str, pd.Series]:
    f = {}

    for name, zone in ZONES.items():
        f[f"np_{name}"] = jao[f"d2cf_coreNetPosition_{zone}"]

    f["min_PL"] = jao["minmax_np_minPL"]
    f["max_PL"] = jao["minmax_np_maxPL"]
    f["headroom_import_PL"] = f["np_PL"] - f["min_PL"]
    f["headroom_export_PL"] = f["max_PL"] - f["np_PL"]
    f["np_PL_delta24"] = f["np_PL"].diff(24)

    return f


def calculate_generation_and_load(jao: pd.DataFrame) -> dict[str, pd.Series]:
    f = {}

    for name, zone in ZONES.items():
        f[f"gen_{name}"] = jao[f"d2cf_generation_{zone}"]

    f["vload_50Hz"] = jao["d2cf_verticalLoad_50Hertz"]

    for name in ["PL", "DE", "50Hz"]:
        f[f"gen_{name}_delta24"] = f[f"gen_{name}"].diff(24)

    return f


def calculate_borders(jao: pd.DataFrame) -> dict[str, pd.Series]:
    f = {}

    def flow(name):
        return jao[f"refprog_border_{name}"]

    signed = {
        "DE": flow("DE_PL"),
        "CZ": -flow("PL_CZ"),
        "SK": -flow("PL_SK"),
        "UA": -flow("PL_UA"),
    }
    for country, series in signed.items():
        f[f"flow_{country}_PL"] = series

    return f


def calculate_allocation_constraints(jao: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "ac_PL_import": jao["allocation_constraint_PL_import"],
        "ac_PL_export": jao["allocation_constraint_PL_export"],
    }


def calculate_pca_inputs(jao: pd.DataFrame) -> dict[str, pd.Series]:
    f = {}

    sources = {
        "pca_np": "d2cf_coreNetPosition",
        "pca_gen": "d2cf_generation",
        "pca_load": "d2cf_verticalLoad",
    }
    for group, source in sources.items():
        for zone in SECOND_RING:
            f[f"{group}_{zone}"] = jao[f"{source}_{zone}"]

    return f


def create_all_jao_features(jao: pd.DataFrame) -> pd.DataFrame:
    f = {}
    f.update(calculate_net_positions(jao))
    f.update(calculate_generation_and_load(jao))
    f.update(calculate_borders(jao))
    f.update(calculate_allocation_constraints(jao))
    f.update(calculate_pca_inputs(jao))

    df = pd.DataFrame(f, index=jao.index)
    df = df.replace([float("inf"), float("-inf")], float("nan"))

    columns = [
        *COLUMNS["trees"]["pl_foreign"],
        *[c for group in PCA_GROUPS for c in df.columns if c.startswith(f"{group}_")],
    ]
    return df[columns]


def select_features(features: pd.DataFrame, model: str, scope: str) -> pd.DataFrame:
    if model not in COLUMNS or scope not in COLUMNS[model]:
        raise KeyError((model, scope))

    return features[COLUMNS[model][scope]].copy()


def select_dataset(features: pd.DataFrame, model: str, scope: str) -> pd.DataFrame:
    selected = select_features(features, model, scope)
    if scope == "pl":
        return selected

    pca = [select_pca_group(features, group) for group in PCA_GROUPS]
    return pd.concat([selected, *pca], axis=1)


def select_pca_group(features: pd.DataFrame, group: str) -> pd.DataFrame:
    if group not in PCA_GROUPS:
        raise KeyError(group)

    columns = [c for c in features.columns if c.startswith(f"{group}_")]
    return features[columns].copy()
