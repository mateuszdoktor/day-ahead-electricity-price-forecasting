from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from src.config import BASE_DIR
from src.utils.plots import FIG_WIDTH_IN, PALETTE, _panel_title, _plain

CHOSEN = {"d1": 112, "d2": 93, "d3": 112}
SEEDS = (1, 2, 3, 4, 5)
STEPS = ("d1", "d2", "d3")
BLOCKS = ("Poland", "market coupling", "foreign markets")
COLORS = {
    "Poland": PALETTE[0],
    "market coupling": PALETTE[1],
    "foreign markets": PALETTE[2],
}
OTHER_COLOR = "#999999"
SIDES = ("encoder", "decoder")
IMPORTANCE_LABEL = "Importance in %"
OTHER_LABEL = "other"


def load_shares(step, seed):
    saved = torch.load(
        BASE_DIR
        / "results"
        / "tft"
        / step
        / "test"
        / f"trial_{CHOSEN[step]}"
        / f"{seed}.pt",
        weights_only=False,
        map_location="cpu",
    )
    model = SimpleNamespace(
        hparams=SimpleNamespace(max_encoder_length=saved["max_encoder_length"]),
        static_variables=saved["static_variables"],
        encoder_variables=saved["encoder_variables"],
        decoder_variables=saved["decoder_variables"],
    )
    raw = TemporalFusionTransformer.interpret_output(
        model, saved["raw"], reduction="sum"
    )
    shares = {}
    for side in SIDES:
        weights = raw[f"{side}_variables"].detach().float()
        weights = weights / weights.sum() * 100
        names = list(getattr(model, f"{side}_variables"))
        shares[side] = pd.Series(weights.numpy(), index=names)
    return shares


def mean_shares(steps=STEPS):
    means = {}
    for step in steps:
        total = None
        for seed in SEEDS:
            shares = load_shares(step, seed)
            if total is None:
                total = shares
            else:
                total = {side: total[side] + shares[side] for side in SIDES}
        means[step] = {side: total[side] / len(SEEDS) for side in SIDES}
    return means


def _block_of(shares):
    present = {
        step: set(shares[step]["encoder"].index) | set(shares[step]["decoder"].index)
        for step in STEPS
    }

    def block_of(name):
        if name in present["d1"]:
            return "Poland"
        if name in present["d2"]:
            return "market coupling"
        return "foreign markets"

    return block_of


def _display_series(series, max_rows):
    if len(series) <= max_rows:
        return series.sort_values()
    ranked = series.sort_values(ascending=False)
    kept = ranked.iloc[: max_rows - 1]
    tail = ranked.iloc[max_rows - 1 :]
    other = pd.Series({f"{OTHER_LABEL} ({len(tail)})": tail.sum()})
    return pd.concat([kept, other]).sort_values()


def _block_legend(fig, blocks):
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color=COLORS[block]) for block in blocks],
        blocks,
        loc="outside lower center",
        ncol=len(blocks),
        frameon=False,
    )


def plot_blocks(shares, title=None):
    block_of = _block_of(shares)
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, 3.5), sharey=True, layout="constrained"
    )
    x = np.arange(len(STEPS))
    width = 0.24
    for ax, side in zip(axes, SIDES):
        for i, block in enumerate(BLOCKS):
            xs, values = [], []
            for j, step in enumerate(STEPS):
                series = shares[step][side]
                grouped = series.groupby(series.index.map(block_of)).sum()
                if block in grouped.index:
                    xs.append(x[j] + (i - 1) * width)
                    values.append(grouped[block])
            if not values:
                continue
            bars = ax.bar(
                xs,
                values,
                width,
                label=block,
                color=COLORS[block],
            )
            ax.bar_label(bars, fmt="%.0f%%", fontsize=7, padding=1)
        ax.set_xticks(x, [step.upper() for step in STEPS])
        _panel_title(ax, side.capitalize())
        ax.set_ylim(0, 115)
        _plain(ax)
    axes[0].set_ylabel(IMPORTANCE_LABEL)
    _block_legend(fig, BLOCKS)
    if title:
        fig.suptitle(title)
    plt.close(fig)
    return fig


def plot_step(shares, step, max_rows=20, title=None):
    block_of = _block_of(shares)
    displayed = {
        side: _display_series(shares[step][side], max_rows)
        for side in SIDES
    }
    value_max = max(series.max() for series in displayed.values())
    fig_height = max_rows * 0.30 + 1.1
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH_IN, fig_height),
        sharex=True,
        layout="constrained",
    )
    used = set()
    for ax, side in zip(axes, SIDES):
        series = displayed[side]
        n = len(series)
        offset = max_rows - n
        y = np.arange(n) + offset
        colors = []
        for name in series.index:
            if name.startswith(OTHER_LABEL):
                colors.append(OTHER_COLOR)
            else:
                block = block_of(name)
                used.add(block)
                colors.append(COLORS[block])
        bars = ax.barh(y, series.to_numpy(), color=colors)
        for rect, value in zip(bars, series.to_numpy()):
            ax.text(
                rect.get_width() + value_max * 0.015,
                rect.get_y() + rect.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=7,
            )
        ax.set_yticks(y, series.index)
        ax.set_ylim(-0.5, max_rows - 0.5)
        ax.set_xlim(0, value_max * 1.14)
        _panel_title(ax, side.capitalize())
        ax.set_xlabel(IMPORTANCE_LABEL)
        _plain(ax)
    labels = [block for block in BLOCKS if block in used]
    _block_legend(fig, labels)
    if title:
        fig.suptitle(title)
    plt.close(fig)
    return fig


GROUP_FEATURES = {
    "G1": ("PL_price", "PL_price_lag24", "PL_price_lag48", "PL_price_lag72", "PL_price_lag168"),
    "G2": (
        "PL_load_fc",
        "PL_load_fc_delta24",
        "PL_wind_fc",
        "PL_solar_fc",
        "PL_gen_fc",
        "PL_coal_gen_lag24",
    ),
    "G3": ("ttf_lag24", "eua_lag24", "coal_lag24", "eurpln_lag24"),
    "G4": ("hour", "dow", "month", "is_holiday", "is_bridge"),
    "G5": ("np_PL", "gen_PL", "vload_PL", "min_PL", "max_PL"),
    "G6": (
        "ac_PL_import",
        "ac_PL_export",
        "flow_DE_PL",
        "flow_CZ_PL",
        "flow_SK_PL",
        "flow_UA_PL",
    ),
    "G7": ("spread_PL_DE_LU", "spread_PL_CZ", "spread_PL_SK", "spread_PL_SE4", "spread_PL_LT"),
    "G8": ("DE_load_fc", "CZ_load_fc", "SK_load_fc", "SE4_load_fc", "LT_load_fc"),
    "G9": ("np_DE", "np_50Hz", "np_CZ", "np_SK"),
    "G10": ("flow_DE_CZ", "flow_CZ_SK"),
}

GROUP_BLOCK = {
    **{g: "Poland" for g in ("G1", "G2", "G3", "G4")},
    **{g: "market coupling" for g in ("G5", "G6")},
    **{g: "foreign markets" for g in ("G7", "G8", "G9", "G10")},
}

STEP_GROUPS = {
    "d1": ("G1", "G2", "G3", "G4"),
    "d2": ("G1", "G2", "G3", "G4", "G5", "G6"),
    "d3": ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"),
}

GROUP_NAMES = {
    "G1": "PL price history",
    "G2": "PL demand & generation",
    "G3": "Fuel, EUA & FX prices",
    "G4": "Calendar variables",
    "G5": "PL market parameters",
    "G6": "PL cross-border exchange",
    "G7": "Price spreads vs neighbours",
    "G8": "Foreign load forecasts",
    "G9": "Foreign net positions",
    "G10": "Foreign-to-foreign flows",
}


def _group_of(name):
    for group, features in GROUP_FEATURES.items():
        if name in features:
            return group
    raise KeyError(name)


def _group_label(group):
    return f"{GROUP_NAMES[group]} ({len(GROUP_FEATURES[group])})"


def plot_feature_groups(shares, step, title=None):
    present = {}
    for side in SIDES:
        series = shares[step][side]
        grouped = series.groupby(series.index.map(_group_of)).sum()
        groups = [g for g in STEP_GROUPS[step] if g in grouped.index]
        present[side] = (groups, [grouped[g] for g in groups])
    max_rows = max(len(groups) for groups, _ in present.values())
    value_max = max(max(values) for _, values in present.values())
    fig_height = max_rows * 0.4 + 1.2
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, fig_height), sharex=True, layout="constrained"
    )
    used = set()
    for ax, side in zip(axes, SIDES):
        groups, values = present[side]
        n = len(groups)
        y = (max_rows - 1) - np.arange(n)
        colors = []
        for group in groups:
            block = GROUP_BLOCK[group]
            used.add(block)
            colors.append(COLORS[block])
        bars = ax.barh(y, values, color=colors)
        for rect, value in zip(bars, values):
            ax.text(
                rect.get_width() + value_max * 0.02,
                rect.get_y() + rect.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=7,
            )
        ax.set_yticks(y, [_group_label(g) for g in groups])
        ax.set_ylim(-0.5, max_rows - 0.5)
        ax.set_xlim(0, value_max * 1.22)
        _panel_title(ax, side.capitalize())
        ax.set_xlabel(IMPORTANCE_LABEL)
        _plain(ax)
    labels = [block for block in BLOCKS if block in used]
    _block_legend(fig, labels)
    if title:
        fig.suptitle(title)
    plt.close(fig)
    return fig
