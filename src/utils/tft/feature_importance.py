from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from src.config import BASE_DIR

CHOSEN = {"d1": 114, "d2": 75, "d3": 99}
SEEDS = (1, 2, 3, 4, 5)
STEPS = ("d1", "d2", "d3")
BLOCKS = ("Poland", "market coupling", "foreign markets")
COLORS = {"Poland": "C0", "market coupling": "C1", "foreign markets": "C2"}
SIDES = ("encoder", "decoder")


def load_shares(step, seed):
    saved = torch.load(
        BASE_DIR / "results" / "tft" / step / "test" / f"trial_{CHOSEN[step]}" / f"{seed}.pt",
        weights_only=False,
        map_location="cpu",
    )
    model = SimpleNamespace(
        hparams=SimpleNamespace(max_encoder_length=saved["max_encoder_length"]),
        static_variables=saved["static_variables"],
        encoder_variables=saved["encoder_variables"],
        decoder_variables=saved["decoder_variables"],
    )
    raw = TemporalFusionTransformer.interpret_output(model, saved["raw"], reduction="sum")
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


def _plain(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_blocks(shares):
    block_of = _block_of(shares)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, layout="constrained")
    x = np.arange(len(STEPS))
    width = 0.24
    for ax, side in zip(axes, SIDES):
        for i, block in enumerate(BLOCKS):
            values = []
            for step in STEPS:
                series = shares[step][side]
                grouped = series.groupby(series.index.map(block_of)).sum()
                values.append(grouped.get(block, 0))
            ax.bar(x + (i - 1) * width, values, width, label=block, color=COLORS[block])
        ax.set_xticks(x, STEPS)
        ax.set_title(side.capitalize())
        ax.set_ylim(0, 105)
        _plain(ax)
    axes[0].set_ylabel("Importance in %")
    axes[1].legend(frameon=False, loc="upper right")
    return fig


def plot_step(shares, step):
    block_of = _block_of(shares)
    n = max(len(shares[step][side]) for side in SIDES)
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 0.26 * n + 0.8), sharex=True, layout="constrained"
    )
    used = set()
    for ax, side in zip(axes, SIDES):
        series = shares[step][side].sort_values()
        used.update(block_of(name) for name in series.index)
        ax.barh(
            np.arange(len(series)),
            series.to_numpy(),
            color=[COLORS[block_of(name)] for name in series.index],
        )
        ax.set_yticks(np.arange(len(series)), series.index, fontsize=8)
        ax.set_title(side.capitalize())
        ax.set_xlabel("Importance in %")
        _plain(ax)
    labels = [block for block in BLOCKS if block in used]
    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color=COLORS[block]) for block in labels],
        labels,
        frameon=False,
        loc="outside lower center",
        ncol=len(labels),
    )
    return fig
