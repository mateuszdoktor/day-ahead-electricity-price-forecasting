import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PAGE_WIDTH_CM = 17.0
FIG_WIDTH_IN = PAGE_WIDTH_CM / 2.54

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Nimbus Roman",
            "Liberation Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "figure.titlesize": 11,
        "figure.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.handlelength": 1.6,
        "legend.labelspacing": 0.35,
        "axes.edgecolor": "#3f3f3f",
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        "grid.color": "#999999",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.4,
    }
)

STEPS = ("d1", "d2", "d3")

PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=list(PALETTE))

_TEST_CMAP = LinearSegmentedColormap.from_list(
    "test_pvalues",
    [
        (0.00, PALETTE[2]),
        (0.50, PALETTE[6]),
        (0.99, PALETTE[3]),
        (1.00, PALETTE[7]),
    ],
)

_NAMED_COLORS = {
    "d1": PALETTE[0],
    "d2": PALETTE[1],
    "d3": PALETTE[2],
    "lear": PALETTE[3],
    "dnn": PALETTE[4],
    "ensemble": PALETTE[5],
    "ens": PALETTE[5],
    "base": PALETTE[6],
    "base + recal.": PALETTE[6],
}
_NAME_TOKENS = ("d1", "d2", "d3", "lear", "dnn")
_auto_colors = {}


def _series_color(name):
    key = str(name).strip().lower()
    if key in _NAMED_COLORS:
        return _NAMED_COLORS[key]
    for token in _NAME_TOKENS:
        if re.search(rf"(?<![a-z0-9]){token}(?![a-z0-9])", key):
            return _NAMED_COLORS[token]
    if name not in _auto_colors:
        taken = set(_NAMED_COLORS.values()) | set(_auto_colors.values())
        free = [c for c in PALETTE if c not in taken] or list(PALETTE)
        _auto_colors[name] = free[len(_auto_colors) % len(free)]
    return _auto_colors[name]


def _plain(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)


def _names(frame, steps):
    return list(frame.columns if steps is None else steps)


def _panel_title(ax, text):
    ax.set_title(text, fontsize=plt.rcParams["axes.labelsize"], fontweight="normal")


def _test_pvalues(real, forecasts, test, norm):
    models = list(forecasts.columns)
    predictions = {
        model: np.asarray(forecasts[model]).reshape(-1, 24) for model in models
    }
    values = np.ones((len(models), len(models)), dtype=float)
    for row, model_1 in enumerate(models):
        for column, model_2 in enumerate(models):
            if model_1 == model_2:
                continue
            result = test(
                p_real=real,
                p_pred_1=predictions[model_1],
                p_pred_2=predictions[model_2],
                norm=norm,
                version="multivariate",
            )
            values[row, column] = np.asarray(result).item()
    return values


def plot_tests(forecasts, prices, title=None, norm=1):
    from epftoolbox.evaluation import DM, GW

    real = np.asarray(prices.loc[forecasts.index]).reshape(-1, 24)
    tests = (("DM test", DM), ("GW test", GW))
    matrices = [_test_pvalues(real, forecasts, test, norm) for _, test in tests]
    models = list(forecasts.columns)
    positions = np.arange(len(models))

    fig, axes = plt.subplots(
        1,
        len(tests),
        figsize=(FIG_WIDTH_IN, 3.8),
        layout="constrained",
        squeeze=False,
    )
    axes = axes[0]
    for ax, (panel_title, _), values in zip(axes, tests, matrices):
        image = ax.imshow(values, cmap=_TEST_CMAP, vmin=0, vmax=0.1)
        ax.set_xticks(positions, models, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(positions, models, fontsize=8)
        ax.plot(
            positions,
            positions,
            linestyle="none",
            marker="x",
            ms=4,
            mew=1,
            color="white",
        )
        _panel_title(ax, panel_title)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="p-value")
    if title:
        fig.suptitle(title)
    plt.close(fig)
    return fig


def plot_lines(
    frame,
    xlabel=None,
    xticks=None,
    rotation=0,
    steps=None,
    title=None,
    ylabel="MAE, EUR/MWh",
):
    names = _names(frame, steps)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.4), layout="constrained")
    for name in names:
        ax.plot(
            frame.index,
            frame[name],
            marker="o",
            ms=3.2,
            lw=1.4,
            label=name,
            color=_series_color(name),
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    if rotation:
        ax.tick_params(axis="x", rotation=rotation)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(names) > 1:
        fig.legend(loc="outside right upper", frameon=False)
    _plain(ax)
    plt.close(fig)
    return fig


def plot_profile(
    frame,
    xlabel=None,
    xticks=None,
    rotation=0,
    title=None,
    ylabel="MAE, EUR/MWh",
):
    return plot_lines(
        frame,
        xlabel=xlabel,
        xticks=xticks,
        rotation=rotation,
        title=title,
        ylabel=ylabel,
    )


def plot_cumulative(
    frame,
    steps=None,
    title=None,
    ylabel="Cumulative Absolute Error, EUR/MWh",
):
    names = _names(frame, steps)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.6), layout="constrained")
    for name in names:
        ax.plot(
            frame.index,
            frame[name],
            lw=1.5,
            label=name,
            color=_series_color(name),
        )
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(names) > 1:
        fig.legend(loc="outside right upper", frameon=False)
    _plain(ax)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    plt.close(fig)
    return fig


def plot_weekday(by_dow, steps=None, title=None, ylabel="MAE, EUR/MWh"):
    names = _names(by_dow, steps)
    n = len(names)
    width = 0.72 / n
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.4), layout="constrained")
    x = np.arange(len(by_dow))
    for i, name in enumerate(names):
        ax.bar(
            x + (i - (n - 1) / 2) * width,
            by_dow[name],
            width,
            label=name,
            color=_series_color(name),
        )
    ax.set_xticks(x, by_dow.index)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if n > 1:
        fig.legend(loc="outside lower center", ncol=min(n, 6), frameon=False)
    _plain(ax)
    plt.close(fig)
    return fig


def plot_top5_boxplots(scores, steps, seeds):
    fig, axes = plt.subplots(
        1,
        len(steps),
        figsize=(FIG_WIDTH_IN, 3.2),
        sharey=True,
        layout="constrained",
    )
    for ax, step in zip(axes, steps):
        sub = scores.loc[scores["step"] == step].sort_values("rank")
        data = [
            sub.loc[sub["trial"] == trial, seeds].to_numpy(dtype=float).ravel()
            for trial in sub["trial"]
        ]
        color = _series_color(step)
        positions = list(range(1, len(data) + 1))
        box = ax.boxplot(
            data,
            positions=positions,
            showmeans=True,
            meanline=True,
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "#3f3f3f"},
            meanprops={"color": "#3f3f3f", "linestyle": "--"},
        )
        ax.set_xticks(positions, [str(trial) for trial in sub["trial"]])
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for key in ("whiskers", "caps"):
            for line in box[key]:
                line.set_color(color)
        _panel_title(ax, step)
        ax.set_xlabel("trial")
        _plain(ax)
    axes[0].set_ylabel("MAE, EUR/MWh")
    plt.close(fig)
    return fig


def plot_heatmap(mats, ylabel, xlabel, steps=None, title=None, value_label="MAE, EUR/MWh"):
    n = len(mats)
    if steps is None:
        steps = STEPS if n == len(STEPS) else [str(i + 1) for i in range(n)]
    panel_titles = list(steps)
    vmin = min(mat.min().min() for mat in mats)
    vmax = max(mat.max().max() for mat in mats)
    wide = mats[0].shape[1] > 8
    fig, axes = plt.subplots(
        n if wide else 1,
        1 if wide else n,
        figsize=(FIG_WIDTH_IN, 2.6 * n if wide else 3.8),
        sharex=wide,
        sharey=not wide,
        layout="constrained",
        squeeze=False,
    )
    axes = axes[:, 0] if wide else axes[0]
    for i, (ax, panel_title, mat) in enumerate(zip(axes, panel_titles, mats)):
        image = ax.imshow(
            mat.to_numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )
        is_last = i == n - 1
        ax.set_xticks(
            range(mat.shape[1]),
            mat.columns,
            fontsize=8,
            rotation=45 if wide else 0,
            ha="right" if wide else "center",
        )
        if wide:
            ax.tick_params(axis="x", labelbottom=is_last)
            ax.set_xlabel(xlabel if is_last else "")
            ax.set_ylabel(ylabel)
        else:
            ax.set_xlabel(xlabel)
        ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=8)
        _panel_title(ax, panel_title)
    if not wide:
        axes[0].set_ylabel(ylabel)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label=value_label)
    if title:
        fig.suptitle(title)
    plt.close(fig)
    return fig
