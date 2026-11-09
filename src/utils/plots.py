import matplotlib.pyplot as plt
import numpy as np

STEPS = ("d1", "d2", "d3")
COLORS = {"d1": "C0", "d2": "C1", "d3": "C2"}


def _plain(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _names(frame, steps):
    return list(frame.columns if steps is None else steps)


def _color(name, i):
    return COLORS.get(name, f"C{i % 10}")


def plot_lines(frame, xlabel=None, xticks=None, rotation=0, steps=None):
    names = _names(frame, steps)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, name in enumerate(names):
        ax.plot(
            frame.index,
            frame[name],
            marker="o",
            ms=3.5,
            label=name,
            color=_color(name, i),
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    if rotation:
        ax.tick_params(axis="x", rotation=rotation)
    ax.set_ylabel("MAE, EUR/MWh")
    ax.legend(frameon=False, loc="upper left")
    _plain(ax)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_weekday(by_dow, steps=None):
    names = _names(by_dow, steps)
    n = len(names)
    width = 0.72 / n
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(by_dow))
    for i, name in enumerate(names):
        ax.bar(
            x + (i - (n - 1) / 2) * width,
            by_dow[name],
            width,
            label=name,
            color=_color(name, i),
        )
    ax.set_xticks(x, by_dow.index)
    ax.set_ylabel("MAE, EUR/MWh")
    ax.legend(frameon=False)
    _plain(ax)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_heatmap(mats, ylabel, xlabel, steps=None):
    n = len(mats)
    if steps is None:
        steps = STEPS if n == len(STEPS) else [str(i + 1) for i in range(n)]
    titles = list(steps)
    vmin = min(mat.min().min() for mat in mats)
    vmax = max(mat.max().max() for mat in mats)
    wide = mats[0].shape[1] > 8
    height = 3.8 if wide else 5.2
    width = 11 if n == len(STEPS) else 3.8 * n + 0.6
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(width, height),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    axes = axes[0]
    image = None
    for ax, title, mat in zip(axes, titles, mats):
        image = ax.imshow(mat.to_numpy(), aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(
            range(mat.shape[1]),
            mat.columns,
            fontsize=8,
            rotation=45 if wide else 0,
        )
        ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="MAE, EUR/MWh")
    plt.close(fig)
    return fig
