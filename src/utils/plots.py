import matplotlib.pyplot as plt
import numpy as np

STEPS = ("d1", "d2", "d3")
COLORS = {"d1": "C0", "d2": "C1", "d3": "C2"}


def _plain(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_lines(frame, xlabel=None, xticks=None, rotation=0, steps=STEPS):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for step in steps:
        ax.plot(
            frame.index,
            frame[step],
            marker="o",
            ms=3.5,
            label=step,
            color=COLORS[step],
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


def plot_weekday(by_dow, steps=STEPS):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(by_dow))
    width = 0.24
    for i, step in enumerate(steps):
        ax.bar(x + (i - 1) * width, by_dow[step], width, label=step, color=COLORS[step])
    ax.set_xticks(x, by_dow.index)
    ax.set_ylabel("MAE, EUR/MWh")
    ax.legend(frameon=False)
    _plain(ax)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_heatmap(mats, ylabel, xlabel, steps=STEPS):
    vmin = min(mat.min().min() for mat in mats)
    vmax = max(mat.max().max() for mat in mats)
    wide = mats[0].shape[1] > 8
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11, 3.8 if wide else 5.2),
        sharey=True,
        layout="constrained",
    )
    image = None
    for ax, step, mat in zip(axes, steps, mats):
        image = ax.imshow(mat.to_numpy(), aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(
            range(mat.shape[1]),
            mat.columns,
            fontsize=8,
            rotation=45 if wide else 0,
        )
        ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=8)
        ax.set_title(step)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="MAE, EUR/MWh")
    plt.close(fig)
    return fig
