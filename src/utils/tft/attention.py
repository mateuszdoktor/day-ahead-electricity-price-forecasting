import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.config import BASE_DIR

CHOSEN = {"d1": 114, "d2": 75, "d3": 99}
SEEDS = (1, 2, 3, 4, 5)


def encoder_attention(step):
    frames = []
    for seed in SEEDS:
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
        weights = saved["raw"]["encoder_attention"].float()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * weights.shape[-1]
        frames.append(weights.mean(dim=(0, 2)))
    values = torch.stack(frames).mean(dim=0).numpy()
    length = values.shape[1]
    return pd.DataFrame(values, index=range(24), columns=range(-length, 0))


def plot_attention_heatmap(attention):
    length = attention.shape[1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    image = ax.imshow(attention.to_numpy(), aspect="auto", origin="upper")
    for edge in range(24, length, 24):
        ax.axvline(edge - 0.5, color="white", lw=0.8, ls="--")
    ax.set_yticks(range(0, 24, 2), [f"{hour:02d}:00" for hour in range(0, 24, 2)])
    ticks = list(range(0, length, 12))
    ax.set_xticks(ticks, [attention.columns[tick] for tick in ticks])
    ax.set_ylabel("Forecast hour")
    ax.set_xlabel("Encoder position")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="A · E")
    fig.tight_layout()
    plt.close(fig)
    return fig
