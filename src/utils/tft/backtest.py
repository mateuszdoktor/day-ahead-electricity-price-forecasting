# Usage: python -m src.utils.tft.backtest --step d1 --trial 114
#        python -m src.utils.tft.backtest --step base --trial 114
# Per seed: results/tft/d1/test/trial_114/{1,2,3,4,5}.pt

import argparse

import torch

from src.config import BASE_DIR
from src.utils.tft.tuning import STEPS, load_data, load_frame, prepare
from src.utils.tft.validate import forecast_seeds, load_top_trials, trial_by_number

TRAIN_DAYS = 3 * 364


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=STEPS)
    ap.add_argument("--trial", required=True, type=int)
    args = ap.parse_args()

    trial = trial_by_number(load_top_trials(args.step), args.trial)

    torch.set_float32_matmul_precision("high")
    trainval, known, unknown, _ = load_data(args.step)
    train_df = trainval.tail(TRAIN_DAYS * 24)
    full = prepare(load_frame(args.step, "full"))
    test = load_frame(args.step, "test")

    path = (
        BASE_DIR
        / "results"
        / "tft"
        / args.step
        / "test"
        / f"trial_{trial['number']}.csv"
    )
    forecast_seeds(
        train_df,
        full,
        known,
        unknown,
        trial,
        test["time_idx"].min(),
        test.index,
        path,
        save_dir=path.parent / path.stem,
    )


if __name__ == "__main__":
    main()
