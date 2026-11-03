# Usage: python -m src.utils.tft.tft_validate --step d1
#        python -m src.utils.tft.tft_validate --step d1 --trial 114

import argparse
import gc
import json

import lightning.pytorch as pl
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from src.config import BASE_DIR
from src.utils.file_io import load_final_dataset
from src.utils.tft.tft_tuning import (
    NUM_WORKERS,
    load_data,
    make_model,
    make_training_dataset,
)

SEEDS = (1, 2, 3, 4, 5)
BATCH_SIZE = 128


def load_top_trials(step):
    path = BASE_DIR / "results" / "tft" / step / "top5.json"
    return json.loads(path.read_text())["trials"]


def trial_by_number(trials, number):
    for trial in trials:
        if trial["number"] == number:
            return trial
    numbers = [trial["number"] for trial in trials]
    raise SystemExit(f"trial {number} is not in top5.json (have {numbers})")


def forecast_seeds(
    train_df, pred_df, known, unknown, trial, pred_start, index, out_path
):
    params = dict(trial["params"])
    enc_len = params.pop("max_encoder_length")
    clip = params.pop("gradient_clip_val")
    params["hidden_continuous_size"] = params["hidden_size"] // 2
    epochs = int(trial["epochs_to_best"])

    forecasts = {}
    for seed in SEEDS:
        print(f"trial {trial['number']} seed {seed}")
        pl.seed_everything(seed, workers=True)

        training = make_training_dataset(train_df.copy(), known, unknown, enc_len)
        model = make_model(training, **params)
        train_dl = training.to_dataloader(
            train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
        )
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            gradient_clip_val=clip,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(model, train_dl)

        dataset = TimeSeriesDataSet.from_dataset(
            training,
            pred_df.copy(),
            min_prediction_idx=pred_start,
            stop_randomization=True,
        )
        loader = dataset.to_dataloader(
            train=False,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            sampler=range(0, len(dataset), 24),
        )
        pred = model.predict(loader, mode="prediction").cpu().numpy().reshape(-1)
        if len(pred) != len(index):
            raise RuntimeError(
                f"forecast length {len(pred)} != index length {len(index)}"
            )
        forecasts[seed] = pred

        del model, trainer, train_dl, dataset, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(forecasts, index=index).to_csv(out_path)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["d1", "d2", "d3"])
    ap.add_argument("--trial", type=int)
    args = ap.parse_args()

    trials = load_top_trials(args.step)
    if args.trial is not None:
        trials = [trial_by_number(trials, args.trial)]

    torch.set_float32_matmul_precision("high")
    df, known, unknown, val_start = load_data(args.step)
    train_df = df[df.time_idx < val_start]
    val = load_final_dataset("tft", args.step, "val")

    for trial in trials:
        path = (
            BASE_DIR
            / "results"
            / "tft"
            / args.step
            / "val"
            / f"trial_{trial['number']}.csv"
        )
        if path.exists():
            print(f"skip {path}")
            continue
        forecast_seeds(
            train_df, df, known, unknown, trial, val_start, val.index, path
        )


if __name__ == "__main__":
    main()
