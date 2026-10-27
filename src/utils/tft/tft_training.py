# Usage: python -m src.utils.tft.tft_training --step d1
# Trains TFT with the best hyperparameters from results/tft/{step}/best.json
# Saves the model to results/tft/{step}/model.ckpt

import argparse
import json

import lightning.pytorch as pl
import torch

from src.config import BASE_DIR
from src.utils.tft.tft_tuning import (
    NUM_WORKERS,
    SEED,
    load_data,
    make_model,
    make_training_dataset,
)

TRAIN_DAYS = 3 * 364


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["d1", "d2", "d3"])
    args = ap.parse_args()

    out_dir = BASE_DIR / "results" / "tft" / args.step
    best = json.loads((out_dir / "best.json").read_text())
    params = dict(best["best_params"])
    enc_len = params.pop("max_encoder_length")
    batch_size = params.pop("batch_size")
    clip = params.pop("gradient_clip_val")

    pl.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("high")

    df, known, unknown, _ = load_data(args.step)
    df = df.tail(TRAIN_DAYS * 24)

    training = make_training_dataset(df, known, unknown, enc_len)
    train_dl = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=NUM_WORKERS
    )
    model = make_model(training, **params)

    trainer = pl.Trainer(
        max_epochs=best["epochs_to_best"],
        accelerator="auto",
        devices=1,
        gradient_clip_val=clip,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dl)
    trainer.save_checkpoint(out_dir / "model.ckpt")


if __name__ == "__main__":
    main()
