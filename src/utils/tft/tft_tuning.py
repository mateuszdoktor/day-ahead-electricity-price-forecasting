# Usage: python -m src.utils.tft.tft_tuning --step d1 --trials 50

import argparse
import json
import math

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from optuna_integration import PyTorchLightningPruningCallback
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from src.config import BASE_DIR
from src.utils.file_io import load_final_dataset

TARGET = "PL_price"
GROUP = "series"

PRED_LEN = 24
MAX_EPOCHS = 30
ES_PATIENCE = 5
NUM_WORKERS = 4
SEED = 42

KNOWN_CATEGORICALS = ["hour", "dow", "month"]
KNOWN_REALS = [
    # D1
    "is_holiday",
    "is_bridge",
    "PL_price_lag24",
    "PL_price_lag48",
    "PL_price_lag168",
    "PL_load_fc",
    "PL_load_fc_delta24",
    # D2
    "np_PL",
    "gen_PL",
    "vload_PL",
    "ac_PL_import",
    "ac_PL_export",
    "flow_DE_PL",
    "flow_CZ_PL",
    "flow_SK_PL",
    "flow_UA_PL",
    "min_PL",
    "max_PL",
    # D3
    "DE_load_fc",
    "CZ_load_fc",
    "SK_load_fc",
    "SE4_load_fc",
    "LT_load_fc",
    "np_DE",
    "np_50Hz",
    "np_CZ",
    "np_SK",
    "flow_DE_CZ",
    "flow_CZ_SK",
]


def prepare(df):
    df = df.reset_index(drop=True)
    df[GROUP] = "series"
    df[KNOWN_CATEGORICALS] = df[KNOWN_CATEGORICALS].astype(str)
    return df


def load_data(step):
    train = load_final_dataset("tft", step, "train")
    val = load_final_dataset("tft", step, "val")
    df = prepare(pd.concat([train, val]))

    known = [c for c in KNOWN_REALS if c in df.columns]
    skip = {"time_idx", GROUP, *KNOWN_CATEGORICALS, *known}
    unknown = [c for c in df.columns if c not in skip]
    return df, known, unknown, val["time_idx"].min()


def make_training_dataset(df, known, unknown, enc_len):
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=[GROUP],
        min_encoder_length=enc_len,
        max_encoder_length=enc_len,
        min_prediction_length=PRED_LEN,
        max_prediction_length=PRED_LEN,
        static_categoricals=[GROUP],
        time_varying_known_categoricals=KNOWN_CATEGORICALS,
        time_varying_known_reals=known,
        time_varying_unknown_reals=unknown,
        target_normalizer=GroupNormalizer(groups=[GROUP]),
        add_relative_time_idx=True,
    )


def make_model(training, **params):
    return TemporalFusionTransformer.from_dataset(
        training, loss=QuantileLoss(), reduce_on_plateau_patience=3, **params
    )


def objective(trial, df, known, unknown, val_start):
    pl.seed_everything(SEED, workers=True)

    enc_len = trial.suggest_categorical("max_encoder_length", [24, 48, 72, 168])
    training = make_training_dataset(df[df.time_idx < val_start], known, unknown, enc_len)
    validation = TimeSeriesDataSet.from_dataset(
        training, df, min_prediction_idx=val_start, stop_randomization=True
    )
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    clip = trial.suggest_float("gradient_clip_val", 0.01, 10.0, log=True)
    model = make_model(
        training,
        hidden_size=trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
        hidden_continuous_size=trial.suggest_categorical(
            "hidden_continuous_size", [8, 16, 32, 64]
        ),
        attention_head_size=trial.suggest_categorical(
            "attention_head_size", [1, 2, 4, 6]
        ),
        lstm_layers=trial.suggest_int("lstm_layers", 1, 4),
        dropout=trial.suggest_float("dropout", 0.05, 0.4),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
    )
    train_dl = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=NUM_WORKERS
    )

    val_dl = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        sampler=range(0, len(validation), 24),
    )

    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=ES_PATIENCE)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        gradient_clip_val=clip,
        callbacks=[
            early_stop,
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dl, val_dl)

    score = early_stop.best_score.item()
    if not math.isfinite(score):
        raise optuna.TrialPruned("non-finite val_loss")
    trial.set_user_attr("epochs_to_best", trainer.current_epoch - early_stop.wait_count)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["d1", "d2", "d3"])
    ap.add_argument("--trials", type=int, default=50, help="trials to run now")
    args = ap.parse_args()

    out_dir = BASE_DIR / "results" / "tft" / args.step
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    df, known, unknown, val_start = load_data(args.step)

    study = optuna.create_study(
        study_name=f"tft_{args.step}",
        storage=f"sqlite:///{out_dir / 'study.db'}",
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=4),
    )
    study.set_metric_names(["val_loss"])
    study.optimize(
        lambda trial: objective(trial, df, known, unknown, val_start),
        n_trials=args.trials,
        catch=(torch.cuda.OutOfMemoryError,),
        gc_after_trial=True,
    )

    best = study.best_trial
    info = {
        "step": args.step,
        "best_value": best.value,
        "best_trial": best.number,
        "best_params": best.params,
        "epochs_to_best": best.user_attrs["epochs_to_best"],
        "n_trials": len(study.trials),
        "prediction_length": PRED_LEN,
        "known_categoricals": KNOWN_CATEGORICALS,
        "known_reals": known,
        "unknown_reals": unknown,
    }
    (out_dir / "best.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
