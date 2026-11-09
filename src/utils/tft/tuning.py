# Usage: python -m src.utils.tft.tuning --step d1 --trials 50

import argparse
import json
import math

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss

from src.config import BASE_DIR
from src.utils.file_io import load_final_dataset

TARGET = "PL_price"
GROUP = "series"

PRED_LEN = 24
ENCODER_LENGTH = 168
MAX_EPOCHS = 40
ES_PATIENCE = 10
WINDOW = 3
NUM_WORKERS = 2
SEED = 42

KNOWN_CATEGORICALS = ["hour", "dow", "month", "is_holiday", "is_bridge"]
KNOWN_REALS = [
    # D1
    "PL_price_lag24",
    "PL_price_lag48",
    "PL_price_lag72",
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
        add_relative_time_idx=False,
    )


class MAEWindow(pl.Callback):
    # Keeps val_MAE per epoch and reports its rolling mean to Optuna, so the
    # pruner judges the same quantity the objective minimizes.

    def __init__(self, trial):
        self.trial = trial
        self.values = []

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        self.values.append(trainer.callback_metrics["val_MAE"].item())
        if len(self.values) < WINDOW:
            return

        mean = sum(self.values[-WINDOW:]) / WINDOW
        if not math.isfinite(mean):
            return

        self.trial.report(mean, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"pruned after epoch {trainer.current_epoch + 1}")


def best_window_mae(values):
    # epochs_to_best is the 1-based middle epoch, so max_epochs stops on it.
    if len(values) < WINDOW:
        raise optuna.TrialPruned(f"fewer than {WINDOW} validation epochs")

    means = []
    for start in range(len(values) - WINDOW + 1):
        window = values[start : start + WINDOW]
        if all(math.isfinite(value) for value in window):
            means.append(sum(window) / WINDOW)
        else:
            means.append(math.inf)

    start = min(range(len(means)), key=means.__getitem__)
    score = means[start]
    if not math.isfinite(score):
        raise optuna.TrialPruned("non-finite val_loss")

    return score, start + WINDOW // 2 + 1


def make_model(training, **params):
    return TemporalFusionTransformer.from_dataset(
        training,
        loss=QuantileLoss(quantiles=[0.5]),
        reduce_on_plateau_patience=MAX_EPOCHS,
        optimizer="adamw",
        **params,
    )


def objective(trial, df, known, unknown, val_start):
    pl.seed_everything(SEED, workers=True)

    training = make_training_dataset(
        df[df.time_idx < val_start], known, unknown, ENCODER_LENGTH
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, min_prediction_idx=val_start, stop_randomization=True
    )

    clip = trial.suggest_float("gradient_clip_val", 0.01, 1.0, log=True)

    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])

    hidden_continous_size = hidden_size // 2

    model = make_model(
        training,
        logging_metrics=[MAE()],
        weight_decay=weight_decay,
        hidden_size=hidden_size,
        hidden_continuous_size=hidden_continous_size,
        attention_head_size=trial.suggest_categorical("attention_head_size", [1, 2, 4]),
        lstm_layers=trial.suggest_categorical("lstm_layers", [1, 2, 4]),
        dropout=trial.suggest_float("dropout", 0.05, 0.25),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
    )

    train_dl = training.to_dataloader(
        train=True, batch_size=128, num_workers=NUM_WORKERS
    )

    val_dl = validation.to_dataloader(
        train=False,
        batch_size=128,
        num_workers=NUM_WORKERS,
        sampler=range(0, len(validation), 24),
    )

    early_stop = EarlyStopping(monitor="val_MAE", min_delta=0.1, patience=ES_PATIENCE)
    window = MAEWindow(trial)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        gradient_clip_val=clip,
        callbacks=[early_stop, window],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_dl, val_dl)

    score, epochs_to_best = best_window_mae(window.values)
    trial.set_user_attr("epochs_to_best", epochs_to_best)

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
        sampler=optuna.samplers.TPESampler(n_startup_trials=60, multivariate=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=30, n_warmup_steps=10, n_min_trials=5
        ),
    )

    study.set_metric_names(["val_MAE"])

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
        "encoder_length": ENCODER_LENGTH,
        "known_categoricals": KNOWN_CATEGORICALS,
        "known_reals": known,
        "unknown_reals": unknown,
    }

    (out_dir / "best.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
