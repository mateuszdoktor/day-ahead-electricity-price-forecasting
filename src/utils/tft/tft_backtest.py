# Usage: python -m src.utils.tft.tft_backtest --step d1
# Backtest of the model from results/tft/{step}/model.ckpt on the test split.
import argparse

import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.config import BASE_DIR
from src.utils.file_io import load_final_dataset
from src.utils.tft.tft_tuning import NUM_WORKERS, prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["d1", "d2", "d3"])
    args = ap.parse_args()

    model = TemporalFusionTransformer.load_from_checkpoint(
        BASE_DIR / "results" / "tft" / args.step / "model.ckpt"
    )

    df = prepare(load_final_dataset("tft", args.step, "full"))
    test = load_final_dataset("tft", args.step, "test")
    test_start = test["time_idx"].min()

    dataset = TimeSeriesDataSet.from_parameters(
        model.dataset_parameters, df, min_prediction_idx=test_start
    )
    dataloader = dataset.to_dataloader(
        train=False,
        batch_size=256,
        num_workers=NUM_WORKERS,
        sampler=range(0, len(dataset), 24),
    )
    pred = model.predict(dataloader, mode="prediction")

    forecast = pd.DataFrame({"Pred": pred.cpu().numpy().reshape(-1)}, index=test.index)
    path = (
        BASE_DIR / "results" / "tft" / "test" / "full" / f"TFT_{args.step.upper()}.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(path)


if __name__ == "__main__":
    main()
