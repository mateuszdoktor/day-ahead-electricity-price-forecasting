# Usage: python -m src.utils.tft.tft_backtest --step d1

import argparse

import pandas as pd
import torch
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
    name = f"TFT_{args.step.upper()}"

    # Point forecast
    pred = model.predict(dataloader, mode="prediction")
    forecast = pd.DataFrame({"Pred": pred.cpu().numpy().reshape(-1)}, index=test.index)
    path = BASE_DIR / "results" / "tft" / "test" / "full" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(path)

    # Raw output
    result = model.predict(dataloader, mode="raw", return_x=True)
    interpretation = model.interpret_output(result.output, reduction="mean")
    out_dir = BASE_DIR / "results" / "tft" / "test" / "interpretation"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(interpretation, out_dir / f"{name}_interpretation.pt")
    torch.save(result.output, out_dir / f"{name}_raw_output.pt")
    torch.save(result.x, out_dir / f"{name}_x.pt")


if __name__ == "__main__":
    main()
