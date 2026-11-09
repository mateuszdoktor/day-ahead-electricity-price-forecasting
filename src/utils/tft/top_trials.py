# Usage: python -m src.utils.tft.top_trials --step d1

import argparse
import json
import math

import optuna

from src.config import BASE_DIR

TOP_N = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["d1", "d2", "d3"])
    args = ap.parse_args()

    out_dir = BASE_DIR / "results" / "tft" / args.step
    db = out_dir / "study.db"
    study = optuna.load_study(
        study_name=f"tft_{args.step}",
        storage=f"sqlite:///file:{db}?mode=ro&uri=true",
    )

    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(trial.value)
    ]
    complete.sort(key=lambda trial: (trial.value, trial.number))
    if len(complete) < TOP_N:
        raise SystemExit(f"{args.step}: only {len(complete)} complete trials")

    info = {
        "step": args.step,
        "trials": [
            {
                "rank": rank,
                "number": trial.number,
                "value": trial.value,
                "epochs_to_best": int(trial.user_attrs["epochs_to_best"]),
                "params": trial.params,
            }
            for rank, trial in enumerate(complete[:TOP_N], start=1)
        ],
    }

    path = out_dir / "top5.json"
    text = json.dumps(info, indent=2)
    path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
