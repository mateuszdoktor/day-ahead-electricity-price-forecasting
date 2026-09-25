import pandas as pd


EXCLUDED_DATASETS = {
    "generation_units_unavailability",
    "production_units_unavailability",
}

EXCLUDED_DATE_GAPS = EXCLUDED_DATASETS | {
    "water_reservoirs_hydro_storage",
}


def print_missing_values(data, name):
    print(f"\n--- [{name}] Missing values ---")

    missing = data.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
        return

    print(missing.to_string())


def print_index_summary(data, name):
    print(f"\n--- [{name}] Index summary ---")

    index = data.index
    duplicate_count = index.duplicated().sum()
    timezone = getattr(index, "tz", None) or "naive"

    print(f"Duplicates: {duplicate_count}")
    print(f"Timezone:  {timezone}")

    if name in EXCLUDED_DATASETS:
        return

    time_steps = index.to_series().diff().dropna().value_counts()

    if time_steps.empty:
        return

    print("\nTime steps:")
    for step, count in time_steps.items():
        print(f"  {step}: {count}")


def print_data_overview(data, name):
    print(f"\n--- [{name}] Overview ---")

    rows, columns = data.shape if len(data.shape) == 2 else (data.shape[0], 1)
    print(f"Shape: {rows} x {columns}")

    print("\nInfo:")
    data.info()

    print("\nDescription:")
    print(data.describe())


def find_missing_dates(data, name):
    if name in EXCLUDED_DATE_GAPS:
        return set()

    expected_dates = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
    )

    missing_dates = sorted(set(expected_dates) - set(data.index))

    print(f"\n--- [{name}] Missing dates ---")
    print(f"Count: {len(missing_dates)}")

    if not missing_dates:
        return set()

    ranges = []
    start = previous = missing_dates[0]

    for current in missing_dates[1:]:
        if current == previous + pd.Timedelta(days=1):
            previous = current
            continue

        ranges.append((start, previous))
        start = previous = current

    ranges.append((start, previous))

    print("\nMissing ranges:")
    for start, end in ranges:
        if start == end:
            print(f"  {start}")
        else:
            print(f"  {start} - {end}")

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    missing_by_day = pd.Series(missing_dates).dt.dayofweek.value_counts().sort_index()

    print("\nMissing by day of week:")
    for day_number, count in missing_by_day.items():
        print(f"  {day_names[day_number]}: {count}")

    return set(missing_dates)
