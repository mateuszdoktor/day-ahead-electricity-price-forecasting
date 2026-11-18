import pandas as pd


DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


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
    tz = getattr(index, "tz", None)

    if tz is None:
        dst_duplicates = 0
        timezone_str = "naive"
    else:
        dst_duplicates = index.tz_localize(None).duplicated().sum() - duplicate_count
        timezone_str = str(tz)

    print(f"Duplicates: {duplicate_count}")
    print(f"Duplicates due to Summer/Winter time switch: {dst_duplicates}")
    print(f"Timezone:   {timezone_str}")

    time_steps = index.to_series().diff().dropna().value_counts()
    if time_steps.empty:
        return

    print("\nTime steps:")
    for step, count in time_steps.items():
        print(f"  {step}: {count}")


def print_data_overview(data, name):
    print(f"\n--- [{name}] Overview ---")

    rows = data.shape[0]
    columns = data.shape[1] if data.ndim == 2 else 1
    print(f"Shape: {rows} x {columns}")

    print("\nInfo:")
    data.info()

    print("\nDescription:")
    print(data.describe())


def _group_consecutive_steps(missing, expected):
    position = {timestamp: i for i, timestamp in enumerate(expected)}
    ranges = []
    start = previous = missing[0]

    for current in missing[1:]:
        if position[current] != position[previous] + 1:
            ranges.append((start, previous))
            start = current
        previous = current

    ranges.append((start, previous))
    return ranges, position


def find_missing_dates(data, name, freq="h"):
    expected = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq)
    missing = sorted(set(expected) - set(data.index))

    print(f"\n--- [{name}] Missing timestamps (freq={freq}) ---")
    print(f"Count: {len(missing)}")

    if not missing:
        return set()

    ranges, position = _group_consecutive_steps(missing, expected)

    print("\nMissing ranges:")
    for start, end in ranges:
        length = position[end] - position[start] + 1
        span = f"{start}" if start == end else f"{start} - {end}"
        print(f"  {span} ({length} timestamp{'s' if length > 1 else ''})")

    missing_by_day = pd.Series(missing).dt.dayofweek.value_counts().sort_index()

    print("\nMissing timestamps by day of week:")
    for day_number, count in missing_by_day.items():
        print(f"  {DAY_NAMES[day_number]}: {count}")

    return set(missing)
