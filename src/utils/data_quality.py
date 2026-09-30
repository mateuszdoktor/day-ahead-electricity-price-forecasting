import pandas as pd

EXCLUDED_DATASETS = {
    "generation_units_unavailability",
    "production_units_unavailability",
}
EXCLUDED_DATE_GAPS = EXCLUDED_DATASETS | {"water_reservoirs_hydro_storage"}

DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
BUSINESS_TYPES = ["Planned maintenance", "Unplanned outage"]


def _to_utc(data, column):
    return pd.to_datetime(data[column], utc=True)


def _group_consecutive_dates(dates):
    ranges = []
    start = previous = dates[0]

    for current in dates[1:]:
        if current != previous + pd.Timedelta(days=1):
            ranges.append((start, previous))
            start = current
        previous = current

    ranges.append((start, previous))
    return ranges


def _print_businesstype_counts(title, data, mask):
    print(f"\n{title}")
    print(data.loc[mask, "businesstype"].value_counts())


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
    if name in EXCLUDED_DATE_GAPS:
        return set()

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


def print_units_unavailability_analysis(data):
    data = data.copy()

    print(f"Initial number of rows: {len(data)}")

    duplicated_rows = data.duplicated(keep="first")
    print(f"Duplicated rows: {duplicated_rows.sum()}")
    data = data.drop_duplicates(keep="first")

    duplicated_mrid_revision = data.duplicated(
        subset=["mrid", "revision"], keep="first"
    )
    print(f"Duplicated (mrid, revision) rows: {duplicated_mrid_revision.sum()}")
    data = data[~duplicated_mrid_revision].reset_index()

    created_over_3_years_before_start = (
        _to_utc(data, "start") - _to_utc(data, "created_doc_time")
    ).dt.days > 3 * 365
    _print_businesstype_counts(
        "Rows with created_doc_time more than 3 years before start:",
        data,
        created_over_3_years_before_start,
    )
    data = data[~created_over_3_years_before_start]

    created_on_or_after_end = _to_utc(data, "created_doc_time") >= _to_utc(data, "end")
    _print_businesstype_counts(
        "Rows with created_doc_time on or after end:",
        data,
        created_on_or_after_end,
    )
    data = data[~created_on_or_after_end]

    print(f"\nFinal number of rows: {len(data)}")

    created_doc_times = _to_utc(data, "created_doc_time")

    for business_type in BUSINESS_TYPES:
        doc_times = created_doc_times[data["businesstype"] == business_type]

        print(f"\n{business_type}:")
        print(f"Number of rows: {len(doc_times)}")

        if doc_times.empty:
            print("created_doc_time range: no data")
        else:
            print(f"created_doc_time range: {doc_times.min()} - {doc_times.max()}")

    return data
