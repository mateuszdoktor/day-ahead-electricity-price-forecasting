from pathlib import Path

import pandas as pd

MAX_STEP = pd.Timedelta(hours=1)


def scan_zones(raw_dir: Path, ranges: list[str]) -> pd.DataFrame:
    rows = []

    for zone_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        found = [r for r in ranges if (zone_dir / r).is_dir()]
        complete = len(found) == len(ranges)

        datasets, not_in_all = [], []
        if complete:
            names = [{f.stem for f in (zone_dir / r).glob("*.parquet")} for r in ranges]
            common = set.intersection(*names)
            datasets = sorted(common)
            not_in_all = sorted(set.union(*names) - common)

        rows.append(
            {
                "zone": zone_dir.name,
                "ranges_found": len(found),
                "complete": complete,
                "datasets": datasets,
                "not_in_all_ranges": not_in_all,
            }
        )

    return pd.DataFrame(rows).set_index("zone")


def _normalise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if list(df.columns) == [0]:
        return df.set_axis([name], axis=1)
    return df


def _mismatches(a: pd.DataFrame, b: pd.DataFrame) -> int:
    a = a[~a.index.duplicated()]
    b = b[~b.index.duplicated()]
    common = a.index.intersection(b.index)
    x, y = a.loc[common], b.loc[common]
    return int((~((x == y) | (x.isna() & y.isna()))).to_numpy().sum())


def find_gaps(index: pd.DatetimeIndex, max_step: pd.Timedelta = MAX_STEP) -> pd.DataFrame:
    steps = index.to_series().diff()
    after = steps[steps > max_step]
    return pd.DataFrame(
        {
            "start": after.index - after.to_numpy(),
            "end": after.index,
            "length": after.to_numpy(),
        }
    ).reset_index(drop=True)


def _seam_gaps(gaps: pd.DataFrame, spans: list[tuple]) -> int:
    inside = [
        any(low <= gap.start and gap.end <= high for low, high in spans)
        for gap in gaps.itertuples()
    ]
    return len(gaps) - sum(inside)


def merge_frames(frames: list[pd.DataFrame], name: str) -> tuple[pd.DataFrame, dict]:
    frames = [_normalise(f, name) for f in frames]

    if len({str(f.index.tz) for f in frames}) != 1:
        raise ValueError(f"{name}: ranges have different time zones")

    columns = [c for c in frames[0].columns if all(c in f.columns for f in frames)]
    dropped = sorted({c for f in frames for c in f.columns} - set(columns))
    spans = [(f.index.min(), f.index.max()) for f in frames]
    frames = [f[columns] for f in frames]

    mismatches = sum(_mismatches(frames[0], f) for f in frames[1:])

    stacked = pd.concat(frames)
    merged = stacked[~stacked.index.duplicated(keep="first")].sort_index()

    gaps = find_gaps(merged.index)
    stats = {
        "rows": len(merged),
        "start": merged.index.min(),
        "end": merged.index.max(),
        "duplicates_removed": len(stacked) - len(merged),
        "overlap_mismatches": mismatches,
        "dropped_columns": dropped,
        "unique": bool(merged.index.is_unique),
        "monotonic": bool(merged.index.is_monotonic_increasing),
        "gaps": len(gaps),
        "longest_gap": gaps["length"].max() if len(gaps) else pd.Timedelta(0),
        "seam_gaps": _seam_gaps(gaps, spans),
    }
    return merged, stats


def merge_entsoe_ranges(
    raw_dir: Path, zones: pd.DataFrame, ranges: list[str]
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    merged, rows = {}, []

    for zone, info in zones[zones["complete"]].iterrows():
        merged[zone] = {}
        for name in info["datasets"]:
            frames = [
                pd.read_parquet(raw_dir / zone / r / f"{name}.parquet") for r in ranges
            ]
            merged[zone][name], stats = merge_frames(frames, name)
            rows.append({"zone": zone, "dataset": name, **stats})

    return merged, pd.DataFrame(rows)
