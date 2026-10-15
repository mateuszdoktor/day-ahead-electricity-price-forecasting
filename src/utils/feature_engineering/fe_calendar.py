import datetime as dt

import holidays
import pandas as pd

CALENDAR_COLUMNS = ["hour", "dow", "month", "is_holiday", "is_bridge"]


def _holiday_dates(index: pd.DatetimeIndex) -> set[dt.date]:
    years = range(index.min().year - 1, index.max().year + 2)
    return set(holidays.country_holidays("PL", years=years))


def _is_day_off(day: dt.date, days_off: set[dt.date]) -> bool:
    return day.weekday() >= 5 or day in days_off


def _is_bridge(day: dt.date, days_off: set[dt.date]) -> bool:
    if _is_day_off(day, days_off):
        return False
    previous_day = day - dt.timedelta(days=1)
    next_day = day + dt.timedelta(days=1)
    return _is_day_off(previous_day, days_off) and _is_day_off(next_day, days_off)


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    days_off = _holiday_dates(index)
    holiday = {}
    bridge = {}
    for day in index.normalize().unique():
        date = day.date()
        holiday[date] = int(date in days_off)
        bridge[date] = int(_is_bridge(date, days_off))

    dates = index.date
    return pd.DataFrame(
        {
            "hour": index.hour,
            "dow": index.dayofweek,
            "month": index.month,
            "is_holiday": [holiday[date] for date in dates],
            "is_bridge": [bridge[date] for date in dates],
        },
        index=index,
    ).astype(int)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, calendar_frame(df.index)], axis=1)


def add_time_idx(df: pd.DataFrame, start: int = 0) -> pd.DataFrame:
    df = df.copy()
    df["time_idx"] = range(start, start + len(df))
    return df
