#!/usr/bin/env python3
"""Tools related to exporting time series."""
import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pympi


def create_elan_annotations(
    path: Path,
    *,
    starts: tuple[int, ...] = (),
    stops: tuple[int, ...] = (),
    tiers: tuple[str, ...] = (),
    texts: tuple[str, ...] = (),
    append: bool = False,
    empty_tiers: tuple[str, ...] = (),
) -> pympi.Eaf:
    """Create an ELAN file.

    Args:
        path: Path of the to be created ELAN file.
        starts: Starts in milli-seconds (integer) of the annotations.
        stops: Stops in milli-seconds (integer) of the annotations.
        tiers: The tiers of the annotations.
        texts: The values of the annotations (if none are provided the
            tier name will be used).
        empty_tiers: List of tiers to be added to the file (can be the
            list of all tiers).
        append: If the ELAN file already exists, new data is added to it.

    Returns:
        The eaf object.
    """
    if len(texts) == 1:
        texts = ("",) * len(tiers)

    # create empty eaf object
    file_path = None
    if path.is_file() and append:
        file_path = path
    eaf = pympi.Eaf(file_path=file_path)

    # add all tiers
    for tier in np.unique(np.concatenate((np.unique(tiers), np.unique(empty_tiers)))):
        if tier not in eaf.get_tier_names():
            eaf.add_tier(tier)

    # add actual annotations
    for args, text in zip(zip(tiers, starts, stops, strict=True), texts, strict=True):
        eaf.add_annotation(*args, value=text)

    # write to file
    pympi.Elan.to_eaf(path, eaf)

    return eaf


def elan_to_dataframe(
    path: Path,
    *,
    time: np.ndarray | None = None,
    shift: float = 0.1,
    values: bool = False,
) -> pd.DataFrame:
    """Convert an ELAN file to a discretized DataFrame.

    Args:
        path: The ELAN file.
        time: The discrete time vector (in seconds).
        shift: If time vector is None, a time vector will be created with the
            specified shift (in seconds).
        values: Whether to add a column containing the values of the annotations.

    Returns:
        The discrete Pandas DataFrame.
    """
    # load elan
    eaf = pympi.Eaf(file_path=path)
    tiers = tuple(eaf.get_tier_names())

    # build time vector
    if time is None:
        start, stop = eaf.get_full_time_interval()
        start /= 1000
        stop /= 1000
        time = np.arange(start, stop, shift)
    assert time is not None

    # populate DataFrame
    columns = {}
    for tier in tiers:
        discrete_data = np.full(time.size, False)
        values_data = np.full(time.size, "")
        for start, stop, value in eaf.get_annotation_data_for_tier(tier):
            start /= 1000
            stop /= 1000
            index = (time >= start) & (time <= stop)
            discrete_data[index] = True
            values_data[index] = value
        columns[tier] = discrete_data
        if values:
            columns[tier + "_values"] = values_data
    return pd.DataFrame(columns, index=time)


def elan_to_interval_dataframe(
    path: Path,
    *,
    remove_empty_tiers: bool = True,
) -> pd.DataFrame:
    """Convert an ELAN file to a start/end DataFrame.

    Args:
        path: The ELAN file.
        remove_empty_tiers: Whether to drop empty tiers.

    Returns:
        A Pandas DataFrame.
    """
    # load elan
    eaf = pympi.Eaf(file_path=path)
    tiers = list(eaf.get_tier_names())

    # populate DataFrame
    columns: dict[str, list[Any]] = {tier: [] for tier in (*tiers, "time", "end")}
    is_string = []
    intervals: dict[tuple[int, int], int] = {}
    for tier in tiers.copy():
        value = None
        for start, stop, value in eaf.get_annotation_data_for_tier(tier):
            start /= 1000
            stop /= 1000
            value = value or True
            if (start, stop) in intervals:
                columns[tier][intervals[start, stop]] = value
                continue
            intervals[start, stop] = len(columns["time"])
            columns["time"].append(start)
            columns["end"].append(stop)
            for tier_ in tiers:
                if tier_ == tier:
                    columns[tier_].append(value)
                else:
                    columns[tier_].append(float("NaN"))
        if remove_empty_tiers and value is None:
            del columns[tier]
            tiers.remove(tier)
        elif isinstance(value, str):
            is_string.append(tier)

    # replace NaN with '' for string
    for tier in is_string:
        columns[tier] = [x if isinstance(x, str) else "" for x in columns[tier]]

    return pd.DataFrame(
        columns,
        pd.IntervalIndex.from_arrays(
            columns.pop("time"),
            columns.pop("end"),
            closed="left",
        ),
    ).sort_index()


def create_subtitles(
    path: Path,
    starts: list[float],
    stops: list[float],
    texts: list[str],
) -> str:
    """Create subtitles.

    Args:
        path: Path to the to be created subtitle file.
        starts: List of starting points.
        stops: List of end points.
        texts: List of strings (the actual subtitle).

    Returns:
        String representing the subtitles
    """
    subtitle = []
    for i_line, (start, stop, text) in enumerate(
        zip(starts, stops, texts, strict=True),
    ):
        subtitle.append(
            f"{i_line+1}\n{_get_timestamps(start)} --> "
            f"{_get_timestamps(stop)}\n{text}",
        )
    result = "\n\n".join(subtitle)
    path.write_text(result)

    return result


def index_to_segments(
    index: np.ndarray,
    time: np.ndarray,
    *,
    exclude_end: bool = False,
) -> np.ndarray:
    """Convert binary index to intervals.

    Args:
        index: Binary array (0 and 1).
        time: Array containing the time.
        exclude_end: Whether to exclude the end time.

    Returns:
        Numpy array with shape t x 2.
    """
    index = index * 1  # make sure it is not binary
    index[np.isnan(index)] = 0
    # make sure first and last index are off
    timestep = np.median(np.diff(time))
    if index[0]:
        new_index = np.zeros(index.size + 1)
        new_index[1:] = index
        index = new_index
        new_time = np.zeros(time.size + 1)
        new_time[1:] = time
        new_time[0] = time[0] - timestep
        time = new_time
    if index[-1]:
        new_index = np.zeros(index.size + 1)
        new_index[:-1] = index
        index = new_index
        new_time = np.zeros(time.size + 1)
        new_time[:-1] = time
        new_time[-1] = time[-1] + timestep
        time = new_time

    diff = np.diff(index)
    starts = np.where(diff == 1)[0] + 1
    stops = np.where(diff == -1)[0] + 1
    if exclude_end:
        stops -= 1
    assert starts.size == stops.size
    assert np.all(np.less_equal(starts, stops))

    results = np.zeros((starts.size, 2))
    results[:, 0] = time[starts]
    results[:, 1] = time[stops]
    return results


def interval_overlap(interval: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Determine with which intervals and intervals overlaps.

    Args:
        interval: The interval.
        intervals: The intervals.

    Returns:
        An index into 'intervals' to indicate which intervals in it overlap
        with the given interval.
    """
    index = np.full(intervals.shape[0], False)
    for current1, current2, others1, others2 in [
        (0, 0, 0, 1),
        (1, 1, 0, 1),
        (1, 0, 1, 0),
    ]:
        index |= (interval[current1] >= intervals[:, others1]) & (
            interval[current2] <= intervals[:, others2]
        )
    # exclude border-only touch
    return index & (intervals[:, 1] != interval[0]) & (intervals[:, 0] != interval[1])


def _get_timestamps(seconds: float) -> str:
    return (
        datetime.datetime(year=1991, month=8, day=28, tzinfo=datetime.UTC)
        + datetime.timedelta(seconds=seconds)
    ).strftime("%H:%M:%S,%f")[:-3]
