#!/usr/bin/env python3
"""Tools related to vectorization."""
import math
import sys
from collections.abc import Iterator

import numpy as np
import psutil

from python_tools import generic


def sliding_partition(
    array: np.ndarray,
    *,
    memory_usage: float = -1.0,
    rate: int = 1,
    shift: float = 1.0,
    window_size: float = 10.0,
    partition_size: int = -1,
) -> Iterator[tuple[int | np.ndarray, np.ndarray]]:
    """Return a generator to yield partitions of the slided signal.

    Note:
        Partitions are created such that approximately 20% of the
        total memory remains free (smallest partition size is two).

    Args:
        array: The 1D signal (numpy-like array).
        memory_usage: Estimated memory consumption for each window_size (in bytes).
        rate: Number of samples per second.
        shift: Shift of the sliding window in seconds.
        window_size: Length of the sliding window in seconds.
        partition_size: Directly specify the size of the partitions.

    Yields:
        First yield: Tuple of partition_size and time vector for each entry.
        Other yields: Tuple of partition and indices for the partition.
    """
    # determine number of entries
    shift_int = round(shift * rate)
    window_size_int = round(window_size * rate)
    instances_int = int((array.size - window_size_int) / shift_int + 1)

    # calculate time vector
    start_time = window_size_int * 0.5 / rate
    step_time = shift_int / rate
    end_time = start_time + instances_int * step_time - step_time / 2.0
    time = np.arange(start_time, end_time, step_time)
    assert time.size == instances_int

    # lower bound for memory usage
    if memory_usage <= -1:
        memory_usage = float(window_size_int * sys.getsizeof(array[0]))

    if partition_size < 1:
        # determine number of samples per partition
        available_memory = (
            generic.get_total_memory() - psutil.virtual_memory().used
        ) * 0.8

        runs = math.ceil(instances_int * memory_usage / available_memory)
        partition_size = max(2, math.ceil(instances_int / runs))
    else:
        runs = math.ceil(instances_int / partition_size)

    # yield partition size
    yield partition_size, time

    # generate partitions
    itime = 0
    view = np.lib.stride_tricks.sliding_window_view(array, window_size_int, axis=0)[
        ::shift_int
    ].transpose()
    for irun in range(runs):
        size_per_run = partition_size
        if irun == runs - 1:  # last run
            size_per_run = instances_int - partition_size * (runs - 1)

        data = view[:, itime : itime + size_per_run]

        yield data, np.arange(itime, itime + size_per_run)
        itime += size_per_run
    assert itime == view.shape[1]
