#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues = false
"""Tools related to caching files."""
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

import cloudpickle
import pandas as pd
import torch

from python_tools import generic

HDF_EXTENSION: Final = "hdf"
PICKLE_EXTENSION: Final = "pickle"

_DataFrameDictT = TypeVar("_DataFrameDictT", pd.DataFrame, dict[str, pd.DataFrame])


def add_extension(path: Path, extension: str, *, has_to_exists: bool = False) -> Path:
    """Add an extension to a path (removes existing extensions).

    Args:
        path: The path.
        extension: The extension to add.
        has_to_exists: Falls back to the existing extension.

    Returns:
        Path with the extensions.
    """
    path = generic.fullpath(path)
    new_path = path.with_suffix(f".{extension}")
    if has_to_exists and not new_path.is_file() and path.is_file():
        return path
    return new_path


def write_pickle(path: Path, data: object) -> Path:
    """Use pickle to save data with the newest protocol.

    Note:
        PyTorch's torch.save is used together with cloudpickle.

    Args:
        path: Where the data is written to.
        data: The data.

    Returns:
        The path written to.
    """
    path = add_extension(path, PICKLE_EXTENSION)
    torch.save(
        data,
        path,
        pickle_module=cloudpickle,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )

    return path


def read_pickle(name: Path, **kwargs: Any) -> Any:
    """Read data from a pickle file.

    Note:
        All tensors are moved to the CPU.

    Args:
        name: Where the data is read from.
        **kwargs: Additional keywords forwarded to load().

    Returns:
        The read data. Is None if file does not exist.
    """
    path = add_extension(name, PICKLE_EXTENSION, has_to_exists=True)
    if not path.is_file() or path.stat().st_size == 0:
        return None

    try:
        data = torch.load(path, map_location="cpu", **kwargs)
    except (RuntimeError, ValueError):
        # pytorch can only load its own pickle files
        with path.open(mode="rb") as file:
            data = pickle.load(file, **kwargs)

    return data


def write_hdfs(
    path: Path,
    dataframes: dict[str, pd.DataFrame],
    hdf_format: Literal["table", "fixed"] = "fixed",
    **kwargs: Any,
) -> Path:
    """Write a dictionary of dataframes to one hdf file.

    Args:
        path: Whetere to save the data.
        dataframes: Dictionary of DataFrames.
        hdf_format: Whether to use "fixed" or "table".
        **kwargs: Forwarded to pd.HDFStore.

    Returns:
        The path where the data was saved.
    """
    path = add_extension(path, HDF_EXTENSION)
    kwargs.setdefault("mode", "w")
    with pd.HDFStore(path, **kwargs) as store:
        for key, value in dataframes.items():
            if isinstance(value.index, pd.IntervalIndex):
                value["end"] = value.index.right.array
                value.index = value.index.left
            store.put(key, value, format=hdf_format)

    return path


def read_hdfs(name: Path, **kwargs: Any) -> dict[str, pd.DataFrame]:
    """Read all dataframes from a HDF file.

    Args:
        name: Path to the HDF file.
        **kwargs: Forwardd to pd.HDFStore.select.

    Returns:
        The loaded DataFrames.
    """
    path = add_extension(name, HDF_EXTENSION, has_to_exists=True)
    if not path.is_file():
        return {}

    result = {}
    with pd.HDFStore(path, mode="r") as store:
        for key in store:
            data = store.select(key, **kwargs)
            assert isinstance(data, pd.DataFrame)
            if "end" in data.columns and data["end"].dtype == data.index.dtype:
                data.index = pd.IntervalIndex.from_arrays(
                    data.index,
                    data.pop("end"),
                    closed="left",
                )

            result[key.removeprefix("/")] = data

    return result


def cache_wrapper(
    function: Callable[..., _DataFrameDictT],
) -> Callable[..., _DataFrameDictT | None]:
    """Handle caching for a function.

    Args:
        function: The to be cached functions.

    Returns:
        The wrapped function.
    """

    def wrapper(
        *args: Any,
        cache: Path,
        check_cache_only: bool = False,
        **kwargs: Any,
    ) -> _DataFrameDictT | None:
        """Read/write cached outputs.

        Args:
            *args: Any positional arguments.
            cache: Where to cache the result.
            check_cache_only: Whether to just check if the cache exists.
            **kwargs: Any keyword arguments.

        Returns:
            The result of the wrapped function.
        """
        # load/check cache
        cache = generic.fullpath(cache)
        saved_result = read_hdfs(cache)
        if saved_result:
            if len(saved_result) == 1 and next(iter(saved_result)) == "df":
                return saved_result["df"]  # type: ignore[return-value]
            return saved_result  # type: ignore[return-value]
        if check_cache_only:
            return None

        # run/save
        result = function(*args, cache=cache, **kwargs)
        if isinstance(result, dict):
            write_hdfs(cache, result)
        else:
            write_hdfs(cache, {"df": result})
        return result

    return wrapper
