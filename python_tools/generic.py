#!/usr/bin/env python3
"""Generic tools related to python."""
import argparse
import functools
import itertools
import math
import multiprocessing as mp
import os
import random
import resource
import shlex
import subprocess
import tempfile
import types
import uuid
import zipfile
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from hashlib import sha3_256
from pathlib import Path
from types import TracebackType
from typing import IO, Any, TypeVar, overload

import dask
import numpy as np
import psutil
import torch
from dask.delayed import Delayed
from dask.distributed import Client, LocalCluster

T = TypeVar("T")
T2 = TypeVar("T2")


def map_parallel(
    function: Callable[[T2], T],
    elements: Iterable[T2],
    *,
    workers: int = 16,
    chunksize: int = 1,
) -> list[T]:
    """Map function with a switch to enable parallel processing.

    Args:
        function: Function to be applied on each element.
        elements: The list of elements.
        workers: The number of workers. Run sequentially, if ``workers < 2``.
        chunksize: How many elements a worker should get each time.

    Returns:
        List of the processed elements.
    """
    # process elements
    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            return pool.map(function, elements, chunksize)
    return list(map(function, elements))


def run_distributed(
    delayed: list[Delayed],
    *,
    workers: int = 4,
    setup: Callable[..., None] | None = None,
    **kwargs: Any,
) -> Sequence[Any]:
    """Evaluate delayed objects in a sequential or parallel manner.

    Args:
        delayed: The delayed objects to be evaluated.
        workers: Number of workers to use. Run sequentially, if ``workers < 2``.
        setup: Optional function to set the workers up.
        **kwargs: Arguments for the Local/SLURMCluster.

    Returns:
        A list of results.
    """
    if workers < 2:
        # sequential, for debugging
        return dask.compute(*delayed, scheduler="synchronous")

    # cluster SLURM/local
    logging_directory = tempfile.TemporaryDirectory(prefix="dask_")
    kwargs["local_directory"] = logging_directory.name
    kwargs.setdefault("threads_per_worker", 1)
    kwargs.setdefault("processes", True)
    kwargs.setdefault("scheduler_port", 0)
    kwargs.setdefault("dashboard_address", None)
    kwargs.setdefault("n_workers", 0)
    kwargs.setdefault("memory_limit", None)

    with logging_directory, dask.config.set(
        {"distributed.worker.daemon": False},
    ), LocalCluster(**kwargs) as cluster:
        cluster.adapt(  # pyright: ignore[reportGeneralTypeIssues]
            minimum=1,
            maximum=workers,
        )
        with Client(cluster) as client:
            if setup is not None:
                client.register_worker_callbacks(setup)
            return dask.compute(*delayed, scheduler=client)


def run(
    command: str | list[str],
    *,
    stdout: int | IO[bytes] | None = subprocess.PIPE,
    stderr: int | IO[bytes] | None = subprocess.PIPE,
    stdin: int | None = subprocess.DEVNULL,
) -> tuple[int, str, str]:
    """Execute a command.

    Args:
        command: The command to be executed.
        stdout: How to capture the standard output.
        stderr: How to capture the standard errors.
        stdin: Whether to allow standard input.

    Note:
        ``subprocess.PIPE`` is replaced with ``tempfile.TemporaryFile`` to
        overcome the buffer limit of ``subprocess.PIPE``.

    Returns:
        The returncode as well as the standard output and errors.
    """
    if isinstance(command, str):
        command = shlex.split(command)

    # replace pipe with tempfile to overcome the buffer limit
    if stdout == subprocess.PIPE:
        stdout = tempfile.TemporaryFile(buffering=0)
    if stderr == subprocess.PIPE:
        stderr = tempfile.TemporaryFile(buffering=0)

    with subprocess.Popen(
        command,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
    ) as process:
        pass

    out = ""
    err = ""
    if stdout is not None and not isinstance(stdout, int):
        with stdout:
            stdout.seek(0)
            out = "".join(line.decode() for line in stdout.readlines())
    if stderr is not None and not isinstance(stderr, int):
        with stderr:
            stderr.seek(0)
            err = "".join(line.decode() for line in stderr.readlines())

    return process.returncode, out, err


def combinations(
    dictionary: dict[T, list[T2]],
    *,
    copy_elements: bool = True,
    not_copy_type: tuple[type, ...] = (
        np.ndarray,
        torch.Tensor,
        types.FunctionType,
        types.BuiltinFunctionType,
        functools.partial,
    ),
) -> list[dict[T, T2]]:
    """Create a dictionary of all combinations.

    Args:
        dictionary: The dictionary with all possible values for each field.
        copy_elements: Whether to makes a copy of the possible values.
        not_copy_type: Do not make copies of objects from the listed types.

    Note:
        Fields with no possible values are removed.

    Returns:
        All possibilities.
    """
    dictionary = {key: value for key, value in dictionary.items() if value}

    # find objects to ignore in deepcopy
    memo: dict[int, Any] = {}
    if copy_elements:
        memo = shallow_find_object_of_type(dictionary, not_copy_type)

    return [
        dict(
            zip(
                dictionary,
                deepcopy(value, memo=memo.copy()) if copy_elements else value,
                strict=True,
            ),
        )
        for value in itertools.product(*dictionary.values())
    ]


def shallow_find_object_of_type(
    parent: object,
    classes: tuple[type, ...],
    *,
    memo: dict[int, Any] | None = None,
) -> dict[int, Any]:
    """Return a deepcopy-compatible memo with objects of the specified types.

    Args:
        parent: The object being traversed.
        classes: Tuple of types.
        memo: Existing memo (for recursion).

    Returns:
        The filled memo.
    """
    memo = memo or {}

    # found object of matching type
    if isinstance(parent, classes):
        memo[id(parent)] = parent
        return memo

    # recurse for simple data structures (list, tuple, and dict)
    if isinstance(parent, list | tuple | dict):
        for item in parent:
            memo = shallow_find_object_of_type(item, classes, memo=memo)
    if isinstance(parent, dict):
        for item in parent.values():
            memo = shallow_find_object_of_type(item, classes, memo=memo)
    assert memo is not None
    return memo


def random_name() -> str:
    """Return a random string.

    Returns:
        The random string.
    """
    return f"{uuid.uuid4()}"


class ChangeDir:
    """Context manager for changing the current working directory.

    stackoverflow.com/questions/431684/how-do-i-cd-in-python
    """

    def __init__(self, directory: Path) -> None:
        """Create the context-manager.

        Args:
            directory: The path to which the current working directory will be changed.
        """
        self.new_dir = fullpath(directory)
        self.old_dir = Path()

    def __enter__(self) -> None:
        """Change into the new directory."""
        self.old_dir = Path.cwd()
        os.chdir(self.new_dir)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Back to the previous directory.

        Args:
            exc_type: Not used.
            exc: Not used.
            traceback: Not used.
        """
        os.chdir(self.old_dir)


def flatten_nested_list(
    nested_list: Sequence[Sequence[T]],
) -> list[T]:
    """Flatten a nested list.

    Args:
        nested_list: The nested list.

    Returns:
        The flattened list.
    """
    return list(itertools.chain.from_iterable(nested_list))


def simplify_strings(
    strings: list[str],
    *,
    prefix: bool = True,
    suffix: bool = True,
) -> list[str]:
    """Remove the longest pre- and suffix from a list of strings.

    Args:
        strings: The list of strings.
        prefix: Whether to remove the prefix.
        suffix: Whether to remove the suffix.

    Returns:
        List of strings.
    """
    prefix_str = ""
    suffix_str = ""
    if prefix:
        prefix_str = os.path.commonprefix(strings)
    if suffix:
        suffix_str = os.path.commonprefix([string[::-1] for string in strings])
    return [
        string.removeprefix(prefix_str).removesuffix(suffix_str) for string in strings
    ]


def identity(item: T, **kwargs: object) -> T:
    """Identity function (to avoid a few lambda function).

    Args:
        item: Anything.
        **kwargs: Are ignored.

    Returns:
        The item.
    """
    return item


def randomized_iterator(iterator: Iterable[T], *, size: int = 10) -> Iterable[T]:
    """Hold up to ``size`` items of the iterator and returns a random one.

    Args:
        iterator: The original iterator.
        size: From how many items to choose from.

    Note:
        This provides only a very basic 'randomness level'.

    Yields:
        The iterators items in random order.
    """
    size = max(size, 1)
    iterator = iter(iterator)
    buffer: list[T] = []
    rng = random.Random()
    rng.seed(1)
    try:
        while True:
            # get at new items
            buffer.extend([next(iterator) for _ in range(size - len(buffer))])
            # choose random entry to take
            yield buffer.pop(rng.randint(0, len(buffer) - 1))
    except StopIteration:
        rng.shuffle(buffer)
        yield from buffer
        return


@overload
def basename(path: None) -> None:
    ...


@overload
def basename(path: Path) -> str:
    ...


def basename(path: Path | None) -> str | None:
    """Get the basename until the first ".".

    Args:
        path: A path or None.

    Returns:
        The filename without extension.
    """
    if path is None:
        return None
    return path.name.split(".", 1)[0]


def fullpath(path: Path) -> Path:
    """Extend the path to its fullpath.

    Args:
        path: A path.

    Returns:
        The absolute expanded path.
    """
    return path.expanduser().resolve()


def compress_files(
    files: list[Path],
    output: Path,
    *,
    delete: bool = False,
    compression: int = zipfile.ZIP_DEFLATED,
) -> None:
    """Create a zip archive from a list of files.

    Args:
        files: List of files that are going to be compressed.
        output: Name of the zip archive that is going to be created.
        delete: Whether to delete the files after adding them to the archive.
        compression: Forwarded to as ``zipfile.ZipFile(...compression=compression)``.
    """
    if len(files) > 1:
        names = simplify_strings(list(map(str, files)), suffix=False)
    elif len(files) == 1:
        names = [files[0].name]
    else:
        return
    with zipfile.ZipFile(output, mode="w", compression=compression) as compressor:
        for file, name in zip(files, names, strict=True):
            compressor.write(file, name)
            if delete:
                file.unlink()


def get_total_memory() -> int:
    """Get the total amount of memory in bytes.

    Returns:
        The maximal amount of memory available to the current python
        process in bytes.
    """
    total = psutil.virtual_memory().total
    hard_limit = resource.getrlimit(resource.RLIMIT_RSS)[1]
    if hard_limit > 0:
        return min(total, hard_limit)
    return total


def find_indices(reference_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Find a mapping between two permutations.

    Finds indices such that ids[indices] == reference_ids. It assumes that
    both arrays are 1D and that one is a permutation of the other one.

    Args:
        reference_ids: The reference array.
        ids: The to be transformed array.

    Returns:
        Indices for ids to be the same as reference_ids.
    """
    reference_ids = np.asarray(reference_ids).reshape(-1)
    ids = np.asarray(ids).reshape(-1)

    index = np.argsort(ids)
    reference_index = np.argsort(reference_ids)
    inverse_reference_index = np.empty(index.size, index.dtype)
    inverse_reference_index[reference_index] = np.arange(index.size)
    return index[inverse_reference_index]


def round_up_to_odd(number: float) -> int:
    """Find the next highest odd number.

    stackoverflow.com/questions/31648729/round-a-float-up-to-next-odd-integer

    Args:
        number: A float.

    Returns:
        The next highest odd integer.
    """
    return math.ceil(number) // 2 * 2 + 1


def get_object(parent: object, attributes: list[str]) -> Any:
    """Traverse an object to get a nested attribute.

    Args:
        parent: Object to be traversed.
        attributes: Names of attributes.

    Returns:
        The nested object.
    """
    for attribute in attributes:
        parent = getattr(parent, attribute)
    return parent


def startswith_list(prefix: str, strings: Sequence[str]) -> bool:
    """Return whether any string in a list of strings starts with the given string.

    Args:
        prefix: The prefix to check for.
        strings: Tuple of strings.

    Returns:
        Whether any string in the list starts with the given string.
    """
    return any(item.startswith(prefix) for item in strings)


def hashname(data: str) -> str:
    """Hash a string.

    Args:
        data: String to be hashed.

    Returns:
        The SHA3 256 hash.
    """
    return sha3_256(data.encode("utf-8")).hexdigest()


def namespace_as_string(
    args: argparse.Namespace,
    *,
    exclude: tuple[str, ...] = (),
) -> str:
    """Convert an argparse namespace to a compact'ish string.

    Args:
        args: The commandline options.
        exclude: Which fields to ignore.

    Returns:
        String representing the namespace.
    """
    name = "_".join(
        f"{key}={value}"
        for key, value in sorted(vars(args).items())
        if key not in exclude and value
    )
    return name.replace(",", "").replace(" ", "").replace(".", "").strip("_")


def available_disk_space(path: Path) -> float:
    """Return the user-accessible disk space in GB.

    Args:
        path: The path where to check for the available space.

    Returns:
        Amount in GB.
    """
    if not path.is_dir():
        return -1.0
    statvfs = os.statvfs(path)
    return statvfs.f_frsize * statvfs.f_bavail / 1024 / 1024 / 1024
