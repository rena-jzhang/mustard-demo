#!/usr/bin/env python3
"""Tools related to opensmile."""
from pathlib import Path
from typing import Any

import pandas as pd

from python_tools import caching, ffmpeg, generic


@caching.cache_wrapper
def run(
    *,
    audio_dict: dict[str, Path],
    cache_options: dict[str, Any] | None = None,
    time: str = "frameTime",
    start: float = 0.0,
    stop: float = -1.0,
    cache: Path,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Run openSMILE with the provided keywords as arguments.

    Args:
        audio_dict: Is treated as kwargs but the audio is converted
            automatically if the format does not match.
        cache_options: Read the created CSV file with the provided
            options (forwarded to pd.read_csv).
        time: The column indicating time.
        start: The start in seconds.
        stop: The stop in seconds.
        cache: Save the result to this file.
        **kwargs: Forwarded to SMILExtract.

    Returns:
        A dictionary with DataFrames.
    """
    audio_dict = audio_dict or {}
    cache_options = cache_options or {"delimiter": ";"}

    # get CSV outputs from opensmile
    csvs, others = outputs(kwargs["C"])
    assert csvs, f"Did not find any CSV outputs for {kwargs['C']}!"
    basename = cache.name.split(".", 1)[0]
    csv_dict = {csv: cache.parent / f"{basename}_{csv}.csv" for csv in csvs}
    for other in others:
        kwargs.setdefault(other, "?")

    # prepare audio
    delete_files = []
    for key in audio_dict:
        audio_dict[key], delete = prepare_audio(
            audio_dict[key],
            start=start,
            stop=stop,
            folder=cache.parent,
        )
        if delete:
            delete_files.append(audio_dict[key])

    # command
    kwargs.update(csv_dict)
    kwargs.update(audio_dict)
    kwargs["nologfile"] = 1
    command = "SMILExtract " + " ".join(
        f"-{key} {value}" for key, value in kwargs.items()
    )

    # execute
    returncode, _, error = generic.run(command, stdout=None)

    # clean up
    for delete_file in delete_files:
        delete_file.unlink()

    if returncode != 0:
        # remove cache files
        for csv_file in csv_dict.values():
            csv_file.unlink(missing_ok=True)
        raise RuntimeError(
            f'"{command}" exited with {returncode}:\n{error}',
        )

    return _get_dataframes(csv_dict, cache_options, time, start)


def _get_dataframes(
    csv_dict: dict[str, Path],
    cache_options: dict[str, Any],
    time: str,
    start: float,
) -> dict[str, pd.DataFrame]:
    """Remove duplicated time columns and save the results.

    Args:
        csv_dict: Mapping of short names and the saved CSV files.
        cache_options: Forwarded to pd.read_csv.
        time: Which column represent the time.
        start: Add this number of seconds to the time column.

    Returns:
        The dataframes extracted by opensmile.
    """
    results: dict[str, pd.DataFrame] = {}
    for key, csv_file in csv_dict.items():
        # read opensmile's csv file
        results[key] = pd.read_csv(csv_file, index_col=time, **cache_options)
        csv_file.unlink()
        # remove duplicate rows
        results[key] = results[key][~results[key].index.duplicated(keep="last")]  # pyright: ignore[reportGeneralTypeIssues]
        # add offset to time
        results[key].index += start
        # set index
        results[key] = results[key].drop(
            columns=["Unnamed: 0", "name"],
            errors="ignore",
        )

    return results


def prepare_audio(
    audio_path: Path,
    *,
    start: float = 0.0,
    stop: float = -1.0,
    folder: Path,
) -> tuple[Path, bool]:
    """Make sure that the audio is a wav file with one mono channel (center).

    Args:
        audio_path: Path to the audio file.
        start: Start in seconds.
        stop: End in seconds.
        folder: New audio files will be created in this folder.

    Returns:
        Tuple of size two: Path to (corrected) audio and boolean whether
        the audio file has been corrected.
    """
    corrected = start != 0 or stop != -1
    new_path = audio_path

    # is wav file?
    if audio_path.suffix.lower() != "wav":
        corrected = True

    # is mono?
    channels = ffmpeg.probe(audio_path, field="channels", stream="a")
    channel_layout = ffmpeg.probe(audio_path, field="channel_layout", stream="a")
    if channels != "1" or channel_layout not in [""]:
        corrected = True

    if corrected:
        new_path = folder / f"{generic.random_name()}.wav"
        assert (
            ffmpeg.seek(
                audio_path,
                new_path,
                start,
                duration=stop - start,
                options="-vn -channel_layout 4",
            )
            == 0
        )
    return new_path, corrected


def list_configs(*, folders: tuple[Path, ...]) -> list[Path]:
    """Return a list of all openSMILE config files.

    Args:
        folders: List of folder to search for config files.

    Returns:
        A list of config files.
    """
    configs = []
    for folder in folders:
        folder = generic.fullpath(folder)
        configs.extend(list(folder.glob("**/*.conf")))
    return configs


def outputs(config: Path) -> tuple[list[str], list[str]]:
    """List the CSV and other output options of the openSMILE config.

    Args:
        config: Path to the config file.

    Returns:
        A tuple of size two:
        0) List of CSV output options;
        1) List of other output options.
    """
    csvs = []
    others = []
    command = f"SMILExtract -nologfile -c -C {config}"
    returncode, _, stderr = generic.run(command, stdout=None)
    if returncode not in (0, 255):
        raise RuntimeError(f'"{command}" exited with {returncode}:\n{stderr}')

    # parse output
    for line in stderr.split("\n"):
        if "output" not in line or "<string>" not in line:
            continue
        line = line.split("<string>", 1)[0].split(",")[-1].strip()
        if not line.startswith("-"):
            continue
        line = line[1:]
        if "csv" in line:
            csvs.append(line)
        else:
            others.append(line)
    return csvs, others
