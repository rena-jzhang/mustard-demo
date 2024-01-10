#!/usr/bin/env python3
"""Tools related to speaker diarization."""
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import medfilt

from python_tools import caching, features, generic, time_series

SPEED_OF_SOUND_M_S: Final = 340.29


def tdoa_diarization(
    vads: tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]],
    tdoa: npt.NDArray[np.floating],
    *,
    distance: float = 2.5,
    alpha: float = 0.125,
) -> np.ndarray:
    """Time-of-flight-based diarization.

    Note:
        Time-of-flight-based diarization for 2 people which each have a head-mounted
        microphone. It assumes a perfectly synchronized signal. Avoid noisy and
        multi-source signals.

    Args:
        vads: List of voice activations for each audio signal (logical).
        tdoa: Time delay of arrival shifts (same length as the VAD vectors).
        distance: Distance between the two microphones in meters.
        alpha: Parameter to tune overlapping speech (0: no overlaps allowed).

    Returns:
        Numpy matrix (time x 2) which is True when the person is speaking.
    """
    speaking = np.full((tdoa.shape[0], 2), False)
    threshold = alpha * distance / SPEED_OF_SOUND_M_S
    for i in range(2):
        j = 1 - i
        # only person speaking (good separation)
        index = vads[i] & ~vads[j]
        # break ties based on delay
        if i == 0:
            index |= vads[i] & (tdoa > -threshold)
        else:
            index |= vads[i] & (tdoa < threshold)
        speaking[index, i] = True
    return speaking


@caching.cache_wrapper
def tdoa_diarization_file(
    paths: tuple[Path, Path],
    *,
    cache: Path,
    vad_caches: tuple[Path, Path],
    tdoa_cache: Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """File interface for tdoa_diarization.

    Args:
        paths: List of audio files.
        cache: Save results to this file.
        vad_caches: Where to save/load voice activation detection from.
        tdoa_cache: Where to save/load the time-delay of arrival estimation from.
        **kwargs: Forwarded to audio.speaker_diarization.tdoa_diarization.

    Returns:
        The speech scores in a Pandas dataframe. The dataframe has also a 'time'
        in seconds.
    """
    kwargs.setdefault("max_shift", 3.0 / SPEED_OF_SOUND_M_S)

    # get VAD
    data = {}
    name = "opensmile_vad_opensource"
    for i in range(2):
        tmp = features.extract_features(audio=paths[i], caches={name: vad_caches[i]})
        assert tmp, paths[i]
        data[f"{i}_{name}"] = tmp[f"{name}_csvoutput"]

    # get TDOA
    tmp = features.extract_features(
        audio=paths[0],
        audio2=paths[1],
        caches={"tdoa": tdoa_cache},
        kwargs={"tdoa": kwargs},
    )
    assert tmp, paths
    data.update(tmp)

    # synchronize TDOA and VAD
    synced_data = features.synchronize_sequences(data)
    del data

    # diarization
    speaking = tdoa_diarization(
        (
            (synced_data[f"0_{name}_vad"] >= 0.5).to_numpy(),
            (synced_data[f"1_{name}_vad"] >= 0.5).to_numpy(),
        ),
        synced_data["tdoa_tdoa"].to_numpy(),
        distance=kwargs["max_shift"] * SPEED_OF_SOUND_M_S,
    )
    size = generic.round_up_to_odd(
        0.3 / (synced_data.index[1:] - synced_data.index[:-1]).array.median().item(),
    )
    for i in (0, 1):
        speaking[:, i] = medfilt(speaking[:, i] * 1, size) >= 0.5

    return pd.DataFrame(speaking, columns=["0", "1"], index=synced_data.index)


def volume_diarization(
    *,
    volumes: tuple[npt.NDArray[np.number], ...],
    vads: tuple[npt.NDArray[np.number], ...],
    threshold: float = 0.0,
) -> np.ndarray:
    """Run a simple volume-based diarization.

    Note:
        Simple volume-based diarization for n people using calibrated microphones. It
        assumes that the loudest person is speaking (or everyone above a specified
        threshold).

    Args:
        volumes: List of volumes for each audio signal (negative values).
        vads: List of voice activations for each audio signal (logical; same time
            across all volumes and vads!).
        threshold: When negative, the cutoff to determine speaking across microphones
            (to handle overlapping speech).

    Returns:
        Numpy matrix (time x |speakers|) which is True when the person is speaking.
    """
    # mark as silent when not speaking
    for volume, vad in zip(volumes, vads, strict=True):
        volume[vad < 0.5] = -500

    volume = np.concatenate([volume.reshape(-1, 1) for volume in volumes], axis=1)
    if threshold < 0:
        # provided cutoff
        speaking = volume >= threshold
    else:
        # maximum volume
        speaking = np.full(volume.shape, False)
        speaking[range(speaking.shape[0]), np.argmin(np.stack(volumes), axis=0)] = True

    # mask with VAD
    speaking[volume <= -500] = False

    return speaking


@caching.cache_wrapper
def volume_diarization_file(
    paths: tuple[Path, ...],
    *,
    vad_caches: tuple[Path, ...],
    volume_caches: tuple[Path, ...],
    cache: Path,
    **kwargs: float,
) -> pd.DataFrame:
    """File interface for volume_diarization.

    Args:
        paths: List of audio files.
        vad_caches: Where to save/load voice activation detection from.
        volume_caches: Where to save/load volume from.
        cache: Save results to this file.
        **kwargs: Forwarded to audio.speaker_diarization.volume_diarization.

    Returns:
        The speech scores in a Pandas dataframe. The dataframe has also a 'time'
        in seconds.
    """
    # get volume and VAD
    name = "opensmile_vad_opensource"
    data = {}
    i = 0
    for i, (audio, vad, volume) in enumerate(
        zip(paths, vad_caches, volume_caches, strict=True),
    ):
        tmp = features.extract_features(
            audio=audio,
            caches={"volume": volume, name: vad},
        )
        assert tmp, audio
        data.update({f"{i}_{key}": value for key, value in tmp.items()})
    synced_data = features.synchronize_sequences(data)
    del data

    # get diarization
    result = volume_diarization(
        volumes=tuple(
            synced_data[f"{j}_volume_volume"].to_numpy() for j in range(i + 1)
        ),
        vads=tuple(
            synced_data[f"{j}_{name}_csvoutput_vad"].to_numpy() for j in range(i + 1)
        ),
        **kwargs,
    )

    return pd.DataFrame(
        result,
        columns=[str(j) for j in range(i + 1)],
        index=synced_data.index,
    )


def merge_pauses(
    segments_a: npt.NDArray[np.number],
    segments_b: npt.NDArray[np.number],
    *,
    pause: float = 0.3,
) -> np.ndarray:
    """Merge segments of the first person if the other person is not talking in-between.

    Args:
        segments_a: Matrix with start and stop time.
        segments_b: Matrix with start and stop time.
        pause: Maximum duration of a pause.

    Returns:
        Merged segment_a matrix.
    """
    possible_merges = np.where((segments_a[1:, 0] - segments_a[:-1, 1]) < pause)[0]
    segments_a = segments_a.copy()
    merged = []
    for i in possible_merges:
        index = time_series.interval_overlap(segments_a[[i, i + 1], [1, 0]], segments_b)
        if not np.any(index):
            segments_a[i + 1, 0] = segments_a[i, 0]
            merged.append(i)

    keep_index = np.setdiff1d(np.arange(segments_a.shape[0]), merged)
    return segments_a[keep_index, :]


def diarization_to_subtitles(diarization: pd.DataFrame, *, filename: Path) -> str:
    """Create a subtitle file from the diarization.

    Args:
        diarization: Output from speaker_diarization.
        filename: Path of the to be created subtitle file.

    Returns:
        The written subtitles as a string.
    """
    #  convert to non-overlapping segments
    time = diarization.index.to_numpy()
    a_speaking = diarization["0"].to_numpy()
    b_speaking = diarization["1"].to_numpy()
    both_speaking = a_speaking & b_speaking

    a_segments = time_series.index_to_segments((a_speaking & ~both_speaking), time)
    b_segments = time_series.index_to_segments((b_speaking & ~both_speaking), time)
    both_segments = time_series.index_to_segments(both_speaking, time)
    segments = np.concatenate((a_segments, b_segments, both_segments))
    who = np.concatenate(
        (
            np.full(a_segments.shape[0], "0"),
            np.full(b_segments.shape[0], "1"),
            np.full(both_segments.shape[0], "01"),
        ),
    )
    index = np.argsort(segments[:, 0])
    segments = segments[index, :]
    who = who[index]

    # create subtitles
    return time_series.create_subtitles(
        filename,
        segments[:, 0].tolist(),
        segments[:, 1].tolist(),
        who.tolist(),
    )
