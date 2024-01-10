#!/usr/bin/env python3
"""Tools related to audio analysis."""
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import generate_binary_structure, morphology
from scipy.signal import find_peaks, peak_prominences

from python_tools import caching, features, time_series, vectorization_helpers
from python_tools.audio import audio_tools


def calculate_volume(
    wave: np.ndarray,
    *,
    rate: int = 16_000,
    window_size: float = 0.06,
    window_shift: float = 0.01,
) -> pd.DataFrame:
    """Calculate the volume of an audio signal.

    Args:
        wave: The one dimensional audio signal.
        rate: The sampling rate.
        window_size: The window size.
        window_shift: The shift of the window.

    Returns:
        Pandas table with 'volume' and 'time'.
    """
    # partition data
    wave = np.power(wave, 2) + np.finfo(np.float64).eps
    partionor = vectorization_helpers.sliding_partition(
        wave,
        rate=rate,
        shift=window_shift,
        window_size=window_size,
    )
    _, time = next(partionor)
    volume = np.zeros(time.shape)
    hanning = np.hanning(round(window_size * rate)).reshape(-1, 1)

    # calculate volume
    for partition, index in partionor:
        volume[index] = np.mean(partition * hanning, axis=0)
    return pd.DataFrame({"volume": 10 * np.log10(volume)}, index=time)


@caching.cache_wrapper
def calculate_volume_file(
    path: Path,
    *,
    cache: Path,
    **kwargs: float,
) -> pd.DataFrame:
    """File-based interface for calculate_volume.

    Args:
        path: Path to the audio file.
        cache: Location were to save the cached result.
        **kwargs: Forwarded to audio.audio_analysis.calculate_volume.

    Returns:
        Pandas table with 'volume' and 'time'.
    """
    wave, rate = audio_tools.read_audio(path, mono=0)
    return calculate_volume(wave, rate=rate, **kwargs)


def signal_to_noise_ratio(volume: np.ndarray, masks: list[np.ndarray]) -> pd.DataFrame:
    """Calculate the signal of noise ratio (SNR) of logarithmic signal.

    Args:
        volume: A 1d-array representing the volume.
        masks: A list of masks. A SNR between each mask will be calculated.

    Returns:
        A matrix which contains all possible combination of SNRs.
    """
    snrs = np.zeros((len(masks), len(masks)))
    for i, mask_a in enumerate(masks):
        signal_a = np.nanmedian(volume[mask_a])
        for j, mask_b in enumerate(masks[i + 1 :], i + 1):
            signal_b = np.nanmedian(volume[mask_b])
            snrs[i, j] = signal_a - signal_b
            snrs[j, i] = -snrs[i, j]
    return pd.DataFrame(snrs)


def signal_to_noise_ratio_elan(
    elan_file: Path,
    tier_lists: tuple[list[str], ...],
    audio_file: Path,
    *,
    volume_cache: Path,
    vad_cache: Path,
    masks: list[np.ndarray],
    cache: Path | None = None,
) -> pd.DataFrame | None:
    """Elan interface for signal_to_noise_ratio_file.

    Args:
        elan_file: Optional elan file.
        tier_lists: List of list containing tier names which are combined to form
            each one mask.
        audio_file: The audio file.
        volume_cache: Cache file for the volume.
        vad_cache: Cache file for the VAD.
        masks: Optional list of masks. The first element needs to be a time vector.
        cache: Save the result to this file.

    Returns:
        A matrix which contains all possible combination of SNRs.
    """
    elan = time_series.elan_to_dataframe(elan_file, shift=0.01)
    masks = [elan.index.to_numpy()] + [
        elan.loc[:, tiers].any(axis=1).to_numpy() for tiers in tier_lists
    ]
    return signal_to_noise_ratio_file(
        audio_file,
        volume_cache=volume_cache,
        vad_cache=vad_cache,
        cache=cache,
        masks=masks,
    )


@caching.cache_wrapper
def signal_to_noise_ratio_file(
    audio_file: Path,
    *,
    volume_cache: Path,
    vad_cache: Path,
    masks: list[np.ndarray],
    cache: Path,
) -> pd.DataFrame:
    """File interface for signal_to_noise_ratio.

    Args:
        audio_file: The audio file.
        volume_cache: Cache file for the volume.
        vad_cache: Cache file for the VAD.
        masks: Optional list of masks. The first element needs to be a time vector.
        cache: Save the result to this file.

    Returns:
        A Numpy matrix which contains all possible combination of SNRs.
    """
    masks = masks or []

    # get VAD & volume
    data = features.extract_features(
        audio=audio_file,
        caches={"opensmile_vad_opensource": vad_cache, "volume": volume_cache},
    )
    assert data, audio_file
    synced_data = features.synchronize_sequences(data)
    del data
    vad = synced_data["opensmile_vad_opensource_csvoutput_vad"].to_numpy()
    volume = synced_data["volume_volume"].to_numpy()
    vad_silence = vad < 0.0
    vad = vad > 0.5

    # define 100ms erosion
    iterations = round(0.1 / np.median(np.diff(synced_data.index.to_numpy())))
    erosion = partial(
        morphology.binary_erosion,
        structure=generate_binary_structure(1, 1),
        iterations=iterations,
    )
    vad = erosion(vad)
    vad_silence = erosion(vad_silence)

    # use existing masks
    if masks:
        sync_fun = partial(
            interp1d,
            assume_sorted=True,
            bounds_error=False,
            fill_value=0,
        )
        time = masks[0]
        del masks[0]
        for i, mask in enumerate(masks):
            mask = sync_fun(time, mask * 1)(synced_data.index.to_numpy()) > 0.5
            mask &= vad
            masks[i] = mask
    else:
        masks.append(vad)

    masks.append(vad_silence)
    return signal_to_noise_ratio(volume, masks)


def find_syllable_nuclei(
    volume: np.ndarray,
    voicing: np.ndarray,
    *,
    silence_threshold: float = 25.0,
    dip_threshold: float = 2.0,
) -> np.ndarray:
    """Find the nucleus of syllables (the loudest voiced part, usually a vowel).

    Note:
        Tries to mimic
        sites.google.com/site/speechrate/Home/praat-script-syllable-nuclei-v2

    Args:
        volume: A vector containing the volume information (in dB);
        voicing: A binary vector indicating whether someone is voicing.
        silence_threshold: Peaks below 99th-percentile - this threshold are ignored.
        dip_threshold: Peaks have to have a dip of this threshold before them.

    Returns:
        A binary vector indicating at which position a nucleus is.
    """
    min_volume = np.percentile(volume, 99) - silence_threshold
    peaks = find_peaks(volume, height=min_volume)[0]
    left_bases = peak_prominences(volume, peaks)[1]
    index = (volume[peaks] - volume[left_bases] > dip_threshold) & (voicing[peaks])
    return peaks[index]


@caching.cache_wrapper
def sliding_speaking_rate_file(
    audio: Path,
    *,
    vad_cache: Path,
    volume_cache: Path,
    diarization_kwargs: dict[str, Any] | None = None,
    speaker: str | None = None,
    duration: float = 4.0,
    cache: Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Calculate sliding speaking rate - file-based interface.

    Args:
        audio: List of audio files (either one or two files).
        vad_cache: Location of cached VAD.
        volume_cache: Location of cached volume.
        diarization_kwargs: All kwargs for features.extract_features to get
            'tdoa_diarization' (optional).
        speaker: Which speaker to use from the diarization (optional).
        duration: Parameter for sliding_speaking_rate.
        cache: Save the result to this file.
        **kwargs: Parameters for find_syllable_nuclei.

    Returns:
        A dataframe with the columns: nuclei, rate, and duration.
    """
    # get volume and vad
    vad_name = "opensmile_vad_opensource"
    data = features.extract_features(
        audio=audio,
        caches={"volume": volume_cache, vad_name: vad_cache},
    )
    assert data, audio

    # get&apply diarization (if asked for)
    diarization_name = "diarization_tdoa"
    if diarization_kwargs is not None:
        tmp = features.extract_features(**diarization_kwargs)
        assert tmp, audio
        data[diarization_name] = tmp[diarization_name]
    synced_data = features.synchronize_sequences(data)
    del data

    # speech rate
    vad = synced_data[f"{vad_name}_csvoutput_vad"].to_numpy() >= 0.5
    if diarization_kwargs is not None:
        vad &= synced_data[f"diarization_tdoa_{speaker}"].to_numpy() >= 0.5
    nuclei_ = find_syllable_nuclei(
        synced_data["volume_volume"].to_numpy(),
        vad,
        **kwargs,
    )
    nuclei = np.full(synced_data.shape[0], False)
    nuclei[nuclei_] = True
    rate_, support_, time_ = sliding_speaking_rate(
        nuclei_,
        vad,
        int(np.median(np.diff(synced_data.index.to_numpy()))),
        duration=duration,
    )

    # align windowed speech rate
    time = np.round(synced_data.index.to_numpy(), 4)
    time_ = np.round(time_, 4)
    rate = np.full(nuclei.size, np.NaN)
    index = (time >= time_.min()) & (time <= time_.max())
    rate[index] = rate_
    support = np.full(nuclei.size, np.NaN)
    support[index] = support_

    return pd.DataFrame(
        {"nuclei": nuclei, "rate": rate, "duration": support},
        index=synced_data.index,
    )


def sliding_speaking_rate(
    nuclei: np.ndarray,
    vad: np.ndarray,
    rate: int,
    *,
    duration: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimates the speaking rate at every moment.

    Args:
        nuclei: Active indices of nuclei (index into vad).
        vad: Voice activation detection (logical).
        rate: Sampling rate of VAD.
        duration: Maximum of time used to calculate the speech rate.

    Returns:
        A tuple of size 2:
        0) Array of the shape of VAD with the estimated speaking rate.
        1) Array of shape of VAD indicating how much speaking time was used
           (confidence measure).
    """
    # unroll active indices
    nucleus = np.zeros(vad.shape)
    nucleus[nuclei] = 1
    nucleus[~vad] = float("NaN")

    window_size = round(duration / rate)
    iterator = vectorization_helpers.sliding_partition(
        nucleus,
        rate=1,
        shift=1,
        window_size=window_size,
    )
    length, time = next(iterator)
    time = time * rate
    assert isinstance(length, int)
    rates = np.zeros(length)
    support = np.zeros(length)
    for windows, index in iterator:
        support[index] = np.mean(~np.isnan(windows), axis=0) * duration
        rates[index] = np.nansum(windows, axis=0) / support[index]

    return rates, support, time


def intra_inter_pauses(
    segments_a: np.ndarray,
    segments_b: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate the duration of intra and inter pauses.

    Note:
        An inter pause is the time between B stopping and A starting to speak.

    Args:
        segments_a: Matrix indicating the start and stop of each segment.
        segments_b: Matrix indicating the start and stop of each segment.

    Returns:
        A dictionary with the keys 'intra_pauses' and 'inter_pauses'.
    """
    results: dict[str, list[float]] = {"intra_pauses": [], "inter_pauses": []}

    # first inter pause before A is speaking
    index = np.where(segments_a[0, 0] > segments_b[:, 1])[0]
    if (
        index.size != 0
        and not time_series.interval_overlap(
            np.asarray([segments_a[0, 0], segments_a[0, 0] + 1e-8]),
            segments_b,
        ).any()
    ):
        results["inter_pauses"].append(segments_a[0, 0] - segments_b[index[-1], 1])

    for i in range(segments_a.shape[0] - 1):
        # is other person speaking in-between?
        between = time_series.interval_overlap(
            np.asarray([segments_a[i, 1], segments_a[i + 1, 0]]),
            segments_b,
        )
        between = np.where(between)[0]
        begin = segments_a[i, 1]
        key = "intra_pauses"
        if between.size != 0 and segments_b[between[-1], 1] < segments_a[i + 1, 0]:
            # inter pause
            begin = segments_b[between[-1], 1]
            key = "inter_pauses"
        elif between.size != 0:
            # overlapping speech
            continue
        results[key].append(segments_a[i + 1, 0] - begin)

    return {key: np.asarray(value) for key, value in results.items()}
