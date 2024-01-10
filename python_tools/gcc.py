#!/usr/bin/env python3
"""Tools related to generalized cross-correlation (GCC)."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from python_tools import caching, vectorization_helpers
from python_tools.audio import audio_tools


def next_power_2(number: float) -> int:
    """Calculate the first power of 2 greater than a given number.

    Args:
        number: A floating number.

    Author:
        Chirag Raman <chiragraman>

    Returns:
        An int.
    """
    exp = math.ceil(math.log(number, 2))
    return 2**exp


def phat(corr_coeff: np.ndarray) -> np.ndarray:
    """Compute the PHAT filter.

    Args:
        corr_coeff: The cross-correlation in the frequency domain.

    Returns:
        The filter array to apply to the cross-correlation.

    Author:
        Chirag Raman <chiragraman>
    """
    weight = np.ones(corr_coeff.shape)
    weight[corr_coeff != 0] = 1.0 / abs(corr_coeff[corr_coeff != 0])
    return weight


def gcc(signal_a: np.ndarray, signal_b: np.ndarray) -> np.ndarray:
    """Calculate the Generalized Cross Correlation of two real signals.

    Note:
        Based on David Renzi's MATLAB implementation found here:
        github.com/hsanson/scde/blob/master/src/GCC.m

        Refer "The Generalized Correlation Method for Estimation of Time Delay" by
        Charles Knapp and Clifford Carter for details.

        The GCC is computed with a pre-whitening filter onto the cross-power
        spectrum in order to weight the magnitude value against its SNR. The
        weighted CPS is used to obtain the cross-correlation in the time domain
        with an inverse Fourier transformation.

    Args:
        signal_a: The first symmetric and real-valued signal.
        signal_b: The second symmetric and real-valued signal.

    Returns:
        The centered GCC in time domain

    Author:
        Chirag Raman <chiragraman>
    """
    # variables are re-used to save memory
    if signal_a.ndim == 1:
        signal_a = signal_a.reshape(-1, 1)
    if signal_b.ndim == 1:
        signal_b = signal_b.reshape(-1, 1)

    length = max(signal_a.shape[0], signal_b.shape[0])
    fft_size = next_power_2(length)

    # Transform the input signals
    signal_a = np.fft.rfft(signal_a, n=fft_size, axis=0)
    signal_b = np.fft.rfft(signal_b, n=fft_size, axis=0)

    # Compute the cross-correlation
    signal_a = np.conj(signal_a) * signal_b
    del signal_b

    # Apply the filter
    signal_a = signal_a * phat(signal_a)

    # Estimate the GCC
    signal_a = np.fft.irfft(signal_a, n=fft_size, axis=0)

    return np.fft.fftshift(signal_a, axes=0)


def best_shifts(
    corr_coef: np.ndarray,
    *,
    rate: int = 1,
    max_shift: float = 2.0**16,
    n_shifts: int = 1,
    offset: float = 0.0,
    abs_coef: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Given the output from gcc, it returns the most plausible n shifts.

    Args:
        corr_coef: The GCC of some signals.
        rate: Number of samples per second
        max_shift: The maximum allowed shift in seconds.
        n_shifts: How many shifts
        offset: Search for the best shift around offset +/- max_shift.
        abs_coef: Whether to return and sort by the the absolute coefficients.

    Returns:
        The sorted shifts in seconds and their correlation coefficient.
    """
    max_shift_int = round(max_shift * rate)
    if 2 * max_shift_int > corr_coef.shape[0]:
        max_shift_int = corr_coef.shape[0] // 2 - 1
    offset_int = round(offset * rate)

    # search for shifts only in the intended interval
    center = round(corr_coef.shape[0] / 2.0 + offset_int)
    assert center > max_shift_int
    assert center + max_shift_int < corr_coef.shape[0]
    corr_coef = corr_coef[-max_shift_int + center : max_shift_int + center + 1, :]

    # find the best n shifts
    if abs_coef:
        corr_coef = np.abs(corr_coef)
    index = np.argsort(corr_coef, axis=0)[: -n_shifts - 1 : -1, :]
    shift = index - max_shift_int + offset_int
    shifts = shift / rate
    corr_coef = corr_coef[index, np.arange(corr_coef.shape[1])]
    shifts[np.isnan(corr_coef)] = float("NaN")

    return shifts, corr_coef


def sliding_shifts(
    wave_a: np.ndarray,
    wave_b: np.ndarray,
    *,
    rate: int = 16_000,
    max_shift: float = 2.0**16,
    window_size: float = 1.0 / 4,
    shift: float = 1.0 / 100,
    n_shifts: int = 1,
    min_std: float = 0.0,
    offset: float = 0.0,
    abs_coef: bool = True,
) -> pd.DataFrame:
    """Calculate shifts for a rolling window.

    Args:
        wave_a: 1d numpy array for the the first signal.
        wave_b: 1d numpy array for the the second signal (of the same size).
        rate: Number of samples per second.
        max_shift: Do not search for shifts larger than this duration.
        window_size: Window size in seconds.
        shift: Shift in seconds.
        n_shifts: Uses the n most prominent shifts to create a smoothed version.
        min_std: Minimal std of window to compute shift for (NaN is returned if
            not enough variance is available).
        offset: Add this number of second to the found shift.
        abs_coef: Whether to take the absolute value of the coefficients.

    Returns:
        Pandas DataFrame with the columns: tdoa and cc.
    """
    assert wave_a.size == wave_b.size

    # estimate memory usage per entry
    window_length = round(window_size * rate)
    memory_usage = (
        8 * (3 + 1) * (next_power_2(window_length) / 2 + 1)
        + 8 * (2 + 1) * n_shifts * window_length
        + 8 * 3
    )
    gen_a = vectorization_helpers.sliding_partition(
        wave_a,
        memory_usage=memory_usage,
        rate=rate,
        shift=shift,
        window_size=window_size,
    )
    partition_size, time = next(gen_a)
    assert isinstance(partition_size, int)
    gen_b = vectorization_helpers.sliding_partition(
        wave_b,
        memory_usage=memory_usage,
        rate=rate,
        shift=shift,
        window_size=window_size,
        partition_size=partition_size,
    )
    partition_size_b, time_b = next(gen_b)
    assert partition_size == partition_size_b
    assert np.all(time == time_b)

    corr_coef = np.full((n_shifts, time.size), np.nan)
    tdoa = np.full(corr_coef.shape, np.nan)
    window = np.hanning(window_length).reshape(-1, 1)
    for partition_a, index in gen_a:
        partition_b = next(gen_b)[0]
        assert isinstance(partition_a, np.ndarray)
        assert isinstance(partition_b, np.ndarray)
        assert np.all(partition_a.shape == partition_b.shape)

        # ignore windows with too little variance
        std_index = (np.std(partition_a, axis=0) > min_std) & (
            np.std(partition_b, axis=0) > min_std
        )
        if std_index.sum() == 0:
            continue
        partition_a = partition_a[:, std_index]
        partition_b = partition_b[:, std_index]
        index = index[std_index]

        # symmetric
        partition_a *= window
        partition_b *= window

        # vectorized gcc-phat consumes much memory!
        assert isinstance(partition_a, np.ndarray)
        assert isinstance(partition_b, np.ndarray)
        tdoa[:, index], corr_coef[:, index] = best_shifts(
            gcc(partition_a, partition_b),
            rate=rate,
            max_shift=max_shift,
            n_shifts=n_shifts,
            offset=offset,
            abs_coef=abs_coef,
        )

    # viterbi smoothing
    if corr_coef.shape[0] > 1:
        corr_coef, tdoa = viterbi_smoothing_for_shifts(corr_coef, tdoa, max_shift)

    return pd.DataFrame({"tdoa": tdoa[0, :], "cc": corr_coef[0, :]}, index=time)


@caching.cache_wrapper
def sliding_shifts_file(
    path_a: Path,
    path_b: Path,
    *,
    cache: Path,
    rate_max: int = 16_000,
    max_shift: float = 2.0**16,
    window_size: float = 1.0 / 4,
    shift: float = 1.0 / 100,
    n_shifts: int = 1,
    min_std: float = 0.0,
    offset: float = 0.0,
    abs_coef: bool = True,
) -> pd.DataFrame:
    """Return the smoothed time delay of arrival between two audio files.

    Args:
        path_a: Path to first audio file.
        path_b: Path to second audio file.
        cache: Path to the cache file.
        rate_max: Upper bound for the sampling rate.
        max_shift: Do not search for shifts larger than this duration.
        window_size: Window size in seconds.
        shift: Shift in seconds.
        n_shifts: Uses the n most prominent shifts to create a smoothed version.
        min_std: Minimal std of window to compute shift for (NaN is returned if
            not enough variance is available).
        offset: Add this number of second to the found shift.
        abs_coef: Whether to take the absolute value of the coefficients.

    Returns:
        Pandas DataFrame with the columns: 'time', 'tdoa', and 'cc'.
    """
    # read audio files
    wave_a, rate_a = audio_tools.read_audio(path_a, mono=0)
    wave_b, rate_b = audio_tools.read_audio(path_b, mono=0)

    # resample
    rate = min(rate_max, rate_a, rate_b)
    wave_a = audio_tools.resample(wave_a, rate_a, rate)
    wave_b = audio_tools.resample(wave_b, rate_b, rate)

    # cut to shortest duration
    if wave_a.size < wave_b.size:
        wave_b = wave_b[: wave_a.size]
    else:
        wave_a = wave_a[: wave_b.size]

    # wrapper for GCC-PHAT
    return sliding_shifts(
        wave_a,
        wave_b,
        rate=rate,
        max_shift=max_shift,
        window_size=window_size,
        shift=shift,
        n_shifts=n_shifts,
        min_std=min_std,
        offset=offset,
        abs_coef=abs_coef,
    )


def viterbi_smoothing_for_shifts(
    corr_coef: np.ndarray,
    tdoa: np.ndarray,
    delay_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a smooth time series of shifts.

    Args:
        corr_coef: Correlation coefficients.
        tdoa: Shifts in seconds.
        delay_max: Maximal shift in seconds.

    Returns:
        Tuple of length 2: correlation values of the smooth shift and the
        smoothed shift.
    """
    scores = np.ones(np.array(corr_coef.shape)[[0, 0, 1]])
    for i in range(1, corr_coef.shape[1]):
        scores[:, :, i] = 2.1 * delay_max - np.abs(tdoa[:, i - 1, None] - tdoa[:, i].T)
    scores *= corr_coef
    index = viterbi(np.log(scores), only_states=True)[0]
    time_index = np.arange(corr_coef.shape[1])

    return corr_coef[index, time_index], tdoa[index, time_index]


def viterbi(
    scores: np.ndarray,
    *,
    only_states: bool = True,
) -> tuple[np.ndarray, float]:
    """Find the most likely state sequence and its log probability.

    Args:
        scores: A |states| x |states| x time matrix. scores[i, j, t] is the score
            associated with going from the i-th state (t-1) to the j-th state at
            time t. scores[i, i, 0] are the initial probabilities for state i.
        only_states: Whether only the state sequence is important.

    Returns:
        A numpy array containing the most likely state sequence and the
        log probability of the sequence.
    """
    best_parent = np.full(scores.shape[1:], -1, dtype=np.int16)
    best_parent[:, 0] = np.arange(scores.shape[0])
    alpha = scores[best_parent[:, 0], best_parent[:, 0], 0]

    for i in range(1, scores.shape[-1]):
        possible_alphas = scores[:, :, i] + alpha[:, None]
        assert np.all(possible_alphas <= 0)
        best_parent[:, i] = np.argmax(possible_alphas, axis=0)
        alpha = possible_alphas[best_parent[:, i], best_parent[:, 0]]

        # avoid numeric issues
        if only_states and np.any(alpha < -1e100):
            alpha = alpha - alpha.min()

    # reconstruct best path
    best = np.zeros(scores.shape[-1], dtype=int)
    best[-1] = np.argmax(alpha)
    for i in range(scores.shape[-1] - 1, 0, -1):
        best[i - 1] = best_parent[best[i], i]

    return best, np.max(alpha).item()
