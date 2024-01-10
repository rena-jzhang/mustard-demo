#!/usr/bin/env python3
"""Tools to process audio file(s)."""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from python_tools import gcc


def read_audio(
    path: Path,
    *,
    mono: int = -1,
    norm: bool = False,
    norm_type: type = np.float32,
) -> tuple[np.ndarray, int]:
    """Read an audio file.

    Args:
        path: Path from which the audio is read.
        mono: If positive, returns only the specified channel.
        norm: Whether to normalize the signal to [-1, 1].
        norm_type: The data type of the normalized audio signal.

    Returns:
        Tuple of length 2: audio signal and number of samples per second.
    """
    # read audio
    wave, rate = sf.read(str(path))

    # make sure the first dimension is time
    if wave.ndim == 2 and wave.shape[1] > wave.shape[0]:
        wave = wave.T

    # select a channel
    if mono >= 0 and wave.ndim == 2:
        wave = wave[:, mono]

    # normalize audio
    if norm:
        normalize_audio(wave, norm_type=norm_type)

    return wave, rate


def normalize_audio(audio: np.ndarray, *, norm_type: type = np.float32) -> np.ndarray:
    """Normalize an audio signal.

    Args:
        audio: The audio signal (numpy array)
        norm_type: The data type of the normalized audio signal.

    Returns:
        The normalized audio signal.
    """
    # temporarily convert to double (might be an integer)
    if audio.dtype != np.float64:
        audio = audio.astype(np.float64)

    # normalize
    channel_max = np.abs(audio).max(axis=0)
    if isinstance(channel_max, np.ndarray):
        channel_max[channel_max == 0] = 1
    elif channel_max == 0:
        channel_max = 1
    audio /= channel_max

    # scale if output type is an integer
    if norm_type(0.1) == 0:
        audio = audio * float(2 ** (norm_type().itemsize * 8 - 1))

    return audio.astype(norm_type)


def write_audio(path: Path, wave: np.ndarray, rate: int) -> None:
    """Write the audio signal to a file.

    Args:
        path: Path where the audio file is written to.
        wave: The audio signal.
        rate: Number of samples per second.
    """
    sf.write(str(path), wave, rate)


def resample(wave: np.ndarray, rate_old: int, rate_new: int) -> np.ndarray:
    """Resample an audio signal.

    Args:
        wave: The original audio.
        rate_old: The current sampling rate.
        rate_new: The new sampling rate.

    Returns:
        The resampled audio signal.
    """
    if rate_old == rate_new:
        return wave

    samples = round(len(wave) * rate_new / rate_old)
    return signal.resample(wave, samples)


def synchronize_audio_file(
    audio_a: Path,
    audio_b: Path,
    *,
    offset: float = 0.0,
    max_shift: float = 2.0**16,
) -> tuple[float, float]:
    """Synchronize two audio files with GCC-PHAT (wrapper for synchronize_audio).

    Args:
        audio_a: Path to first audio file.
        audio_b: Path to second audio file.
        max_shift: Maximum shift in seconds.
        offset: Search for the best shift around offset +/- max_shift.

    Returns:
        The estimated offset between the two audio files.
        And the correlation coefficient.
    """
    # read audio and sample to same rate
    wave_a, rate_a = read_audio(audio_a, mono=0, norm=True)
    wave_b, rate_b = read_audio(audio_b, mono=0, norm=True)
    rate = min(rate_a, rate_b)
    wave_a = resample(wave_a, rate_a, rate)
    wave_b = resample(wave_b, rate_b, rate)

    # Cross-correlation
    shift, corr = gcc.best_shifts(
        gcc.gcc(wave_a, wave_b),
        max_shift=max_shift,
        rate=rate,
        offset=offset,
    )
    return shift[0, 0], corr[0, 0]
