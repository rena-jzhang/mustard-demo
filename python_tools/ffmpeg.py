#!/usr/bin/env python3
"""Tools related to ffmpeg."""
import os
from pathlib import Path
from typing import Final

from python_tools import generic

FFMPEG: Final = "ffmpeg -hide_banner -nostats"


def concat(
    inputs: list[Path],
    output: Path,
    *,
    options: str = "",
    reencode_before: bool = False,
) -> int:
    """Concatenate video/audio files with ffmpeg.

    Args:
        inputs: List of input files.
        output: The output file.
        options: Options inserted before output.
        reencode_before: Whether to re-encode every file before lossless
            concatenating them (might help with some weird formats).

    Returns:
        The returncode of ffmpeg.
    """
    # re-encode files (for streaming protocols)
    if reencode_before:
        extension = output.suffix
        for i, video in enumerate(inputs):
            new_input = Path(generic.random_name()).with_suffix(extension)
            assert (
                generic.run(
                    f"{FFMPEG} -i {video} {options} {new_input}",
                    stdout=None,
                    stderr=None,
                )[0]
                == 0
            )
            inputs[i] = new_input
        if not options:
            options = "-c copy"

    tmp_file = output.parent / generic.random_name()
    tmp_file.write_text(
        "\n".join(f"file '{os.path.relpath(x, tmp_file.parent)}'" for x in inputs),
    )

    command = f"{FFMPEG} -f concat -safe 0 -i {tmp_file} {options} {output}"
    returncode = generic.run(command, stdout=None, stderr=None)[0]

    # delete files
    tmp_file.unlink()
    if reencode_before:
        for video in inputs:
            video.unlink()

    return returncode


def probe(in_file: Path, *, field: str, stream: str = "a") -> str:
    """Return the requested field from the (last) selected stream.

    Args:
        in_file: The audio file.
        field: The requested field.
        stream: Type of the stream.

    Returns:
        Value of the requested field.
    """
    command = (
        f"ffprobe -v quiet -select_streams {stream} -show_entries "
        f"stream={field} -of default=noprint_wrappers=1:nokey=1 {in_file}"
    )
    returncode, stdout, _ = generic.run(command, stderr=None)
    assert returncode == 0, returncode
    return stdout.split("\n")[-2]


def split_stero_into_monos(
    in_file: Path,
    left: Path,
    right: Path,
    *,
    options: str = "",
) -> int:
    """Split in stero audio file into left and right channel.

    Args:
        in_file: The stero audio file.
        left: The to be created left audio file.
        right: The to be created right audio file.
        options: Options inserted before output.

    Returns:
        The returncode of ffmpeg.
    """
    command = (
        f"{FFMPEG} -i {in_file} {options} -vn -filter_complex "
        '"[0]pan=1|c0=c0[left];[0]pan=1|c0=c1[right]" -map "[left]"'
        f' {left} -map "[right]" {right}'
    )
    return generic.run(command, stdout=None, stderr=None)[0]


def ebu_r128(
    in_file: Path,
    out_file: Path,
    *,
    rate: int = -1,
    options: str = "",
    integrated_loudness: int = -24,
    loudness_range: int = 7,
    maximum_true_peak: int = -2,
) -> int:
    """Accurate (linear) EBU R128  volume normalization.

    Note:
        integrated_loudness, loudness_range, and maximum_true_peak are explained
        here ffmpeg.org/ffmpeg-all.html#loudnorm

    Args:
        in_file: The input file.
        out_file: The to be created normalized file.
        rate: Sampling rate of the output file.
        options: Options inserted after the input.
        integrated_loudness: See note.
        loudness_range: See note.
        maximum_true_peak: See note.

    Returns:
        The returncode of ffmpeg.
    """
    # get rate
    if rate < 1:
        rate = int(probe(in_file, field="sample_rate", stream="a"))
    # first run to measure the values
    base = (
        f"{FFMPEG} -i {in_file} {options} -af "
        f"loudnorm=I={integrated_loudness}:TP={maximum_true_peak}"
        f":LRA={loudness_range}:linear=true"
    )
    command = f"{base}:print_format=json -f null -"
    returncode, _, output = generic.run(command, stdout=None)
    if returncode != 0:
        return returncode

    # parse output
    fields = ["input_i", "input_tp", "input_lra", "input_thresh", "target_offset"]
    lines = [
        x.replace('"', "").replace(",", "")
        for x in output.split("\n")
        if ":" in x and len([1 for field in fields if field in x]) == 1
    ]
    measures = {}
    for line in lines:
        key, value = (part.strip() for part in line.split(":"))
        measures[key] = float(value)

    # actual run
    command = (
        base + f':measured_I={measures["input_i"]}'
        f':measured_TP={measures["input_tp"]}'
        f':measured_LRA={measures["input_lra"]}'
        f':measured_thresh={measures["input_thresh"]}'
        f':offset={measures["target_offset"]} -ar {rate} {out_file}'
    )
    return generic.run(command, stdout=None, stderr=None)[0]


def create_black_frames(
    out_file: Path,
    *,
    duration: float = -1.0,
    width: int = -1,
    height: int = -1,
    rate: int | str = -1,
    like_file: Path | None = None,
    options: str = "",
) -> int:
    """Create a black video.

    Args:
        out_file: The file to be created.
        duration: Duration in seconds of the video.
        width: The width of the video.
        height: The height of the video.
        rate: ffmpeg frame rate.
        like_file: Get all the necessary information except duration from an
            existing video file.
        options: Additional ffmpeg options.

    Returns:
        The returncode of ffmpeg.
    """
    # get meta data
    if like_file is not None and like_file.is_file():
        if width < 1:
            width = int(probe(like_file, field="width", stream="v"))
        if height < 1:
            height = int(probe(like_file, field="height", stream="v"))
        if isinstance(rate, int) and rate < 1:
            rate = probe(like_file, field="r_frame_rate", stream="v")
        if duration < 0:
            duration = get_duration(like_file, stream="v")

    # build command
    command = (
        f"{FFMPEG} -f lavfi -i color=black:s={width}x{height}:r={rate}"
        f" -an {options} -t {duration} {out_file}"
    )
    return generic.run(command, stdout=None, stderr=None)[0]


def create_silent_audio(
    out_file: Path,
    *,
    duration: float = -1.0,
    sample_rate: int | str = -1,
    channel_layout: str = "",
    bits: int | str = -1,
    like_file: Path | None = None,
) -> int:
    """Add silence to the beginning of an audio file.

    Args:
        out_file: The to be created audio file.
        duration: The duration of silence.
        sample_rate: The sample rate.
        channel_layout: The channel layout of the new audio file.
        bits: The number of bits for the amplitude.
        like_file: Get all the necessary information except duration from an
            existing audio file.

    Returns:
        The returncode of ffmpeg.
    """
    # get information from input file
    if like_file is not None and like_file.is_file():
        if isinstance(sample_rate, int) and sample_rate < 1:
            sample_rate = probe(like_file, field="sample_rate", stream="a")
        if isinstance(bits, int) and bits < 1:
            bits = probe(like_file, field="sample_fmt", stream="a")
        if not channel_layout:
            channel_layout = probe(like_file, field="channel_layout", stream="a")
            if " " in channel_layout:
                channel_layout = channel_layout.split(" ")[0]
            elif channel_layout == "unknown":
                raise RuntimeError("Could not read channel_layout")
        if duration < 0:
            duration = get_duration(like_file, stream="a")

    # create silence
    command = (
        f"{FFMPEG} -f lavfi -i anullsrc=channel_layout="
        f"{channel_layout}:sample_rate={sample_rate}"
        f" -sample_fmt {bits} -t {duration} {out_file}"
    )
    return generic.run(command, stdout=None, stderr=None)[0]


def seek(
    in_file: Path,
    out_file: Path,
    seconds: float,
    *,
    duration: float = -1.0,
    options: str = "",
    input_options: str = "",
) -> int:
    """Trim the beginning of a file.

    Note:
        It uses input seeking trac.ffmpeg.org/wiki/Seeking

    Args:
        in_file: The input file.
        out_file: The to be created file.
        seconds: Number of seconds to be skipped.
        duration: Duration in seconds of the segment.
        options: Options inserted after the input.
        input_options: Options before '-i'.

    Returns:
        The returncode of ffmpeg.
    """
    assert "-c copy" not in options, options
    command = f"{FFMPEG} {input_options} -ss {seconds} -i {in_file} {options}"
    if duration > 0:
        command += f" -t {duration}"
    command += f" {out_file}"
    return generic.run(command, stdout=None, stderr=None)[0]


def get_duration(video: Path, *, stream: str = "") -> float:
    """Infer the duration from a stream and fallback to the container.

    Args:
        video: The video/audio file.
        stream: Which stream to use.

    Returns:
        The duration in seconds.
    """
    stream_duration = probe(video, stream=stream, field="duration")
    if stream_duration != "N/A":
        return float(stream_duration)

    # try to use container information
    command = (
        "ffprobe -v error -show_entries format=duration -of"
        f" default=noprint_wrappers=1:nokey=1 {video}"
    )
    return float(generic.run(command, stderr=None)[1])


def count_frames(video: Path) -> int:
    """Count the number of frames in a video by decoding it.

    Note:
        This is slow but accurate (meta data might be wrong).

    Args:
        video: The video.

    Returns:
        The number of frames. It is -1 if ffmpeg's return code is non-zero.
    """
    command = f"{FFMPEG} -i {video} -map 0:v:0 -c copy -f null -"
    returncode, _, output = generic.run(command, stdout=None)
    if returncode != 0:
        return -1

    # find 'frame='
    output = [x for x in output.split("\n") if "frame=" in x][-1]
    output = next(x for x in output.split(" ") if "frame=" in x)
    return int(output.split("=")[1].strip())
