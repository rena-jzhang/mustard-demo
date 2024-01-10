#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues = false
"""Tools related to extracting features/statistics."""
import re
import shutil
import warnings
from collections.abc import Callable
from itertools import chain
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import pandas as pd
import pympi
from scipy.signal import medfilt

from python_tools import caching, gcc, generic, time_series
from python_tools.audio import audio_analysis, opensmile, speaker_diarization

T = TypeVar("T")


def _run_matlab(command: str, **kwargs: Any) -> tuple[int, str, str]:
    """Run a MATLAB command and catch stdout/err."""
    command = (
        f"try, {command}; catch err, fprintf(2, getReport(err)); exit(1); end; exit(0)"
    )
    return generic.run(
        [
            "matlab",
            "-nodisplay",
            "-nojvm",
            "-nodesktop",
            "-nosplash",
            "-singleCompThread",
            "-r",
            command,
        ],
        **kwargs,
    )


@caching.cache_wrapper
def run_covarep_vowel_space(
    *,
    diarization: pd.Series,
    covarep_cache: Path,
    gender: int = 0,
    covarep: Path | None = None,
    cache: Path,
) -> pd.DataFrame:
    """Get COVAREP's vowelspace knowing when the person is speaking.

    Args:
        diarization: A Pandas Series indicating when the formants should be used.
        covarep_cache: Existing(!) covarep cache which contains at least VUV, F1 and F2.
        gender: The gender of the person speaking: 0: female, 1: male, and 2: child.
        covarep: Path to the COVAREP directory.
        cache: Where to save the vowel space.

    Returns:
        DataFrame containing only the vowelspace.
    """
    covarep = covarep or generic.fullpath(Path("~/git/covarep"))

    # load COVAREP
    data_none = caching.read_hdfs(covarep_cache).get("df", None)
    assert data_none is not None, covarep_cache
    data = data_none
    assert "F1" in data.columns
    assert "F2" in data.columns
    assert "VUV" in data.columns

    # synchronize diarization and COVAREP
    diarization.name = "0"
    data = synchronize_sequences(
        {
            "covarep": data.loc[:, ["F1", "F2", "VUV"]],
            "diarization": pd.DataFrame(diarization),
        },
    )
    index = (data["diarization_0"] > 0.3) & (data["covarep_VUV"] > 0.3)
    data = data.loc[index, ["covarep_F1", "covarep_F2"]]

    # build command
    csv_path = generic.fullpath(Path(f"{generic.random_name()}.csv"))
    data.to_csv(csv_path, header=False, index=False)
    command = (
        f"cd {covarep}; startup; disp(['vowelspace ', "
        f"num2str(getVowelSpace(csvread('{csv_path}'), {int(gender)}), 8)])"
    )

    # run command
    returncode, output, error = _run_matlab(command)
    csv_path.unlink(missing_ok=True)
    if returncode != 0:
        raise RuntimeError(f'"{command}" exited with {returncode}:\n{error}')

    # extract vowelspace
    vowelspace = next(
        float(line.split()[1])
        for line in output.split("\n")
        if line.startswith("vowelspace")
    )
    return pd.DataFrame({"vowelSpace": [vowelspace]})


@caching.cache_wrapper
def run_covarep(
    audio_path: Path,
    *,
    covarep: Path | None = None,
    cache: Path,
    gender: int = 0,
    features: tuple[str, ...] = (),
    channel: int = 0,
    feature_sampling: float = 0.01,
    start: float = 0.0,
    stop: float = -1.0,
) -> pd.DataFrame:
    """Run COVAREP on an audio file and returns its features.

    Args:
        audio_path: Path to the audio file.
        cache: Path to the cache.
        covarep: Path to the COVAREP directory.
        features: List of features to be extracted.
        gender: Necessary for vowel space (0 female, 1 male, 2 child).
        channel: Which channel of the audio to use (starting from 0).
        feature_sampling: Sample feature ever n seconds.
        start: When to start the feature extraction (in seconds).
        stop: When to stop the feature extraction (in seconds).

    Returns:
        A pandas table containing the output from COVAREP.
    """
    covarep = covarep or generic.fullpath(Path("~/git/covarep"))

    # COVAREP uses CSVs
    audio_path = generic.fullpath(audio_path)
    cache = caching.add_extension(cache, "csv")

    # build COVAREP command
    command = (
        f"struct('gender', {gender}, 'channel', {channel+1}, "
        f"'feature_fs', {feature_sampling}, 'save_mat', false, "
        f"'save_csv', '{cache}', 'start', {start}, 'stop', {stop}"
    )
    if features:
        command += ", 'features', {{"
        for feature in features:
            command += f"'{feature}', "
        command = command[:-2] + "}}"
    command = (
        f"cd {covarep}; startup; COVAREP_feature_formant_extraction_perfile("
        f"'{audio_path}', {command}))"
    )

    # run COVAREP
    returncode, _, error = _run_matlab(command, stdout=None)
    if returncode != 0:
        cache.unlink(missing_ok=True)
        raise RuntimeError(f'"{command}" exited with {returncode}:\n{error}')

    result = pd.read_csv(cache, index_col="time")
    cache.unlink(missing_ok=True)
    return result


def run_mfa(
    *,
    folder: Path,
    output_folder: Path,
    dictionary: Path | None = None,
    model: str | Path = "english",
    options: dict[str, Any],
    check_cache_only: bool = False,
    delete_textgrid: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run Montreal Forced Aligner on a folder.

    Note:
        If no TextGrid exists, ELAN files will temporarily be converted to TextGrid.
        Non-wav files will temporarily be converted to compatible wav files.
        Existing wav files have to be compatible with MFA!
        Please read the documentation of MFA (tier names are used for speaker
        adaptation by default)!

    Args:
        folder: Folder containing pairs of audio files and ELAN/TextGrid files.
        output_folder: Where to save the resulting ELAN and HDF files to.
        dictionary: Name or path to a dictionary.
        model: Name or path of an acoustic model.
        options: Other options forwarded to mfa_align. temp_directory will be deleted!
        delete_textgrid: Whether to delete the TextGrid after it is converted to ELAN.
        check_cache_only: If cache exists, load and return it. Otherwise, return an
            empty dict.

    Returns:
        A dictionary of DataFrames.
    """
    dictionary = dictionary or generic.fullpath(
        Path("~/local/mfa/pretrained_models/librispeech-lexicon.txt"),
    )

    # options
    options = options or {}
    options.setdefault("beam", 16)
    options.setdefault(
        "temp_directory",
        generic.fullpath(output_folder.parent) / generic.random_name(),
    )
    options.setdefault("num_jobs", 8)

    # find all textgrids and elan files. If they have a matching audio files, convert
    # elan and non-wav files
    pairs = []
    transcripts = []
    audios = []
    for transcript in chain(folder.glob("*.TextGrid"), folder.glob("*.eaf")):
        # is there a matching audio/video file
        audio_files = transcript.parent.glob(
            f"{transcript.stem}.[womaf][agpkvl][vg34ia]*",
        )
        try:
            audio = next(audio_files)
        except StopIteration:
            continue
        audios.append(audio)
        transcripts.append(transcript)
        pairs.append(transcript.stem)

    # are all HDF files present
    hdf_files = [(output_folder / pair).with_suffix(".hdf") for pair in pairs]
    if all(file.is_file() for file in hdf_files):
        return {
            pair: caching.read_hdfs(file)["df"]
            for pair, file in zip(pairs, hdf_files, strict=True)
        }
    if check_cache_only:
        return {}

    # convert to wav/TextGrid
    delete = _convert_for_mfa(transcripts, audios)

    # run MFA
    command = (
        f"mfa_align {folder} {dictionary} {model} {output_folder} --quiet " " ".join(
            f"--{key} {argument}" for key, argument in options.items()
        )
    )
    returncode, _, error = generic.run(command, stdout=None)

    # warning about unknown words
    unknown_file = output_folder / "oovs_found.txt"
    if unknown_file.is_file():
        unknown_words = unknown_file.read_text().split()
        if unknown_words:
            warnings.warn(
                f"Unknown words: {', '.join(unknown_words)}",
                RuntimeWarning,
                stacklevel=2,
            )
        unknown_file.unlink()

    # delete temporary files
    for file in delete:
        file.unlink()
    (output_folder / "utterance_oovs.txt").unlink(missing_ok=True)
    if options["temp_directory"].is_dir():
        shutil.rmtree(options["temp_directory"])

    # handle errors
    if returncode != 0:
        raise RuntimeError(f'"{command}" exited with {returncode}:\n{error}')

    # convert to ELAN and HDF
    results = {}
    for pair in pairs:
        textgrid = (output_folder / pair).with_suffix(".TextGrid")
        assert textgrid.is_file()
        eaf = (output_folder / pair).with_suffix(".eaf")
        hdf = (output_folder / pair).with_suffix(".hdf")
        pympi.TextGrid(textgrid).to_eaf().to_file(eaf)
        if delete_textgrid:
            textgrid.unlink()
        results[pair] = time_series.elan_to_interval_dataframe(eaf)
        # try to insert unknown words
        eaf_transcript = folder / eaf.name
        if eaf_transcript.is_file():
            results[pair] = _fill_in_unknown_words_mfa(results[pair], eaf_transcript)
        caching.write_hdfs(hdf, {"df": results[pair]})
    return results


def _convert_for_mfa(transcripts: list[Path], audios: list[Path]) -> list[Path]:
    delete = []
    for transcript, audio in zip(transcripts, audios, strict=True):
        # convert to TextGrid
        if transcript.suffix == ".eaf":
            textgrid = audio.parent / f"{audio.stem}.TextGrid"
            pympi.Eaf(transcript).to_textgrid().to_file(textgrid)
            delete.append(textgrid)
        # convert to wav file
        if audio.suffix != ".wav":
            wav = audio.parent / f"{audio.stem}.wav"
            cmd = (
                f"ffmpeg -i {audio} -ar 16000 -c:a pcm_s16le -channel_layout mono {wav}"
            )
            assert generic.run(cmd, stdout=None, stderr=None)[0] == 0
            delete.append(wav)
    return delete


def _fill_in_unknown_words_mfa(mfa: pd.DataFrame, eaf: Path) -> pd.DataFrame:
    mfa = mfa.sort_index()
    transcript = time_series.elan_to_interval_dataframe(eaf)

    for tier in transcript.columns:
        mfa_name = f"{tier} - words"
        if mfa_name not in mfa.columns:
            continue
        unknown_index = mfa[mfa_name] == "<unk>"
        unknowns = mfa.loc[unknown_index]
        mfa_tier = mfa.loc[mfa[mfa_name] != ""].sort_index()
        transcript_tier = transcript.loc[transcript[tier] != ""].sort_index()
        count = 0
        for time in unknowns.index:
            # find segment in ELAN
            match = transcript_tier.loc[transcript_tier.index.overlaps(time)]
            if match.shape[0] != 1:
                continue  # couldn't find an exact match
            match = match.iloc[0]

            # find matching MFA context
            mfa_context = mfa_tier.loc[match.name.left : match.name.right]
            eaf_context = [
                re.sub(r"[^-\w']+$", "", re.sub(r"^[^-\w']+", "", xx))
                for x in match[tier].lower().split()
                for xx in x.split("-")
            ]
            eaf_context = [x for x in eaf_context if x not in ["", "-", "'"]]
            if len(eaf_context) != mfa_context.shape[0]:
                continue  # there is some mismatch

            # verify that the eaf and mfa context are the same
            match = True
            for eaf_word, mfa_word in zip(
                eaf_context,
                mfa_context[mfa_name].tolist(),
                strict=True,
            ):
                if mfa_word == "<unk>":
                    continue
                mfa_word = re.sub(
                    r"[^-\w']+$",
                    "",
                    re.sub(r"^[^-\w']+", "", mfa_word),
                ).strip("'")
                match &= eaf_word.strip("'") == mfa_word
                if not match:
                    break

            if not match:
                continue  # something is different

            # insert word from eaf in mfa
            for iword, (eaf_word, mfa_word) in enumerate(
                zip(eaf_context, mfa_context[mfa_name].tolist(), strict=True),
            ):
                if mfa_word != "<unk>":
                    continue
                index = unknown_index & (mfa.index == mfa_context.index[iword])
                assert index.sum() == 1, index.sum()
                mfa.loc[index, mfa_name] = eaf_word
                count += 1

    return mfa


@caching.cache_wrapper
def run_openface(
    video_path: Path,
    *,
    cache: Path,
    options: str = "-q -2Dfp -3Dfp -pose -aus -gaze -multi-view 1 -wild",
) -> pd.DataFrame:
    """Extract OpenFace features.

    Args:
        video_path: The path to the video or folder containing an image sequence.
        options: Options for OpenFace.
        cache: Name of the cache file.

    Returns:
        A Pandas DataFrame containing the OpenFace features.
    """
    # prepare command
    video_path = generic.fullpath(video_path)
    output_folder = cache.parent
    cache_name = generic.basename(cache)
    arg = "-f"
    if video_path.is_dir():
        arg = "-fdir"
    command = (
        f"../OpenFace/exe/FeatureExtraction/FeatureExtraction {arg} {video_path} {options} -out_dir "
        f"{output_folder} -of {cache_name}"
    )

    # run OpenFace
    returncode, _, error = generic.run(command, stdout=None)
    detail_file = output_folder / f"{cache_name}_of_details.txt"
    detail_file.unlink(missing_ok=True)

    cache = output_folder / f"{cache_name}.csv"
    if returncode != 0:
        cache.unlink(missing_ok=True)
        raise RuntimeError(f'"{command}" exited with {returncode}:\n{error}')

    # read features
    result = pd.read_csv(
        cache,
        skipinitialspace=True,
        na_values=["-nan(ind)"],
        index_col="timestamp",
    )
    cache.unlink(missing_ok=True)

    # add HOG features
    cache = output_folder / f"{cache_name}.hog"
    if cache.is_file():
        result = _add_hog(result, cache)

    # fix frame numbering (for sequence of images)
    if video_path.is_dir():
        return _use_image_numbers(result, video_path)

    return result


def _add_hog(result: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Get HOG features.

    Use MATLAB to decode the binary HOG features and append them to the
    dataframe.
    """
    # call MATLAB
    tmp = cache.parent / f"{cache.stem}_hog.csv"
    command = (
        f"[hog, valid] = Read_HOG_file('{cache}'); "
        "hog = array2table(hog); hog.is_valid = valid; "
        f"writetable(hog, '{tmp}'); exit(0)"
    )
    name = Path(__file__).resolve()
    with generic.ChangeDir(name.parent):
        assert (
            _run_matlab(command, stdout=None, stderr=None)[0] == 0
        ), f"Failed to load the HOG features {cache}"

    # read CSV
    hog = pd.read_csv(tmp)
    assert hog.shape[0] == result.shape[0], f"{hog.shape[0]} {result.shape[0]}"
    result = pd.concat((result.reset_index(), hog), axis=1).set_index("timestamp")
    cache.unlink(missing_ok=True)
    tmp.unlink(missing_ok=True)

    return result


def _use_image_numbers(result: pd.DataFrame, video_path: Path) -> pd.DataFrame:
    """Use the image file names as frame numbers."""
    filetypes = ("jpg", "jpeg", "png", "bmp")
    paths: list[Path] = []
    for filetype in filetypes:
        paths.extend(video_path.glob(f"*.{filetype}"))
    if len(paths) == result.shape[0]:
        names = [generic.basename(path) for path in sorted(paths)]
        if all(name.isdigit() for name in names):
            result["frame"] = [int(name) for name in names]
        else:
            result["frame"] = names

    return result


def extract_features(
    caches: dict[str, Path],
    audio: Path | None = None,
    video: Path | None = None,
    audio2: Path | None = None,
    kwargs: dict[str, Any] | None = None,
    check_cache_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run several feature extraction programs.

    Args:
        audio: Path to an audio file. If multi-channel files are listed, only the
            first channels will be analyzed.
        audio2: Optional second audio file, used by some modules.
        video: Path to a video file.
        caches: Dict functions as a look-up table for the caches and to determine
            which features to extract.
        check_cache_only: If cache exists for all features, load and return it.
            Otherwise return an empty dict.
        kwargs: Dict with the same keys as caches. Each item is a dictionary which is
            forwarded as keyword-arguments to the function.

    Note:
        The following valid keys for caches/kwargs use internally these functions:
        'opensmile_<config name>', and 'opensmile_file_<path>' uses audio.opensmile.run
        'openface' uses features.run_openface
        'covarep' uses features.run_covarep
        'volume' uses audio.audio_analysis.calculate_volume_file
        'tdoa' uses gcc.sliding_shifts_file
        'diarization_tdoa' uses audio.speaker_diarization.tdoa_diarization_file
        'diarization_volume' uses audio.speaker_diarization.volume_diarization_file
        'speech_rate' uses audio.audio_analysis.sliding_speaking_rate_file
        'mfa' uses features.run_mfa

    Returns:
        A dictionary with fields corresponding to the extracted features. Each
        item is a pandas dataframe, which has time (in seconds) has the index.
    """
    audio = audio or Path()
    audio2 = audio2 or Path()
    video = video or Path()
    assert caches is not None
    data: dict[str, pd.DataFrame] = {}

    # input validation
    audio, audio2, video, kwargs = _extract_features_validate_input(
        audio,
        audio2,
        video,
        kwargs,
        caches,
        check_cache_only,
    )
    assert kwargs is not None

    # 'all' features except opensmile
    datum: pd.DataFrame | dict[str, pd.DataFrame] | None
    for key in caches:
        match key:
            case "openface":
                datum = run_openface(video, cache=caches[key], **kwargs[key])
            case "covarep":
                datum = run_covarep(audio, cache=caches[key], **kwargs[key])
            case "volume":
                datum = audio_analysis.calculate_volume_file(
                    audio,
                    cache=caches[key],
                    **kwargs[key],
                )
            case "tdoa":
                datum = gcc.sliding_shifts_file(
                    audio,
                    audio2,
                    cache=caches[key],
                    **kwargs[key],
                )
            case "diarization_tdoa":
                datum = speaker_diarization.tdoa_diarization_file(
                    (audio, audio2),
                    cache=caches[key],
                    **kwargs[key],
                )
            case "diarization_volume":
                datum = speaker_diarization.volume_diarization_file(
                    (audio, audio2),
                    cache=caches[key],
                    **kwargs[key],
                )
            case "speech_rate":
                datum = audio_analysis.sliding_speaking_rate_file(
                    (audio, audio2),
                    cache=caches[key],
                    **kwargs[key],
                )
            case "mfa":
                datum = run_mfa(output_folder=caches[key], **kwargs[key])
            case _:
                continue

        if check_cache_only and (datum is None or len(datum) == 0):
            return {}
        if isinstance(datum, dict):
            for subkey, datum_ in datum.items():
                data[f"{key}_{subkey}"] = datum_
        elif isinstance(datum, pd.DataFrame):
            data[key] = datum

    # opensmile
    return _extract_opensmile(data, audio, caches, kwargs, check_cache_only)


def _extract_features_validate_input(
    audio: Path,
    audio2: Path,
    video: Path,
    kwargs: dict[str, Any] | None,
    caches: dict[str, Path],
    check_cache_only: bool,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    # set options
    kwargs = kwargs or {}
    for key in caches:
        kwargs.setdefault(key, {})
        kwargs[key]["check_cache_only"] = check_cache_only
    if "opensmile_vad_opensource" in kwargs:
        kwargs["opensmile_vad_opensource"].setdefault(
            "cache_options",
            {"names": ("time", "vad")},
        )
        kwargs["opensmile_vad_opensource"].setdefault("time", "time")

    return (
        generic.fullpath(audio),
        generic.fullpath(audio2),
        generic.fullpath(video),
        kwargs,
    )


def _extract_opensmile(
    data: dict[str, pd.DataFrame],
    audio: Path,
    caches: dict[str, Path],
    kwargs: dict[str, Any],
    check_cache_only: bool,
) -> dict[str, pd.DataFrame]:
    """Run and adds opensmile to the dictionary."""
    for key, cache in caches.items():
        if not key.startswith("opensmile_"):
            continue
        # find config file
        config, short_key = _find_opensmile_config(key)
        config = generic.fullpath(config)
        cache = generic.fullpath(cache)

        with generic.ChangeDir(config.parent):
            tmp = opensmile.run(
                audio_dict={"I": audio},
                C=config,
                cache=cache,
                **kwargs[key],
            )
        if check_cache_only and tmp is None:
            return {}
        assert tmp is not None

        # post-processing
        data.update(
            {f"{short_key}_{output_key}": datum for output_key, datum in tmp.items()},
        )

    return data


def _find_opensmile_config(key: str) -> tuple[Path, str]:
    if key.startswith("opensmile_file_"):
        config = Path(key.split("opensmile_file_", 1)[1])
        short_key = f"opensmile_{generic.basename(config)}"
    else:
        short_key = key.split("opensmile_", 1)[1]
        configs = opensmile.list_configs(
            folders=(
                Path("/usr/local/share/opensmile/"),
                Path("~/local/share/opensmile/"),
            ),
        )
        configs = [
            config for config in configs if short_key == generic.basename(config)
        ]
        assert configs, f"Found no config file containing {short_key}!"
        config = configs[0]
        short_key = key
    return config, short_key


def synchronize_sequences(
    data: dict[str, pd.DataFrame],
    *,
    time: np.ndarray | None = None,
    rate: float = -1.0,
    intersect: bool = True,
) -> pd.DataFrame:
    """Synchronize several sequences (dataframes) to a given time.

    Note:
        By default it is synchronized to the highest sampling rate,
        the latest begin, and the earliest end.

        A nearest-neighbor interpolation is used for synchronization. This
        function is not recommended to downsample s signal (aliasing)
        unless low-pass filter is applied before).

    Args:
        data: A dictionary containing DataFrames. Keys will be used to prefix
            columns in the merged  DataFrame.
        time: Time vector of the synchronized dataframe.
        rate: Samples per seconds. Ignored if 'time' is provided.
        intersect: Whether to use the time interval covered by all sequences.
            If False, the maximum time interval will be used and the first/last
            value will be used for padding. Ignored when 'time' is provided.

    Returns:
        A DataFrame.
    """
    # Determine begin/end/rate
    if time is None:
        time = _determine_time(data, rate, intersect)

    # Interpolate
    result = []
    for key, datum in data.items():
        assert datum.index.is_monotonic_increasing, key

        method: Literal["nearest"] | None = "nearest"
        if isinstance(datum.index, pd.IntervalIndex):
            method = None
            datum = datum.copy()
            datum["interval_number"] = range(datum.shape[0])

            if isinstance(datum.index, pd.IntervalIndex) and datum.index.is_overlapping:
                # adjust intervals if they overlap
                left = datum.index.left
                right = datum.index.right
                overlap = np.nonzero((left[1:] - right[:-1]) < 0)[0]
                new_border = (left[overlap] + right[overlap + 1]) / 2
                new_left = left.to_numpy()
                new_right = right.to_numpy()
                new_left[overlap + 1] = new_border
                new_right[overlap] = new_border
                datum.index = pd.IntervalIndex.from_arrays(
                    new_left,
                    new_right,
                    closed=datum.index.closed,
                )

        result.append(datum.reindex(time, method=method).add_prefix(f"{key}_"))

    return pd.concat(result, axis=1)


def _determine_time(
    data: dict[str, pd.DataFrame],
    rate: float,
    intersect: bool,
) -> np.ndarray:
    """Determine begin/end/rate."""
    # get begin/end/step
    begins: list[float] = []
    ends: list[float] = []
    steps: list[float] = []
    for datum in data.values():
        index = datum.index
        if isinstance(index, pd.IntervalIndex):
            begins.append(index[0].left)
            ends.append(index[-1].right)
            steps.append((index.right - index.left).array.median())
        else:
            begins.append(index[0])
            ends.append(index[-1])
            steps.append((index[1:] - index[:-1]).array.median())

    # find time
    begin, end = min(begins), max(ends)
    if intersect:
        begin, end = max(begins), min(ends)

    step = min(steps)
    if rate > 0:
        step = 1.0 / rate

    time = np.arange(begin, end + step, step)
    return time[(time >= begin) & (time <= end)]


def take_statistics(
    data: pd.DataFrame,
    statistics: dict[str, list[str]],
) -> pd.DataFrame:
    """Calculate the statistics.

    Args:
        data: DataFrame to calculate the statistics on.
        statistics: A dictionary describing which statistics to take.

    Note:
        Sets IQR and std to 0 if they would be NaN.

    Returns:
        A DataFrame of size 1 with the statistics.
    """
    result = []
    for key in statistics:
        columns = [column for column in statistics[key] if column in data.columns]
        if not columns:
            continue
        values = (
            data[columns].quantile([0.25, 0.75]).diff().loc[0.75, :]
            if key == "iqr"
            else getattr(data.loc[:, columns], key)()
        )
        if key in ("std", "iqr", "var"):
            values = values.fillna(0)
        result.append(values.add_suffix(f"_{key}"))
    if not result:
        return pd.DataFrame()
    return pd.concat(result).to_frame().T


def aggregate_to_intervals(
    data: pd.DataFrame,
    intervals: np.ndarray,
    statistics: dict[str, list[str]],
) -> pd.DataFrame:
    """Aggregate statistics from regular sampled sequences for different intervals.

    Args:
        data: DataFrame.
        intervals: Array with two columns, begin and end.
        statistics: Dictionary of functions (key) and columns (value) to apply on.

    Returns:
        DataFrame with only columns generated by 'statistics' and columns
        'begin' and 'end'.
    """
    rows = []
    for begin, end in intervals:
        rows.append(take_statistics(data.loc[begin:end, :], statistics))
    data = pd.concat(rows, ignore_index=True)
    return data.assign(begin=intervals[:, 0], end=intervals[:, 1])


def apply_masking(
    data: pd.DataFrame,
    *,
    diarization: np.ndarray | None = None,
) -> pd.DataFrame:
    """Apply some common masking/filtering operations.

    Args:
        data: DataFrame to apply the masking on.
        diarization: Optional binary array to mask all acoustic features.

    Returns:
        The same DataFrame with NaNs at the masked places.
    """
    step_size = (data.index[1:] - data.index[:-1]).array.median().item()

    # define indices and columns
    indices = []
    columns = []

    # diarization
    if diarization is not None:
        indices.append(diarization < 0.5)
        columns.append(
            [
                name
                for name in data.columns
                if name.startswith(
                    ("covarep_", "opensmile_", "volume_", "speech_rate_"),
                )
            ],
        )

    # covarep
    if "covarep_VUV" in data.columns:
        columns.append(
            [
                name
                for name in data.columns
                if name.startswith("covarep_")
                and name not in ("covarep_VUV", "covarep_VAD", "covarep_vowelSpace")
            ],
        )
        if "volume_volume" in data.columns:
            columns[-1].append("volume_volume")
        data["covarep_VUV"] = medfilt(
            data["covarep_VUV"],
            generic.round_up_to_odd(0.1 / step_size),
        )
        indices.append(data["covarep_VUV"] < 0.5)

    # openface/afar
    for au_extractor in ("openface", "afar"):
        if f"{au_extractor}_success" in data.columns:
            columns.append(
                [
                    name
                    for name in data.columns
                    if name.startswith(f"{au_extractor}_")
                    and name
                    not in (f"{au_extractor}_success", f"{au_extractor}_confidence")
                ],
            )
            indices.append(data[f"{au_extractor}_success"] != 1)

    # opensmile
    if "opensmile_vad_opensource_csvoutput_vad" in data.columns:
        columns.append(
            [
                name
                for name in data.columns
                if name.startswith("opensmile_")
                and name
                not in (
                    "opensmile_vad_opensource_csvoutput_vad",
                    "opensmile_prosodyAcf_voiceProb_sma",
                )
            ],
        )
        indices.append(data["opensmile_vad_opensource_csvoutput_vad"] < 0.5)

    # apply masking
    for index, columns_ in zip(indices, columns, strict=True):
        data.loc[index, columns_] = float("NaN")

    return data


def suggest_statistics(columns: list[str]) -> dict[str, list[str]]:
    """Suggest statistics (mean/std/median/iqr) for each column name.

    Args:
        columns: List of column names, name should follow the
            synchronize_sequences(extract_features) format.

    Returns:
        A dictionary where the keys are the statistic names and the values are the
        list of features.
    """
    statistics: dict[str, list[str]] = {
        "mean": [],
        "std": [],
        "median": [],
        "iqr": [],
        "min": [],
        "max": [],
    }

    # openface (AUs, head pose)
    openface = [
        "openface_confidence",
        "openface_gaze_angle_x",
        "openface_gaze_angle_y",
        "openface_gaze_angle_xy",
        "openface_gaze_angle_x_abs",
        "openface_pose_Rx",
        "openface_pose_Ry",
        "openface_pose_Rx_abs",
        "openface_pose_Txyz_delta_abs",
        "openface_pose_Rxy_delta_abs",
        "openface_gaze_angle_xy_delta",
        "openface_gaze_angle_xy_delta_abs",
    ]
    tmp = [name for name in columns if name.startswith("openface_AU")]
    statistics["max"].extend(tmp)
    tmp = [name for name in columns if any(map(name.startswith, openface))] + tmp
    statistics["mean"].extend(tmp)
    statistics["std"].extend(tmp)

    # afar
    tmp = [name for name in columns if name.startswith("afar_AU")]
    statistics["max"].extend(tmp)
    statistics["mean"].extend(tmp)
    statistics["std"].extend(tmp)

    # eGeMAPS - functionals (median)
    tmp = [
        name
        for name in columns
        if name.startswith("opensmile_eGeMAPS") and "_csvoutput" in name
    ]
    statistics["median"].extend(tmp)
    # eGeMAPS - low-level (median+iqr)
    tmp = [
        name
        for name in columns
        if name.startswith("opensmile_eGeMAPS") and "lldcsvoutput" in name
    ]
    statistics["median"].extend(tmp)
    statistics["iqr"].extend(tmp)

    # other opensmile
    tmp = [
        name
        for name in columns
        if name.startswith("opensmile_") and not name.startswith("opensmile_eGeMAPS")
    ]
    statistics["median"].extend(tmp)
    statistics["iqr"].extend(tmp)

    # volume
    volume = [name for name in columns if name.startswith("volume_volume")]
    statistics["median"].extend(volume)
    statistics["iqr"].extend(volume)

    # covarep
    tmp_mean = ["covarep_vowelSpace", "covarep_MCEP_0", "covarep_MCEP_1", "covarep_VAD"]
    tmp = [
        "covarep_f0",
        "covarep_NAQ",
        "covarep_QOQ",
        "covarep_MDQ",
        "covarep_peakSlope",
        "covarep_F1",
        "covarep_F2",
    ]
    statistics["median"].extend(
        [name for name in columns if any(map(name.startswith, tmp_mean + tmp))],
    )
    statistics["iqr"].extend(
        [name for name in columns if any(map(name.startswith, tmp))],
    )

    # mean of unknown features
    known = [
        "openface",
        "opensmile_eGeMAPS",
        "opensmile_vad_opensource",
        "opensmile_prosodyAcf",
        "volume",
        "covarep",
        "afar",
    ]
    unknown = [name for name in columns if not any(map(name.startswith, known))]
    statistics["mean"].extend(unknown)

    # sanity check
    assert all(len(features) == len(set(features)) for features in statistics.values())

    return statistics


def rolling_statistics(
    data: pd.DataFrame,
    *,
    statistics: dict[str, list[str]],
    window: int = 10,
) -> pd.DataFrame:
    """Apply rolling mean/std/median/iqr statistics.

    Args:
        data: The dataframe.
        statistics: Dictionary describing the statistics.
        window: Size of the window for the rolling statistics.

    Returns:
        A dataframe.
    """
    result = []
    for statistic, columns in statistics.items():
        columns = [name for name in columns if name in data.columns]
        if not columns:
            continue
        tmp = data.loc[:, columns].rolling(window, min_periods=window // 4)
        tmp = (
            tmp.quantile(0.75) - tmp.quantile(0.25)
            if statistic == "iqr"
            else getattr(tmp, statistic)()
        )
        if statistic in ("std", "iqr", "var"):
            tmp = tmp.fillna(0)
        result.append(tmp.add_suffix(f"_{statistic}"))
    return pd.concat(result, axis=1)


def lookup_with_fallback(
    series: pd.Series,
    data: pd.DataFrame,
    clean_fun: Callable[[T], T] | None,
) -> pd.DataFrame:
    """Look all values in a series in a DataFrame up.

    Note:
        Apply 'clean_fun' to series values if when no index matches.

    Args:
        series: Contains keys to look up .
        data: Where to look keys up.
        clean_fun: function to generate a fallback.

    Returns:
        Look ups.
    """
    # try to lookup all entries
    result = data.reindex(series.array)
    map_series_index_to_reindex = pd.Series(data=result.index.array, index=series.index)
    result.index = series.index

    # handle not found entries
    if clean_fun is not None:
        indices = result.index[result.isna().all(axis=1)]
        cleaned_map = map(clean_fun, map_series_index_to_reindex.loc[indices])
        zipped = [
            [clean, index]
            for clean, index in zip(cleaned_map, indices, strict=True)
            if clean != series.loc[index]
        ]
        if zipped:
            cleaned, indices = list(zip(*zipped, strict=True))
            result.loc[list(indices)] = data.reindex(cleaned).to_numpy()
    return result
