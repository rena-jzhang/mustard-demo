#!/usr/bin/env python3
"""A function to extract_behavioral markers."""
import math

import numpy as np
import pandas as pd

from python_tools import features
from python_tools.audio.audio_analysis import intra_inter_pauses
from python_tools.time_series import index_to_segments


def get_markers(
    data: pd.DataFrame,
    *,
    prefer_parametric: bool = False,
    thresholds: dict[str, float] | None = None,
    durations: dict[str, float] | None = None,
    avoid_nans: bool = True,
) -> dict[str, float]:
    """Calculate various behavior markers.

    Args:
        data: DataFrame with all the features.
        prefer_parametric: Whether to use mean/std or median/iqr for acoustic features.
        thresholds: Dictionary of feature names and their cutoff, used to extract
            intervals when the feature is active.
        durations: Dictionary of feature names which have intervals. Keep only
            intervals which have the specified duration or are longer (duration in
            seconds).
        avoid_nans: Try to replace NaNs with feature-specific empty values.

    Citations for the returned markers, citations with '*' measure a similar
    behavior in a different manner and/or different context:
        Suicidal Ideation:
            covarep_MDQ_mean [3]
            covarep_NAQ_mean [3, 16]
            covarep_QOQ_mean [3, 16]
            covarep_peakSlope_mean [3, 16]
            covarep_NAQ_std [3, 16]
            covarep_QOQ_std [3, 16]
            covarep_F1_mean [3]
            openface_AU12_r/c_mean [24]
            openface_AU17_r/c_mean [24]
            openface_AU12_r/c_std [24]
            openface_AU12_r/c_threshold_filtered_count [24]
            openface_AU17_r/c_threshold_filtered_count [24]
            duchenne_smile_ratio [24]
        Psychological Distress, Depression, PTSD:
            covarep_QOQ_mean [17, 19, 22]
            covarep_NAQ_mean [17, 19]
            covarep_peakSlope_mean [17, 22]
            covarep_NAQ_std [19]
            covarep_QOQ_std [19]
            covarep_H1H2_mean [12]
            covarep_vowelSpace_mean [8]
            covarep_VUV_threshold_mean [12]
            covarep_VAD_threshold_mean [*12]
            covarep_VUV_threshold_duration_mean [*22]
            covarep_VAD_threshold_duration_mean [*22]
            diarization_tdoa_*_threshold_filtered_duration_mean [*22]
            openface_AU4_r/c_mean [20]
            openface_AU12_r/c_mean [*15]
            openface_AU12_r/c_threshold_filtered_mean [*15]
            volume_volume_mean [22]
            speech_rate_1sec_mean [*22]
            openface_pose_Rx_mean [15]
            openface_gaze_angle_y [15]
            openface_euclidean_norm_pose_Rxyz_std [*12, *15, *20]
            openface_euclidean_distance_pose_Rxyz_mean [25]
            openface_euclidean_distance_pose_Rxyz_diff_abs_mean [25]
            inter_pauses_*_mean [*18, *22, *26]
            inter_pauses_*_std [*18, *26]
            intra_pauses_*_mean [*18]
            intra_pauses_*_std [*18]
        Anxiety:
            covarep_MCEP_0_std [9]
            covarep_F1_std [9]
            covarep_VUV_threshold_mean [9]
            openface_gaze_angle_y [15]
        Psychotic Disorders, Schizophrenia:
            openface_AU4_r/c_mean [21]
            openface_AU12_r/c_mean [21]
            openface_AU12_r/c_std [21]
            covarep_MCEP_0_iqr [7]
            covarep_peakSlope_median [7]
            covarep_peakSlope_iqr [7]
            covarep_QOQ_median [7]
            covarep_F1_iqr [7]
            covarep_F2_iqr [7]
            covarep_vowelSpace_mean [7]
            speech_rate_1sec_iqr [*7]

    Papers:
    3   Adolescent Suicidal Risk Assessment in Clinician-Patient-Interaction
    7   Computational Analysis of Acoustic Descriptors in Psychotic Patients
    8   Reduced Vowel Space is a Robust Indicator of Psychological Distress A
        Cross-Corpus Analysis
    9   Automatic Assessment and Analysis of Public Speaking Anxiety A Virtual
        Audience Case Study
    12  A Multimodal Context-based Approach for Distress Assessment
    15  Automatic Behavior Descriptors for Psychological Disorder Analysis
    16  Investigating the Speech Characteristics of Suicidal Adolescents
    17  Investigating Voice Quality as a Speaker-Independent Indicator of
        Depression and PTSD
    18  Verbal Indicators of Psychological Distress in Interactive Dialogue with a
        Virtual Human
    19  Audiovisual Behavior Descriptors for Depression Assessment
    20  Automatic Nonverbal Behavior Indicators of Depression and PTSD
        Exploring Gender Differences
    21  Automatic prediction of psychosis symptoms from facial expressions
    22  Multimodal Prediction of Psychological Disorders Learning Verbal and
        Nonverbal Commonalities in Adjacency Pairs
    24  Investigating Facial Behavior Indicators of Suicidal Ideation
    25  Nonverbal social withdrawal in depression Evidence from manual and
        automatic analyses

    Returns:
        A dictionary with all the markers.
    """
    # initialize optional parameters
    thresholds = thresholds or {}
    thresholds_defaults = {
        "covarep_VAD": 0.5,
        "covarep_VUV": 0.5,
        "openface_AU12_r": 1.0,
        "openface_AU12_c": 0.5,
        "openface_AU17_r": 1.0,
        "openface_AU17_c": 0.5,
        "diarization_tdoa_0": 0.5,
        "diarization_tdoa_1": 0.5,
    }
    thresholds_defaults.update(thresholds)
    thresholds = {
        key: value for key, value in thresholds_defaults.items() if key in data.columns
    }
    durations = durations or {}
    durations_defaults = {
        "openface_AU12_r": 0.2,
        "openface_AU12_c": 0.2,
        "openface_AU17_r": 0.2,
        "openface_AU17_c": 0.2,
        "covarep_VAD": 0.3,
        "covarep_VUV": 0.3,
        "diarization_tdoa_0": 0.3,
        "diarization_tdoa_1": 0.3,
    }
    durations_defaults.update(durations)
    durations = {
        key: value for key, value in durations_defaults.items() if key in thresholds
    }

    # prepare markers based on transformed features
    data = _add_transformed_features(data, thresholds)

    # prepare markers based on intervals
    intervals = {}
    for key in thresholds:
        key = f"{key}_threshold"
        intervals[key] = index_to_segments(data[key].numpy(), data.index.to_numpy())

    # prepare markers based on filtering
    for key, duration in durations.items():
        key = f"{key}_threshold"
        index = np.diff(intervals[key], axis=1)[:, 0] > duration
        intervals[f"{key}_filtered"] = intervals[key][index, :]

    # simple markers: mean/median/std/iqr
    statistics = {
        "mean": [
            "openface_AU12_r",
            "openface_AU12_c",
            "openface_AU17_r",
            "openface_AU17_c",
            "openface_AU4_r",
            "openface_AU4_c",
            "covarep_VUV_threshold",
            "covarep_VAD_threshold",
            "openface_pose_Rx",
            "openface_gaze_angle_y",
            "openface_euclidean_distance_pose_Rxyz",
            "openface_euclidean_distance_pose_Rxyz_diff_abs",
        ],
        "std": [
            "openface_AU12_r",
            "openface_AU12_c",
            "openface_euclidean_norm_pose_Rxyz",
        ],
        "median": [
            "volume_volume",
            "covarep_MDQ",
            "covarep_NAQ",
            "covarep_QOQ",
            "covarep_peakSlope",
            "covarep_H1H2",
            "covarep_vowelSpace",
            "covarep_F1",
        ],
        "iqr": [
            "volume_volume",
            "covarep_NAQ",
            "covarep_QOQ",
            "covarep_peakSlope",
            "covarep_MCEP_0",
            "covarep_F1",
            "covarep_F2",
        ],
    }
    if prefer_parametric:
        statistics["mean"].extend(statistics["median"])
        statistics["std"].extend(statistics["iqr"])
        del statistics["median"]
        del statistics["iqr"]
    markers_df = features.take_statistics(data, statistics)
    markers = {} if markers_df.shape[0] == 0 else markers_df.loc[0, :].to_dict()

    # count markers: count number of intervals
    for key in (
        "openface_AU12_r_threshold_filtered",
        "openface_AU12_c_threshold_filtered",
        "openface_AU17_r_threshold_filtered",
        "openface_AU17_c_threshold_filtered",
    ):
        if key not in intervals:
            continue
        markers[f"{key}_count"] = intervals[key].shape[0] / (
            data.index[-1] - data.index[0]
        )

    # average interval duration markers
    for key in (
        "covarep_VUV_threshold_filtered",
        "covarep_VAD_threshold_filtered",
        "diarization_tdoa_0_threshold_filtered",
        "diarization_tdoa_1_threshold_filtered",
        "openface_AU12_r_threshold_filtered",
        "openface_AU12_c_threshold_filtered",
    ):
        if key not in intervals:
            pass
        elif intervals[key].shape[0] == 0:
            markers[f"{key}_duration_mean"] = 0.0
        else:
            markers[f"{key}_duration_mean"] = np.diff(intervals[key], axis=1).mean()

    # ratio of Duchenne smiles
    markers = _add_ratio_duchenne_smiles(data, intervals, markers)

    # inter/intra pauses (inter: latency, switch pauses)
    markers = _add_pauses(intervals, markers)

    # speech rate
    markers = _add_speech_rate(data, markers)

    # avoid NaNs
    if avoid_nans:
        # openface&count -> 0
        markers = {
            key: 0.0
            if (math.isnan(value) and ("openface" in key or "count" in key))
            else value
            for key, value in markers.items()
        }

    return markers


def _add_ratio_duchenne_smiles(
    data: pd.DataFrame,
    intervals: dict[str, np.ndarray],
    markers: dict[str, float],
) -> dict[str, float]:
    # ratio of Duchenne smiles
    key = "openface_AU12_r_threshold_filtered"
    if key not in intervals:
        return markers
    duchenne = 0.0
    for begin, end in intervals[key]:
        if data.loc[begin:end, "openface_AU06_r"].mean() > 1:
            duchenne += 1
    if intervals[key].shape[0] != 0:
        duchenne /= intervals[key].shape[0]
    markers["duchenne_smile_ratio"] = duchenne
    return markers


def _add_speech_rate(data: pd.DataFrame, markers: dict[str, float]) -> dict[str, float]:
    # speech rate
    if "speech_rate_duration" not in data.columns:
        return markers
    rate = data.loc[data["speech_rate_duration"] > 1, "speech_rate_rate"]
    markers["speech_rate_1sec_mean"] = rate.mean()
    markers["speech_rate_1sec_iqr"] = rate.quantile([0.25, 0.75]).diff()[0.75]
    return markers


def _add_transformed_features(
    data: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    # prepare markers based on thresholds
    tmp = data.loc[:, thresholds.keys()] > thresholds.values()
    tmp[data.loc[:, thresholds.keys()].isna()] = False
    tmp = tmp.add_suffix("_threshold")
    data = pd.concat([data, tmp], axis=1)

    # prepare markers based on Euclidean Norm/Distance
    for name, group in (
        (
            ("openface", "pose_Rxyz"),
            ("openface_pose_Rx", "openface_pose_Rx", "openface_pose_Rz"),
        ),
    ):
        if not all(column in data.columns for column in group):
            continue
        index = data.loc[:, group].isna().any(axis=1)
        # Norm
        name_ = f"{name[0]}_euclidean_norm_{name[1]}"
        data[name_] = data.loc[:, group].pow(2).sum(axis=1).pow(0.5)
        data.loc[index, name_] = float("NaN")
        # Distance
        name_ = f"{name[0]}_euclidean_distance_{name[1]}"
        data[name_] = data.loc[:, group].diff().pow(2).sum(axis=1).pow(0.5)
        data.loc[index, name_] = float("NaN")

    # prepare markers based on absolute difference
    name_ = "openface_euclidean_distance_pose_Rxyz"
    if name_ in data.columns:
        data[f"{name_}_diff_abs"] = data[name_].diff().abs()

    return data


def _add_pauses(
    intervals: dict[str, np.ndarray],
    markers: dict[str, float],
) -> dict[str, float]:
    # inter and intra pauses
    if "diarization_tdoa_0_threshold_filtered" not in intervals:
        return markers
    for i in range(2):
        for key in ("intra_pauses", "inter_pauses"):
            suffix = "_threshold_filtered"
            if key == "intra_pauses":
                suffix = "_threshold"
            pauses = intra_inter_pauses(
                intervals[f"diarization_tdoa_{i}{suffix}"],
                intervals[f"diarization_tdoa_{1-i}{suffix}"],
            )[key]
            markers[f"{key}_{i}_mean"] = pauses.mean().item()
            markers[f"{key}_{i}_std"] = pauses.std().item()
    return markers
