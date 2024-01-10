#!/usr/bin/env python3
"""A collection of bootstrapping functions."""
from functools import partial
from typing import cast, overload

import numpy as np

from python_tools.ml.metrics import scores_as_matrix
from python_tools.typing import MetricGenericFun


def bootstrap(
    y_true: np.ndarray,
    y1_hat: np.ndarray,
    *,
    y2_hat: np.ndarray | None = None,
    metric_fun: MetricGenericFun,
    clusters: np.ndarray | None = None,
    ci_interval: float = 0.95,
    resamples: int = 10_000,
    y1_hat_kwargs: dict[str, np.ndarray] | None = None,
    y2_hat_kwargs: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Find percentile-based bootstrap intervals.

    Note:
        All array-like objects have to have the same size along the first dimension.

        Clustered bootstrapping is implemented as in
        "Nonparametric bootstrapping for hierarchical data"

    Args:
        y_true: The ground truth.
        y1_hat: The prediction of the ground truth.
        y2_hat: The optional second prediction of the ground truth.
        metric_fun: A function which is given parts of y_true, y_hat, y_hat_kwargs.
            The function has to return a dictionary of scalars. Confidence intervals
            are calculated for all scalars.
        clusters: Same shape as y_true, indicates which observations belong together
            (paired clusters). To support a paired metrics, the caller has to provide
            the difference in y1_hat and an appropriate metric_fun.
        ci_interval: Which confidence interval to use.
        resamples:  How many times to resample.
        y1_hat_kwargs:  Dictionary of array-like objects. They are passed to the metric
            fun. y1_hat and y2_hat have to have the same keyword arguments. Meant to
            have prediction probabilities or sample weights.
        y2_hat_kwargs: Same as y1_hat_kwargs.

    Returns:
        A dictionary with these fields '1', '2' (if y2_hat is provided), and
        '1-2' (if y2_hat is provided). All values are dictionaries with the keys
        of the metric_fun. There values are dictionaries with scalars as
        values. The keys are these 'lower', 'center', and 'upper'.

    """
    # be reproducible
    rng = np.random.default_rng(1)

    # validate input: convert all inputs to matrix form: n x ?
    y_true, y1_hat = scores_as_matrix(  # pylint: disable=unbalanced-tuple-unpacking
        y_true,
        y1_hat,
    )
    if y2_hat is not None:
        y2_hat = scores_as_matrix(y2_hat)[0]
    if clusters is not None:
        clusters = scores_as_matrix(clusters)[0]

    assert metric_fun is not None
    y1_hat_kwargs = y1_hat_kwargs or {}
    y2_hat_kwargs = y2_hat_kwargs or {}
    for key in y1_hat_kwargs:
        (  # pylint: disable=unbalanced-tuple-unpacking
            y1_hat_kwargs[key],
            y2_hat_kwargs[key],
        ) = scores_as_matrix(y1_hat_kwargs[key], y2_hat_kwargs[key])

    # cluster data
    if clusters is not None:
        (
            y1_hat,
            y2_hat,
            y_true,
            y1_hat_kwargs,
            y2_hat_kwargs,
            metric_fun,
        ) = _clustered_bootstrapping(
            y1_hat,
            y2_hat,
            y_true,
            y1_hat_kwargs,
            y2_hat_kwargs,
            metric_fun,
            clusters,
        )

    # sampling
    scores1: dict[str, np.ndarray] = {}
    scores2: dict[str, np.ndarray] = {}
    for i in range(resamples):
        # resample
        index = rng.choice(y_true.shape[0], size=y_true.shape[0], replace=True)

        # determine scores for y1
        scores1 = _add_metrics(
            scores1,
            y_true,
            y1_hat,
            index,
            y1_hat_kwargs,
            resamples,
            i,
            metric_fun,
        )
        # determine scores for y2
        if y2_hat is None:
            continue
        scores2 = _add_metrics(
            scores2,
            y_true,
            y2_hat,
            index,
            y2_hat_kwargs,
            resamples,
            i,
            metric_fun,
        )

    # prepare output
    result = {"1": _get_percentiles(scores1, ci_interval)}
    if y2_hat is not None:
        result["2"] = _get_percentiles(scores2, ci_interval)
        result["1-2"] = _get_percentiles(
            {key: value - scores2[key] for key, value in scores1.items()},
            ci_interval,
        )
    return result


def _clustered_bootstrapping(
    y1_hat: np.ndarray,
    y2_hat: np.ndarray | None,
    y_true: np.ndarray,
    y1_hat_kwargs: dict[str, np.ndarray],
    y2_hat_kwargs: dict[str, np.ndarray],
    metric_fun: MetricGenericFun,
    clusters: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    MetricGenericFun,
]:
    """Calculate metrics for each cluster."""
    cluster_values = np.unique(clusters)

    # aggregate clusters
    scores1: dict[str, np.ndarray] = {}
    scores2: dict[str, np.ndarray] = {}
    for icluster, cluster in enumerate(cluster_values):
        index = clusters == cluster
        scores1 = _add_metrics(
            scores1,
            y_true,
            y1_hat,
            index,
            y1_hat_kwargs,
            cluster_values.size,
            icluster,
            metric_fun,
        )
        if y2_hat is None:
            continue
        scores2 = _add_metrics(
            scores2,
            y_true,
            y2_hat,
            index,
            y2_hat_kwargs,
            cluster_values.size,
            icluster,
            metric_fun,
        )

    # redefine metric function and scores
    names = tuple(scores1)
    y1_hat = _flatten_values(scores1)
    y2_hat = _flatten_values(scores2)
    y1_hat_kwargs = {}
    y2_hat_kwargs = {}
    y_true = y1_hat
    metric_fun = partial(  # pyright: ignore[reportGeneralTypeIssues]
        _clustered_metric_fun,
        names=names,
    )

    return y1_hat, y2_hat, y_true, y1_hat_kwargs, y2_hat_kwargs, metric_fun


@overload
def _flatten_values(scores: dict[str, np.ndarray]) -> np.ndarray:
    ...


@overload
def _flatten_values(scores: None) -> None:
    ...


def _flatten_values(scores: dict[str, np.ndarray] | None) -> np.ndarray | None:
    if not scores:
        return None
    return np.concatenate([value.reshape(-1, 1) for value in scores.values()], axis=1)


def _clustered_metric_fun(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    *,
    names: tuple[str, ...] = (),
    **kwargs: np.ndarray,
) -> dict[str, float]:
    """Only need to return the mean of the previously calculated metrics."""
    return {
        name: mean
        for mean, name in zip(
            cast(np.ndarray, np.nanmean(y_hat, axis=0)),
            names,
            strict=True,
        )
    }


def _add_metrics(
    scores: dict[str, np.ndarray],
    y_true: np.ndarray,
    y_hat: np.ndarray,
    index: np.ndarray,
    kwargs: dict[str, np.ndarray],
    n_samples: int,
    ith_sample: int,
    metric_fun: MetricGenericFun,
) -> dict[str, np.ndarray]:
    """Accumulate metrics."""
    index = index.reshape(-1)
    scores_ = metric_fun(
        y_true[index, :],
        y_hat[index, :],
        **{key: value[index, :] for key, value in kwargs.items()},
    )
    for key, value in scores_.items():
        if key not in scores:
            scores[key] = np.full(n_samples, float("NaN"))
        scores[key][ith_sample] = value
    return scores


def _get_percentiles(
    scores: dict[str, np.ndarray],
    ci_interval: float,
) -> dict[str, dict[str, float]]:
    """Calculate PCA confidence intervals."""
    result = {}
    quantiles = [(100 - 100 * ci_interval) / 2, 50, (100 + 100 * ci_interval) / 2]
    for key, value in scores.items():
        percentiles = np.percentile(value, quantiles)
        result[key] = {
            "lower": percentiles[0],
            "center": percentiles[1],
            "upper": percentiles[2],
        }
    return result
