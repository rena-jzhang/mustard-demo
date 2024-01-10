#!/usr/bin/env python3
"""A collection of metrics."""
import math
from collections.abc import Callable
from typing import overload

import numpy as np
import numpy.typing as npt
from scipy.special import expit, softmax
from scipy.stats import pearsonr, wilcoxon

from python_tools import generic
from python_tools.typing import MetricFun, MetricWrappedFun


def _averaged_metric_per_cluster(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    metric_function: MetricFun,
    *,
    ids: np.ndarray,
    names: tuple[str, ...] = (),
    which: tuple[str, ...] = (),
    y_scores: np.ndarray | None = None,
) -> dict[str, float]:
    # calculate metrics for each person
    metric_list: list[dict[str, float]] = []
    ids = ids.reshape(-1)
    ratios = []
    for identifier in np.unique(ids):
        index = identifier == ids
        ratios.append(index.mean().item())
        y_scores_ = y_scores
        if y_scores is not None:
            y_scores_ = y_scores[index]
        metric_list.append(
            metric_function(
                y_true[index],
                y_hat[index],
                names=names,
                y_scores=y_scores_,
                which=which,
            ),
        )
    # and take the average
    return average_list_dict_values(metric_list, ratios=np.array(ratios))


def average_list_dict_values(
    metric_list: list[dict[str, float]],
    *,
    ratios: np.ndarray | None = None,
) -> dict[str, float]:
    """Average scores from multiple groups.

    Args:
        metric_list: Scores from multiple groups.
        ratios: If non-none, also calculate the weighted average.

    Returns:
        The averaged scores
    """
    metrics = {}
    for key in metric_list[0]:
        values = np.array([metric_dict[key] for metric_dict in metric_list])
        index = ~np.isnan(values)
        metrics[key] = np.mean(values[index]).item()
        if ratios is None:
            continue
        metrics[f"{key}_proportional"] = float("NaN")
        if not index.any():
            continue
        metrics[f"{key}_proportional"] = np.average(
            values[index],
            weights=ratios[index],
        ).item()
    return metrics


def scalar_dict(
    values: dict[str, list[float]],
    *,
    names: tuple[str, ...] = (),
) -> dict[str, float]:
    """Flatten a dictionary of non-scalars. Converts torch to numpy.

    Args:
        values: The dictionary.
        names: List of names to prefix scalars.

    Returns:
        The flattened dictionary.
    """
    result = {}
    for key, value in values.items():
        if len(value) == 1:
            result[key] = value[0]
        else:
            assert len(value) == len(names)
            for name, item in zip(names, value, strict=True):
                result[f"{key}_{name}"] = item
            result[key] = np.nanmean(value).item()
    return result


def scores_as_matrix(*args: np.ndarray) -> list[np.ndarray]:
    """Convert scores to matrix format.

    Args:
        *args: List of vectors/matrices.

    Returns:
        List of numpy matrices.
    """
    result = []
    for scores in args:
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        result.append(scores)
    return result


def concat_dicts(
    values: list[dict[str, np.ndarray]],
    *,
    keys: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Concatenate dicts of Numpy arrays."""
    result = {}
    for key in values[0]:
        if keys and key not in keys:
            continue
        result[key] = scores_as_matrix(
            np.concatenate([value[key] for value in values], axis=0),
        )[0]
    return result


def metric_wrapper(metric_function: MetricFun) -> MetricWrappedFun:
    """Pre- and post-processing for all metrics.

    In addition it allows to average metrics across ids (e.g., calculate
    metrics for each person separately and then average them).

    Pre-Processing:
        values have to be in matrix form
    Post-Processing:
        Dictionary should only contain scalar
    """

    def wrapper(
        y_true: np.ndarray,
        y_hat: np.ndarray,
        *,
        which: tuple[str, ...] = (),
        names: tuple[str, ...] = (),
        ids: np.ndarray | None = None,
        y_scores: np.ndarray | None = None,
        clustering: bool = False,
    ) -> dict[str, float]:
        # convert to matrices
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_hat = y_hat.reshape(y_true.shape[0], -1)
        if y_scores is not None:
            y_scores = y_scores.reshape(y_true.shape[0], -1)
        assert y_hat.shape[1] == y_true.shape[1], f"{y_true.shape} {y_hat.shape}"

        if ids is not None and clustering:
            # it is slow, only when requested
            metrics = _averaged_metric_per_cluster(
                y_true,
                y_hat,
                metric_function,
                names=names,
                ids=ids,
                y_scores=y_scores,
                which=which,
            )
            metrics["cluster_average"] = 1.0
        else:
            # calculate metrics overall observations
            metrics = metric_function(
                y_true,
                y_hat,
                names=names,
                y_scores=y_scores,
                which=which,
            )
            metrics["cluster_average"] = 0.0
        return metrics

    return wrapper


@overload
def sample_weights(ids: dict[int, int]) -> dict[int, float]:
    ...


@overload
def sample_weights(ids: list[int] | np.ndarray) -> np.ndarray:
    ...


def sample_weights(
    ids: dict[int, int] | list[int] | np.ndarray,
) -> dict[int, float] | np.ndarray:
    """Calculate inverted frequencies from the IDs.

    Args:
        ids: ID identifying each instance (group) or a dictionary with already
            aggregated counts for each group.

    Returns:
        Weights for instances.
    """
    if isinstance(ids, dict):
        total: float = sum(ids.values())
        ids_inverse = {sample: total / count for sample, count in ids.items()}
        total = float(len(ids_inverse)) / sum(ids_inverse.values(), 0.0)
        return {sample: count * total for sample, count in ids_inverse.items()}

    ids = np.asarray(ids)
    weights = np.array([1.0 / (ids == identifier).mean() for identifier in ids])
    return weights.reshape(-1, 1) * (weights.size / weights.sum())


def interval_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    *,
    which: tuple[str, ...] = ("mse",),
    names: tuple[str, ...] = (),
    y_scores: np.ndarray | None = None,
    ids: np.ndarray | None = None,
    clustering: bool = False,
) -> dict[str, float]:
    """Calculate interval metrics.

    Note:
        Wrapped with metric_wrapper.

    Args:
        y_true: The ground truth.
        y_hat: The predictions.
        which: Which metrics to calculate.
        names: Not used.
        y_scores: Not used.
        ids: Determines the clusters.
        clustering: Whether to calculate the metrics for each cluster and
            then average.

    Return:
        The calculated metrics.
    """
    return metric_wrapper(_interval_metrics)(
        y_true,
        y_hat,
        which=which,
        names=names,
        y_scores=y_scores,
        ids=ids,
        clustering=clustering,
    )


def _interval_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    *,
    which: tuple[str, ...] = ("mse",),
    names: tuple[str, ...] = (),
    y_scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Interval metrics."""
    scores: dict[str, list[float]] = {}

    # mean squared error
    if generic.startswith_list("mse", which):
        scores.update(_mean_squared_error(y_true, y_hat))

    # mean absolute error
    if generic.startswith_list("mae", which):
        scores.update(_mean_absolute_error(y_true, y_hat))

    # concordance correlation coefficient
    if generic.startswith_list("ccc", which):
        scores.update(concordance_correlation_coefficient(y_true, y_hat))

    # correlation
    if generic.startswith_list("pearson", which):
        scores.update(
            {
                f"pearson_{key}": value
                for key, value in _correlation(y_true, y_hat).items()
            },
        )
    if generic.startswith_list("spearman", which):
        scores.update(
            {
                f"spearman_{key}": value
                for key, value in _correlation(_rank(y_true), _rank(y_hat)).items()
            },
        )

    # krippendorff
    if generic.startswith_list("krippendorff", which):
        scores.update(_krippendorff(y_true, y_hat=y_hat, weighting=_weighting_interval))

    # intraclass correlation coefficient
    if generic.startswith_list("icc", which):
        scores.update(intraclass_correlation_coefficient(y_true, y_hat))

    return scalar_dict(scores, names=names)


def ordinal_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    *,
    which: tuple[str, ...] = ("spearman",),
    names: tuple[str, ...] = (),
    y_scores: np.ndarray | None = None,
    ids: np.ndarray | None = None,
    clustering: bool = False,
) -> dict[str, float]:
    """Calculate ordinal metrics.

    Note:
        Wrapped with metric_wrapper.

    Args:
        y_true: The ground truth.
        y_hat: The predictions.
        which: Which metrics to calculate.
        names: Names of ordinal classes.
        y_scores: Not used.
        ids: Determines the clusters.
        clustering: Whether to calculate the metrics for each cluster and
            then average.

    Return:
        The calculated metrics.
    """
    return metric_wrapper(_ordinal_metrics)(
        y_true,
        y_hat,
        which=which,
        names=names,
        y_scores=y_scores,
        ids=ids,
        clustering=clustering,
    )


def _ordinal_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    *,
    which: tuple[str, ...] = ("spearman",),
    names: tuple[str, ...] = (),
    y_scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Ordinal metrics."""
    scores: dict[str, list[float]] = {}
    matrix: np.ndarray | None = None

    if generic.startswith_list("krippendorff", which) or generic.startswith_list(
        "cohens_kappa",
        which,
    ):
        matrix = _confusion_matrix(y_true, y_hat, len(names))

    # ordinal Krippendorff
    if matrix is not None and generic.startswith_list("krippendorff", which):
        scores.update(
            _krippendorff(
                y_true,
                y_hat=y_hat,
                weighting=_weighting_ordinal,
                confusion=matrix,
            ),
        )

    # ordinal Cohen's Kappa
    if matrix is not None and generic.startswith_list("cohens_kappa", which):
        scores.update(_cohen_kappa(matrix, weighting=_weighting_ordinal))

    # correlation
    if generic.startswith_list("spearman", which):
        scores.update(
            {
                f"spearman_{key}": value
                for key, value in _correlation(_rank(y_true), _rank(y_hat)).items()
            },
        )

    # intraclass_correlation coefficient
    if generic.startswith_list("icc", which):
        scores.update(intraclass_correlation_coefficient(y_true, y_hat))

    return scalar_dict(scores, names=names)


def _mean_squared_error(
    y_true: np.ndarray,
    y_hat: np.ndarray,
) -> dict[str, list[float]]:
    """Calculate the mean square error (MSE).

    Note:
        Uses Wilcoxon one-sided paired test to test whether the MSE is
        better than the MSE of a constant mean prediction.

    Returns:
        A dictionary with the fields: 'mse', 'mse_mean', 'mse_mean_p'
    """
    scores = {}
    error = (y_true - y_hat) ** 2
    scores["mse"] = error.mean(axis=0).tolist()

    # mean prediction
    error_mean = (y_true - y_true.mean(axis=0, keepdims=True)) ** 2
    scores["mse_mean"] = error_mean.mean(axis=0).tolist()

    # wilcoxon (can be slow for many samples)
    if error.shape[0] < 5000:
        difference = error - error_mean
        scores.update(_wilcoxon(difference, larger_than=False, name="mse_mean"))

    return scores


def _mean_absolute_error(
    y_true: np.ndarray,
    y_hat: np.ndarray,
) -> dict[str, list[float]]:
    """Calculate the mean absolute error."""
    scores = {}
    error = np.abs(y_true - y_hat)
    scores["mae"] = error.mean(axis=0).tolist()

    # median prediction
    error_median = np.abs(y_true - np.median(y_true, axis=0, keepdims=True))
    scores["mae_median"] = error_median.mean(axis=0).tolist()

    # wilcoxon (can be slow for many samples)
    if error.shape[0] < 5000:
        difference = error - error_median
        scores.update(_wilcoxon(difference, larger_than=False, name="mae_median"))

    return scores


def _wilcoxon(
    errors: np.ndarray,
    *,
    larger_than: bool = False,
    name: str = "wilcoxon",
) -> dict[str, list[float]]:
    """Calculate the p-value of a on-sided wilcoxon test."""
    name += "_p"
    scores: list[float] = []
    positive = errors.mean(axis=0) > 0
    for dim in range(errors.shape[1]):
        p_value = 1.0
        if errors[:, dim].any():
            p_value = wilcoxon(errors[:, dim], zero_method="wilcox").pvalue
        if isinstance(p_value, np.floating):
            p_value = p_value.item()
        # convert two-sided to one-sided
        p_value /= 2
        if positive[dim] != larger_than:
            p_value = 1 - p_value

        scores.append(p_value)

    return {name: scores}


def _confusion_matrix(y_true: np.ndarray, y_hat: np.ndarray, size: int) -> np.ndarray:
    """Create a confusion matrix.

    Args:
        y_true: The ground truth.
        y_hat: The most likely guess.
        size: The number of possible classes.

    Returns:
        The confusion matrix.
    """
    # populate matrix
    matrix = np.zeros((size, size))
    for ground_truth in range(size):
        y_hat_ = y_hat[y_true == ground_truth]
        if y_hat_.size == 0:
            continue
        matrix[ground_truth] = np.bincount(y_hat_, minlength=size)

    return matrix


def nominal_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    *,
    y_scores: np.ndarray | None = None,
    names: tuple[str, ...] = (),
    which: tuple[str, ...] = ("accuracy",),
    ids: np.ndarray | None = None,
    clustering: bool = False,
) -> dict[str, float]:
    """Calculate nominal metrics.

    Note:
        Wrapped with metric_wrapper.

    Args:
        y_true: The ground truth.
        y_hat: The predictions.
        which: Which metrics to calculate.
        names: Names of nominal classes.
        y_scores: The logits for each class.
        ids: Determines the clusters.
        clustering: Whether to calculate the metrics for each cluster and
            then average.

    Return:
        The calculated metrics.
    """
    return metric_wrapper(_nominal_metrics)(
        y_true,
        y_hat,
        y_scores=y_scores,
        names=names,
        which=which,
        ids=ids,
        clustering=clustering,
    )


def _nominal_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    *,
    y_scores: np.ndarray | None = None,
    names: tuple[str, ...] = (),
    which: tuple[str, ...] = ("accuracy",),
) -> dict[str, float]:
    """Nominal metrics, mostly all based on the confusion matrix.

    Names have to be in the same order has the sorted unique values.

    names       List of class names. All classes have to be present in the
                ground truth!
    """
    scores: dict[str, list[float]] = {}
    matrix = _confusion_matrix(y_true, y_hat, len(names))

    # normalized matrix
    matrix_norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    # confusion matrix
    if "confusion" in which:
        scores.update(_confusion(matrix, names))

    # macro: F1, recall, precision
    if generic.startswith_list("macro", which):
        scores.update(_precision_recall_f1(matrix))

    # accuracy
    if generic.startswith_list("accuracy", which):
        scores.update(_accuracy(matrix))
        scores.update({"accuracy_balanced": _accuracy(matrix_norm)["accuracy"]})
        scores.update(_accuracy_majority(matrix))

    # nominal Cohen's Kappa
    if generic.startswith_list("cohens_kappa", which):
        scores.update(_cohen_kappa(matrix, weighting=_weighting_nominal))

    # nominal Krippendorff
    if generic.startswith_list("krippendorff", which):
        scores.update(
            _krippendorff(
                y_true,
                y_hat=y_hat,
                weighting=_weighting_nominal,
                confusion=matrix,
            ),
        )

    # AUC/EER
    if generic.startswith_list("roc", which) and y_scores is not None:
        scores.update(_roc_auc(y_true, y_scores))

    # Brier score
    if generic.startswith_list("brier", which) and y_scores is not None:
        scores.update(_brier_score(y_true, y_scores))

    return scalar_dict(scores, names=names)


def _confusion(matrix: np.ndarray, names: tuple[str, ...]) -> dict[str, list[float]]:
    """Output for the confusion matrix."""
    scores = {}
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            scores[f"confusion___{name_i}___{name_j}"] = [matrix[i, j].item()]
    return scores


def _precision_recall_f1(matrix: np.ndarray) -> dict[str, list[float]]:
    """Calculate precision, recall, and F1."""
    scores: dict[str, list[float]] = {
        "macro_precision": [],
        "macro_recall": [],
        "macro_f1": [],
    }

    for i in range(matrix.shape[0]):
        true_positive = matrix[i, i]
        false_positive = matrix[:, i].sum() - true_positive
        false_negative = matrix[i, :].sum() - true_positive

        # precision: TP / (TP + FP)
        precision = true_positive + false_positive
        if precision != 0:
            precision = true_positive / precision
        scores["macro_precision"].append(float(precision))

        # recall: TP / (TP + FN)
        recall = true_positive + false_negative
        if recall != 0:
            recall = true_positive / recall
        scores["macro_recall"].append(float(recall))

        # F1: 2 * precision * recall / (precision + recall)
        f_score = precision + recall
        if f_score != 0 and np.isfinite(f_score):
            f_score = 2 * precision * recall / f_score
        scores["macro_f1"].append(float(f_score))

    return scores


def _accuracy(matrix: np.ndarray) -> dict[str, list[float]]:
    """Accuracy scores."""
    return {"accuracy": [matrix.trace().item() / matrix.sum().item()]}


def _accuracy_majority(matrix: np.ndarray) -> dict[str, list[float]]:
    """Accuracy of the majority class."""
    sums = matrix.sum(axis=1)
    argmax = np.argmax(sums)
    matrix_majority = np.zeros(matrix.shape)
    matrix_majority[argmax, :] = sums

    return {"accuracy_majority": _accuracy(matrix_majority)["accuracy"]}


def concordance_correlation_coefficient(
    y_true: np.ndarray,
    y_hat: np.ndarray,
) -> dict[str, list[float]]:
    """Calculate the concordance correlation coefficient(s)."""
    mean_y = y_true.mean(axis=0)
    mean_y_hat = y_hat.mean(axis=0)
    y_mean = y_true - mean_y
    y_hat_mean = y_hat - mean_y_hat
    cov = (y_mean * y_hat_mean).mean(axis=0)
    var = y_true.var(axis=0, ddof=0) + y_hat.var(axis=0, ddof=0)
    mse = (mean_y - mean_y_hat) ** 2
    return {"ccc": ((2 * cov) / (var + mse)).tolist()}


def _correlation(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, list[float]]:
    """Calculate the Pearson correlation."""
    scores: dict[str, list[float]] = {"r": [], "p": []}

    for dim in range(y_hat.shape[1]):
        r_value, p_value = pearsonr(y_true[:, dim], y_hat[:, dim])

        if math.isnan(r_value):
            r_value = 0.0
            p_value = 1.0
        if isinstance(r_value, np.number):
            r_value = r_value.item()
        if isinstance(p_value, np.number):
            p_value = p_value.item()

        scores["p"].append(p_value)
        scores["r"].append(r_value)

    return scores


def _krippendorff(
    y_true: np.ndarray,
    *,
    y_hat: np.ndarray,
    weighting: Callable[[np.ndarray, np.ndarray], np.ndarray],
    confusion: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Calculate Krippendorff's alpha.

    weighting   A weighting function.
    """
    scores: list[float] = []
    for idim in range(y_true.shape[1]):
        tmp = (y_true[:, idim, None], y_hat[:, idim, None])
        data = np.concatenate(tmp, axis=1).T
        scores.append(_krippendorff_(data, weighting=weighting, confusion=confusion))

    return {"krippendorff": scores}


def _krippendorff_(
    data: np.ndarray,
    *,
    weighting: Callable[[np.ndarray, np.ndarray], np.ndarray],
    confusion: np.ndarray | None = None,
) -> float:
    """Calculate Krippendorff's alpha for intervally and nominally scaled annotations.

    Note:
        It uses a memory-efficient implementation which might be slow!

    weighting   A weighting function.
    """
    # remove useless entries
    data = data[:, np.isfinite(data).all(axis=0)]
    assert np.isfinite(data).all()

    # faster version for many classes
    if weighting is _weighting_interval:
        return _krippendorff_interval(data)

    assert confusion is not None

    return _krippendorff_matrix(data, weighting, confusion)


def _krippendorff_interval(data: np.ndarray) -> float:
    """Calculate the intervally-scaled Krippendorff (for many distinct values)."""
    expected = 0.0
    flat = data.reshape(1, -1)
    for i in range(data.shape[1]):
        expected += ((data[:, i, None] - flat) ** 2).mean().item()
    observed = ((data[0, :] - data[1, :]) ** 2).mean().item()
    expected = expected * 2 / (data.size - 1)
    return 1 - observed / expected


def _krippendorff_matrix(
    data: np.ndarray,
    weighting: Callable[[np.ndarray, np.ndarray], np.ndarray],
    confusion: np.ndarray,
) -> float:
    """Calculate Krippendorff alpha for few distinct values."""
    # observed coincidence matrix
    observed = confusion + confusion.T
    # expected coincidence matrix
    n_value = observed.sum(axis=0)
    expected = n_value.reshape(-1, 1) * n_value.reshape(1, -1)
    np.fill_diagonal(expected, np.diag(expected) - n_value)
    expected /= observed.sum() - 1

    weights = weighting(observed, np.unique(data))
    observed = (observed * weights).sum()
    expected = (expected * weights).sum()
    return (1 - observed / max(1.0, expected)).item()


def _weighting_nominal(matrix: np.ndarray, values: np.ndarray | None) -> np.ndarray:
    """Calculate nominal weight."""
    return 1 - np.diag(np.ones(matrix.shape[0]))


def _weighting_ordinal(matrix: np.ndarray, values: np.ndarray | None) -> np.ndarray:
    """Calculate ordinal weights."""
    assert values is not None
    weights = np.zeros(matrix.shape)
    n_value = matrix.sum(axis=0)
    for i in range(n_value.shape[0]):
        for j in range(i, n_value.shape[0]):
            tmp = n_value[i : j + 1].sum()
            weights[i, j] = (tmp - (n_value[i] + n_value[j]) / 2) ** 2
            weights[j, i] = weights[i, j]
    return weights


def _weighting_interval(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Calculate interval weights."""
    return (values.reshape(-1, 1) - values.reshape(1, -1)) ** 2


def _cohen_kappa(
    matrix: np.ndarray,
    *,
    weighting: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
    values: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Calculate Cohen's kappa based on a (normalized) confusion matrix.

    matrix  The confusion matrix.
    weighting   A weighting function.
    values  Values corresponding to rows/columns of the matrix. Necessary
            for 'interval' scale.
    """
    # observed and expected
    sum_y_true = matrix.sum(axis=0)
    sum_y_hat = matrix.sum(axis=1)
    expected = np.outer(sum_y_true, sum_y_hat) / sum_y_true.sum()
    # weighting
    weight = weighting(matrix, values)
    return {
        "cohens_kappa": [
            (1 - (matrix * weight).sum() / max(1.0, (expected * weight).sum())).item(),
        ],
    }


def _rank(vector: np.ndarray) -> npt.NDArray[np.float_]:
    """Convert an array to ranks.

    Note:
        It breaks ties based on the mean rank.

    Args:
        vector: A numpy array

    Returns:
        The ranks.
    """
    # ranks without ties
    ranks = np.argsort(np.argsort(vector, axis=0), axis=0).astype(float)
    values = np.unique(vector)
    has_ties = values.size != ranks.size

    # break ties
    if has_ties:
        for value in values:
            index = vector == value
            ranks[index] = ranks[index].mean()

    return ranks


def _roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, list[float]]:
    """Calculate ROC AUC for multi-class classifications.

    Args:
        y_true: Vector indicating the active class (starting from 0).
        y_scores: Matrix of scores for the classes.

    Returns:
        ROC AUC and ROC EER for all dimensions.
    """
    # for binary predictions
    if y_scores.shape[1] == 1:
        return {
            f"{key}_macro": [value]
            for key, value in _binary_roc_auc(y_true == 1, y_scores).items()
        }

    scores: dict[str, list[float]] = {"roc_auc_macro": [], "roc_eer_macro": []}
    y_scores = softmax(y_scores, axis=1)
    for idim in range(y_scores.shape[1]):
        auc = _binary_roc_auc(y_true == idim, y_scores[:, idim])
        scores["roc_auc_macro"].append(auc["roc_auc"])
        scores["roc_eer_macro"].append(auc["roc_eer"])
    return scores


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Calculate ROC AUC for one binary classification.

    y_true  0-1-Vectoring encoding the positive class with 1.
    y_score The prediction scores for the positive class.
    """
    y_true = y_true.reshape(-1)
    y_score = y_score.reshape(-1)
    y_true = y_true[np.argsort(y_score)]

    # calculate ROC AUC
    auc = np.float64(0.0)
    false = 0.0
    # and equal error rate
    eer = np.float64("NaN")
    true = 0.0
    diff_false_neg_rate_false_pos_rate = 1.0
    active = (y_true != 0).sum()
    inactive = (y_true == 0).sum()
    if active == 0 or inactive == 0:
        return {"roc_auc": float("NaN"), "roc_eer": eer.item()}
    for y_i in y_true:
        # AUC
        false = false + (1.0 - y_i)
        auc = auc + y_i * false
        # ERR
        false_neg_rate = true / active
        false_pos_rate = (inactive - false) / inactive
        true = true + y_i
        new_diff = (false_neg_rate - false_pos_rate) ** 2
        if new_diff < diff_false_neg_rate_false_pos_rate:
            diff_false_neg_rate_false_pos_rate = new_diff
            eer = (false_neg_rate + false_pos_rate) / 2.0
    auc = auc / (false * (len(y_true) - false))
    return {"roc_auc": auc.item(), "roc_eer": eer.item()}


def intraclass_correlation_coefficient(
    y_true: np.ndarray,
    y_hat: np.ndarray,
) -> dict[str, list[float]]:
    """Intraclass Correlation Coefficient.

    y_true  n x r ndarray where n are the number of samples and r are the number of
            ratings from the same rater and about the same sample but assessing
            different aspects of the sample (same as calling this function r times with
            n x 1).
    y_hat   Same as y_true but from a different rater.

    Returns:
        ICC(C, 1) and ICC(A, 1)
    """
    data = np.stack((y_true, y_hat))
    size = y_true.shape[0]

    # Sum Square Total
    overall_means = data.mean(axis=(0, 1), keepdims=True)
    sum_squre_total = ((data - overall_means) ** 2).sum(axis=(0, 1))

    # Sum Square Error
    rater_means = data.mean(axis=1, keepdims=True)
    sum_square_errors = (
        np.power(np.diff(data - data.mean(axis=1, keepdims=True), axis=0), 2).sum(
            axis=(0, 1),
        )
        / 2
    )
    mean_square_erros = sum_square_errors / (size - 1)

    # Sum square column effect - between raters
    sum_square_raters = np.power(rater_means - overall_means, 2).sum(axis=(0, 1)) * size

    # Sum Square subject effect - between samples
    sum_square_samples = sum_squre_total - sum_square_raters - sum_square_errors
    mean_square_samples = sum_square_samples / (size - 1)

    return {
        "icc_c_1": (
            (mean_square_samples - mean_square_erros)
            / (mean_square_samples + mean_square_erros)
        ).tolist(),
        "icc_a_1": (
            (mean_square_samples - mean_square_erros)
            / (
                mean_square_samples
                + mean_square_erros
                + 2 / size * (sum_square_raters - mean_square_erros)
            )
        ).tolist(),
    }


def _brier_score(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, list[float]]:
    """Calculate the Brier score from logits."""
    y_true = y_true.reshape(-1)

    prob_diff = (
        # two-classes
        expit(y_scores) - y_true
        if y_scores.shape[1] == 1
        # n-classes
        else 1 - softmax(y_scores, axis=1)[range(y_true.size), y_true]
    )
    return {"brier_score": [np.power(prob_diff, 2).mean().item()]}
