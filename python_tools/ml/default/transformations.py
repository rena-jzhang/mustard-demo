#!/usr/bin/env python3
"""Default values for model.py and trainer.py."""
from collections import defaultdict
from functools import partial
from typing import Any, Literal, cast

import numpy as np
from sklearn.feature_selection import (
    SelectFromModel,
    SelectPercentile,
    f_classif,
    f_regression,
)
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from python_tools.ml import metrics
from python_tools.typing import DataLoader, DataLoaderT, TransformDict


class DefaultTransformations:
    """Learn standard dataset transformations."""

    def __init__(
        self,
        interval: bool = False,
        nominal: bool = False,
        binary: bool = False,
    ) -> None:
        """Feature selection/scaling."""
        self.interval = interval
        self.nominal = nominal
        self.binary = binary

    def _calculate_mean_std(
        self,
        iterator: DataLoader,
        keep: np.ndarray,
    ) -> TransformDict:
        """Calculate statistics over the dataset.

        Args:
            iterator: The dataset.
            keep: Which features should already be removed.

        Returns:
            The transformation information.
        """
        results: TransformDict = {"x": {}, "y": {}}
        meta_id: dict[int, int] = defaultdict(int)

        # mean, variance, min, max, y-label count, sample count
        scalers = {key: StandardScaler() for key in results}
        label_frequency: np.ndarray | None = None
        minmax: MinMaxScaler | None = None
        if self.nominal or self.binary:
            label_frequency = np.zeros(len(iterator.properties["y_names"]))
            del scalers["y"]
        elif self.interval:
            minmax = MinMaxScaler()

        # loop only one time over the data
        sample_count = 0
        batch_count = 0
        dtypes = {}
        for items in iterator:
            batch_count += 1
            sample_count += items["x"][0].shape[0]
            for key in results:
                item = items[key][0]
                dtypes[key] = item.dtype

                if key == "x":
                    item = item[:, keep]

                if key == "y" and label_frequency is not None:
                    # label frequency for classification
                    tmp = (
                        item.sum(axis=0)
                        if self.binary
                        else np.bincount(item.reshape(-1))
                    )
                    label_frequency[: tmp.size] += tmp
                else:
                    # clip min/max for regression
                    scalers[key].partial_fit(item)
                    if key == "y" and minmax is not None:
                        minmax.partial_fit(item)
            # count meta_id
            for identifier, count in zip(
                *cast(
                    tuple[np.ndarray, np.ndarray],
                    np.unique(items["meta_id"][0], return_counts=True),
                ),
                strict=True,
            ):
                meta_id[identifier.item()] += count.item()

        # mean and var
        for key in scalers:
            key = cast(Literal["x", "y"], key)
            results[key]["mean"] = scalers[key].mean_.astype(dtypes[key])
            results[key]["std"] = np.sqrt(scalers[key].var_).astype(dtypes[key])

        # inverse label frequency
        if label_frequency is not None:
            label_frequency = 1.0 - label_frequency / sample_count

        # min/max of labels
        if minmax is not None:
            results["y"]["min"] = minmax.data_min_.astype(dtypes["y"])
            results["y"]["max"] = minmax.data_max_.astype(dtypes["y"])
            assert "mean" in results["y"]
            for key in ("min", "max"):
                results["y"][f"{key}_transformed"] = (
                    results["y"][key] - results["y"]["mean"]
                ) / results["y"]["std"]

        # remove features with no variance
        min_value = np.finfo(np.float32).eps * 1000
        index = np.where(results["x"]["std"] > min_value)[0]
        for key in ("mean", "std"):
            key = cast(  # pyright: ignore[reportUnnecessaryCast]
                Literal["mean", "std"],
                key,
            )
            results["x"][key] = results["x"][key][index]
        results["x"]["keep"] = keep[index]

        # clip extremes
        results["meta_id_count"] = dict(
            sorted(meta_id.items(), key=lambda item: item[0]),
        )
        data = np.asarray(list(results["meta_id_count"].values()))
        median = np.median(data)
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        data = np.maximum(data, median - 1.5 * iqr)
        data = np.minimum(data, median + 1.5 * iqr).astype(int).tolist()
        id_count_clip = dict(zip(results["meta_id_count"], data, strict=True))
        results["meta_id"] = metrics.sample_weights(results["meta_id_count"])
        results["meta_id_clip"] = metrics.sample_weights(id_count_clip)
        results["meta_batches"] = batch_count
        results["meta_y_inverse_weights"] = label_frequency

        return results

    def _feature_selection(
        self,
        data: DataLoader,
        feature_selection: str = "",
    ) -> np.ndarray:
        assert next(iter(data))["x"][0].shape[1] == len(data.properties["x_names"])
        if not feature_selection or feature_selection == "univariate_100":
            return np.arange(len(data.properties["x_names"]))

        # assume everything fits into memory!
        features_list = []
        labels_list = []
        for item in data:
            features_list.append(item["x"][0])
            labels_list.append(item["y"][0])
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)[:, 0]
        features -= features.mean(axis=0)
        features /= np.clip(features.std(axis=0), 1e-3, None)
        if self.interval:
            labels -= labels.mean(axis=0)
            labels /= np.clip(labels.std(axis=0), 1e-3, None)

        # parse threshold
        assert "_" in feature_selection, feature_selection
        feature_selection, threshold = feature_selection.split("_", 1)

        if feature_selection == "multivariate":
            parameters: dict[str, tuple[Any, ...]] = {
                "random_state": (1,),
                "alpha": (0.0001, 0.001, 0.01, 0.1, 0.5, 1.0),
                "penalty": ("l2", "l1"),
                "eta0": (0.1,),
                "power_t": (0.5,),
                "tol": (0.01,),
            }
            if self.interval:
                search = GridSearchCV(
                    SGDRegressor(),
                    parameters
                    | {
                        "loss": ("epsilon_insensitive",),
                        "epsilon": (0.0, 0.01, 0.1, 0.5),
                    },
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                )
            else:
                search = GridSearchCV(
                    SGDClassifier(),
                    parameters | {"class_weight": ("balanced",)},
                    scoring="balanced_accuracy",
                    n_jobs=-1,
                )
            search.fit(features, labels)
            return np.nonzero(
                SelectFromModel(
                    search.best_estimator_,
                    prefit=True,
                    threshold=threshold,
                ).get_support(),
            )[0]

        assert feature_selection == "univariate", feature_selection
        score_function = f_classif
        if self.interval:
            score_function = f_regression
        return np.nonzero(
            SelectPercentile(score_func=score_function, percentile=float(threshold))
            .fit(features, labels)
            .get_support(),
        )[0]

    def define_transform(
        self,
        data: DataLoader,
        feature_selection: str = "",
    ) -> TransformDict:
        """Learn data transform.

        Note:
            1) removes features with very little variance;
            2) z-normalization of features and labels (only for regression);
            3) runs a linear feature selection.

        Args:
            data: The training dataset.
            feature_selection: Whether to run a linear feature selection.

        Returns:
            Transformation information.
        """
        keep = self._feature_selection(data, feature_selection)

        # calculate data transformations
        transform = self._calculate_mean_std(data, keep=keep)
        transform["meta_x_names"] = data.properties["x_names"]
        transform["meta_y_names"] = data.properties["y_names"]

        # remove 'y' if it is not intervally scaled
        if not self.interval:
            del transform["y"]

        return transform


def set_transform(
    data: DataLoaderT,
    transform: TransformDict,
    *,
    optimizable: bool = True,
) -> DataLoaderT:
    """Sets/overwrites the transform function of the DataLoader.

    Args:
        data: The dataloader.
        transform: The transform information.
        optimizable: Whether to apply the transformation.

    Returns:
        The dataloader with the transformation.
    """
    # set transform function
    data.add_transform(partial(apply_transform, transform), optimizable=optimizable)
    # update meta data
    data.properties.update(transform)
    data.properties["x_names"] = data.properties["x_names"][transform["x"]["keep"]]
    return data


def apply_transform(
    transform: TransformDict,
    item: dict[str, list[np.ndarray | list[int]]],
) -> dict[str, list[np.ndarray | list[int]]]:
    """Apply learned data transform.

    Args:
        transform: The learned transformation.
        item: Dictionary representing the data.

    Returns:
        A dictionary with the transformed data.
    """
    # Z-normalization
    item = item.copy()
    for key, transform_dict in transform.items():
        if key.startswith("meta_"):
            continue
        data = cast(np.ndarray, item[key][0])
        transform_dict = cast(dict[str, np.ndarray], transform_dict)

        # remove no-variance features
        if "keep" in transform_dict:
            data = data[:, transform_dict["keep"]]

        # z-norm
        data = (data - transform_dict["mean"]) / transform_dict["std"]
        assert np.isfinite(data).all(), key

        # clip outliers at p~0.998
        if key == "x":
            data.clip(-3, 3, out=data)

        item[key] = item[key].copy()
        item[key][0] = data

    return item


def revert_transform(
    predict: dict[str, np.ndarray],
    *,
    transform: TransformDict,
) -> dict[str, np.ndarray]:
    """Reverts data and model transform on the predicted output (labels).

    Args:
        transform: The learned transformation.
        predict: The input for the loss functions.

    Returns:
        The originally scaled labels.
    """
    if "y" not in transform:
        return predict

    for key in ["y", "y_hat"]:
        predict[key] = (predict[key] * transform["y"]["std"]) + transform["y"]["mean"]

    return predict
