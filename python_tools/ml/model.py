#!/usr/bin/env python3
"""A model wrapper for the hyper-parameter evaluation."""
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from python_tools import caching, generic
from python_tools.ml.metrics import concat_dicts
from python_tools.typing import DataLoader, MetricWrappedFun, ModelFitFun, PredictFun


class Model:
    """A wrapper for models."""

    def __init__(
        self,
        *,
        fit: ModelFitFun,
        predict: PredictFun,
        save: Callable[[Any], Any] = generic.identity,
        load: Callable[..., Any] = generic.identity,
        name: str = "",
    ) -> None:
        """Instantiate a wrapper to house the actual model.

        Args:
            fit: A function returning a tuple containing the trained model and
                dictionary which will be included in the final CSV.
            predict: A function return the predicted labels and the ground truth.
            save: A function extracting all parameters from a model.
            load: A function reconstructing the model.
            name: Name of the model.
        """
        self.fit_function = fit
        self.save_function = save
        self.load_function = load
        self.predict_function = predict
        self.name = name

    def fit(
        self,
        train: DataLoader,
        validate: DataLoader,
        metric_fun: Callable[..., dict[str, float]],
        name: Path,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Train a model.

        This function is supposed to be called only once.

        Args:
            train: The training partition.
            validate: The validation partition.
            metric_fun: Function returning metrics in a dictionary.
            name: Prefix used to write/load results to.
            **kwargs: Parameters for the model.

        Returns:
            A tuple of size 2:
            0) the trained model;
            1) dictionary to be included in the overview.
        """
        # model fit
        start = time.time()
        partition_name = name.name.split("partition_", 1)[1].split("_", 1)[0]
        model, output = self.fit_function(
            train,
            validate,
            metric_fun,
            partition_name,
            **kwargs,
        )
        output["duration"] = time.time() - start
        return model, output

    def save(
        self,
        name: Path,
        model: Any,
        output: dict[str, Any],
        **kwargs: dict[str, Any],
    ) -> None:
        """Save the model with all its parameters.

        All information has to be saved to restore the model from only this file.

        Args:
            model: The model to be saved.
            output: The debug output while training the model.
            name: Path (without an extension) to save the model to.
            **kwargs: Arguments passed to the model.
        """
        caching.write_pickle(
            name.parent / name.name,
            {"arguments": kwargs, "output": output, "model": self.save_function(model)},
        )

    def load(self, name: Path) -> tuple[Any, dict[str, Any]]:
        """Load a model from file.

        Args:
            name: Path to save the model to.

        Returns:
            The model
        """
        try:
            data = caching.read_pickle(name.parent / name.name)
        except OSError:
            data = None

        if data is None:
            return data, {}
        return (
            self.load_function(data["model"], **data["arguments"]),
            data["output"],
        )

    def predict(
        self,
        model: Any,
        data: DataLoader,
        dataset: str,
    ) -> dict[str, np.ndarray]:
        """Predict the labels.

        Args:
            model: The model used for the prediction.
            data: The partition object with the data.
            dataset: Name of the dataset.
        """
        return self.predict_function(model, data, dataset)

    def metric(
        self,
        predict_output: list[dict[str, np.ndarray]],
        metric_fun: MetricWrappedFun,
    ) -> dict[str, float]:
        """Calculate the metrics.

        Args:
            predict_output: Its output or its output in a list.
            metric_fun: Function returning metrics in a dictionary.

        Has to return a dictionary of scalars.
        """
        output = concat_dicts(
            predict_output,
            keys=("y_hat", "y", "meta_id", "y_scores"),
        )
        return metric_fun(
            output["y"],
            output["y_hat"],
            ids=output["meta_id"],
            y_scores=output.get("y_scores", None),
        )

    def __str__(self) -> str:
        """Print the model name."""
        return self.name
