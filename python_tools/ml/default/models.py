#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues = false
"""Model factory for ml.model and ml.evaluator.

- SVMModel: A SVM; and
- ForestModel: Random Forest.
"""
# @beartype
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

from python_tools import generic
from python_tools.ml import metrics
from python_tools.ml.metrics import concat_dicts
from python_tools.ml.model import Model
from python_tools.typing import DataLoader, MetricWrappedFun


class BaseModel(ABC):
    """Provides the external interface to return a model."""

    def __init__(
        self,
        interval: bool = False,
        nominal: bool = False,
        binary: bool = False,
    ) -> None:
        """Base-class to define model wrappers.

        Args:
            interval: Whether the label is intervally-scaled.
            nominal: Whether the label are nominal classes.
            binary: Whether there are only two classes.
        """
        self.interval = interval
        self.nominal = nominal
        self.binary = binary

        # should be updated(!) by inherited classes
        self.parameters: dict[str, list[Any]] = {}
        self.model_dict = {"name": "base"}

        # functions to add to model_dict
        self.model_dict_functions = {
            "fit": "_fit",
            "predict": "_predict",
        }

        self._set_parameters()

    @abstractmethod
    def _set_parameters(self) -> None:
        ...

    def get_models(
        self,
    ) -> tuple[
        tuple[Model, ...],
        tuple[dict[str, Any], ...],
        Callable[[dict[str, list[np.ndarray]]], dict[str, list[np.ndarray]]],
    ]:
        """Return models and its parameters."""
        for key, function in self.model_dict_functions.items():
            self.model_dict[key] = getattr(self, function)
        model = Model(**self.model_dict)  # type: ignore[arg-type]
        parameters = tuple(generic.combinations(self.parameters))
        return (model,) * len(parameters), parameters, self._get_transformation

    def _get_transformation(
        self,
        item: dict[str, list[np.ndarray]],
    ) -> dict[str, list[np.ndarray]]:
        return item

    @abstractmethod
    def _fit(
        self,
        training: DataLoader,
        validation: DataLoader,
        metric_fun: MetricWrappedFun,
        partition_name: str = "",
        **kwargs: Any,
    ) -> tuple[Any, dict[str, float]]:
        ...

    @abstractmethod
    def _predict(
        self,
        model: Any,
        data: DataLoader,
        dataset: str,
    ) -> dict[str, np.ndarray]:
        ...


class SVMModel(BaseModel):
    """SVM wrapper."""

    def __init__(self, *, linear: bool = False, **kwargs: Any) -> None:
        """SVM wrapper.

        Args:
            linear: Whether to use LinearSVC/R.
            **kwargs: Forwarded to BaseModel.
        """
        self.linear = linear
        super().__init__(**kwargs)

    def _set_parameters(self) -> None:
        # SVM/SVR
        self.categorical = self.nominal or self.binary

        # set parameters
        self.parameters = {
            "kernel": ["linear", "rbf"],
            "epsilon": [0.0, 0.1, 0.2, 0.5, 1.0],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "gamma": ["auto", "scale"],
            "sample_weight": [False, True],
        }

        if self.categorical:
            del self.parameters["epsilon"]
            self.parameters["class_weight"] = [None, "balanced"]
            self.model_dict["name"] = "SVM"
        else:
            self.model_dict["name"] = "SVR"

        if self.linear:
            del self.parameters["kernel"]
            del self.parameters["gamma"]

    def _init(self, **kwargs: Any) -> Any:
        parameters = {
            key: value for key, value in kwargs.items() if key != "sample_weight"
        }
        if self.linear:
            parameters["random_state"] = 1
            parameters["max_iter"] = 10000
            # expect to have more observations than features
            parameters["dual"] = False
            return (
                LinearSVC(loss="squared_hinge", **parameters)
                if self.categorical
                else LinearSVR(loss="squared_epsilon_insensitive", **parameters)
            )
        if self.categorical:
            return SVC(cache_size=5000, random_state=1, **parameters)
        return SVR(cache_size=5000, **parameters)

    def _fit(
        self,
        training: DataLoader,
        validation: DataLoader,
        metric_fun: MetricWrappedFun,
        partition_name: str = "",
        **kwargs: Any,
    ) -> tuple[Any, dict[str, float]]:
        # convert batches to one giant batch
        data = concat_dicts(
            [{key: value[0] for key, value in item.items()} for item in training],
        )

        model = self._init(**kwargs)
        weights = None
        if kwargs["sample_weight"]:
            weights = metrics.sample_weights(
                [identifier[0] for identifier in data["meta_id"]],
            )[:, 0]
        return model.fit(data["x"], data["y"][:, 0], sample_weight=weights), {}

    def _predict(
        self,
        model: Any,
        data: DataLoader,
        dataset: str,
    ) -> dict[str, np.ndarray]:
        result = []
        for item in data:
            # make shallow copy and remove batching
            item = {key: value[0] for key, value in item.items()}
            # predict
            item["y_hat"] = model.predict(item["x"])
            # add scores
            if hasattr(model, "decision_function"):
                item["y_scores"] = model.decision_function(item["x"])
            elif hasattr(model, "predict_proba"):
                item["y_scores"] = model.predict_proba(item["x"])
            if (
                self.binary
                and "y_scores" in item
                and item["y_scores"].ndim == 2
                and item["y_scores"].shape[1] > 1
            ):
                item["y_scores"] = item["y_scores"][:, -1, None]
            # delete non-y and non-meta
            for key in tuple(item):
                if key.startswith(("y", "meta")):
                    continue
                del item[key]
            result.append(item)
        return concat_dicts(result)


class ForestModel(SVMModel):
    """Random Forest."""

    def _set_parameters(self) -> None:
        # set parameters
        self.parameters = {
            "n_estimators": [2000],
            "criterion": ["mse", "mae"],
            "max_features": ["auto", None],
            "min_samples_leaf": [0.1, 0.01, 0.001, 10],
            "sample_weight": [False, True],
            "oob_score": [True],
        }
        self.model_dict["name"] = "RandomForestRegressor"
        if self.categorical:
            self.parameters.update(
                {"criterion": ["gini", "entropy"], "class_weight": [None, "balanced"]},
            )
            self.model_dict["name"] = "RandomForestClassifier"

    def _init(self, **kwargs: Any) -> Any:
        forest = RandomForestRegressor
        if self.categorical:
            forest = RandomForestClassifier
        return forest(
            random_state=1,
            **{key: value for key, value in kwargs.items() if key != "sample_weight"},
        )
