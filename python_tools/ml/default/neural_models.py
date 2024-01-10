#!/usr/bin/env python3
"""Default values for model.py and trainer.py."""
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
from beartype import beartype

from python_tools.ml import neural, pytorch_tools
from python_tools.ml.default.models import BaseModel
from python_tools.ml.mixed import NeuralMixedEffects
from python_tools.ml.neural_wrapper import Trainer, device_if_possible, load_model_state
from python_tools.typing import DataLoader, MetricWrappedFun, ModelFun, ModelType


@beartype
def _weighted_loss_function(
    loss_function: str | Callable,
    weights: torch.Tensor | None,
    device: str,
) -> str | Callable:
    if weights is None:
        return loss_function
    weights = weights.to(device, non_blocking=True)
    if loss_function == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
    if loss_function == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss(pos_weight=weights[0], reduction="none")
    return loss_function


@beartype
def _keep_fields_needed_by_loss(
    meta: dict[str, torch.Tensor],
    keep_all: bool,
) -> dict[str, torch.Tensor]:
    """Shrink meta-dict to only keep fields that might be needed by a loss function.

    Args:
        meta: The potentially large dictionary.
        keep_all: Do not remove any fields.

    Returns:
        A subset of fields from the dictionary.
    """
    if keep_all:
        return meta.copy()

    return {
        key: value
        for key, value in meta.items()
        if key in ("meta_id", "meta_sample_weight") or key.startswith("meta_loss_")
    }


def _save(model: Trainer) -> dict[str, Any]:
    model.save_state()
    return model.saved_state


def _convert_input_sizes(kwargs: dict[str, Any], x_names: np.ndarray) -> dict[str, Any]:
    kwargs = kwargs.copy()
    for key, value in kwargs.items():
        if not key.endswith("input_sizes"):
            continue
        if isinstance(value[0], int):
            continue
        kwargs[key] = tuple(np.intersect1d(view, x_names).size for view in value)
    return kwargs


class DeepModel(BaseModel, ABC):
    """Abstract neural module.

    Requirements:
        same as BaseModel
        data loader has to return dictionaries with the field 'x'
    """

    @beartype
    def __init__(self, *, device: str = "cpu", **kwargs: Any) -> None:
        """Base-class for neural network wrappers."""
        self.device = device
        super().__init__(**kwargs)

    def _set_parameters(self) -> None:
        # Determine activation function and losses
        loss_function = "MSELoss"
        if self.binary:
            loss_function = "BCEWithLogitsLoss"
        elif self.nominal:
            loss_function = "CrossEntropyLoss"
        self.parameters["loss_function"] = [loss_function]

        # default parameters
        self.parameters["epochs"] = [500]
        self.parameters["early_stop"] = [50]
        self.parameters["optimizer"] = [torch.optim.Adam]
        self.parameters["lr"] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        self.parameters["dropout"] = [0.0, 0.5]
        self.parameters["lr_factor"] = [0.25]
        self.parameters["lr_patience_ratio"] = [4.0]
        self.parameters["attenuation"] = ["", "gaussian", "horserace"]
        self.parameters["sample_weight"] = [False, True]
        self.parameters["class_weight"] = [True]
        self.forward_names: tuple[str, ...] = (
            "loss_function",
            "attenuation",
            "attenuation_lambda",
        )

        self.sample_weight = None
        self.minmax = {"min": None, "max": None}
        self.parameters["minmax"] = [False]
        if self.interval:
            self.parameters["minmax"] = [False, True]
        else:
            self.parameters["class_weight"] = [True]

        # BCEWithLogitsLoss and CrossEntropy apply softmax/sigmoid themselves
        self.parameters["final_activation"] = [
            {"name": "linear"},
            {"name": "dwac", "embedding_size": 10, "regression": not self.nominal},
        ]
        if not self.nominal:
            self.parameters["final_activation"] += [
                {
                    "name": "gpvfe",
                    "inducing_points": 2000,
                    "embedding_size": 10,
                    "iterative_init": -1,
                },
                {"name": "lme", "embedding_size": 10},
            ]
        self.attenuation = ""

        # functions to add to model_dict
        self.model_dict["save"] = _save
        self.model_dict_functions["load"] = "_load"

        # label type
        if self.interval or self.binary:
            self.label_dtype = torch.float32
        else:
            self.label_dtype = torch.int64

        # keywords forwarded independent of model
        self.static_kwargs: tuple[str, ...] = (
            "model_class",
            "sample_weight",
            "minmax",
            "load_from_file",
            "ignore",
            "optimizer",
            "epochs",
            "early_stop",
            "lr",
            "lr_factor",
            "lr_patience_ratio",
            "weight_decay",
            "exclude_parameters_prefixes",
            "class_weight",
            "metric",
            "metric_max",
        )
        self.forward_prefix: str = ""

    @abstractmethod
    def _get_module_fun(
        self,
        kwargs: dict[str, Any],
        input_size: int,
        output_size: int,
        weights: torch.Tensor | None,
        training: DataLoader,
        x_names: np.ndarray,
    ) -> ModelFun:
        pass

    @beartype
    def _intrinsic_loss(
        self,
        model: ModelType,
        output: torch.Tensor,
        y_true: torch.Tensor,
        meta: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculate the loss to update the model with."""
        return model.loss(output, y_true, meta)

    @beartype
    def _model_wrapper(
        self,
        model: ModelType,
        data: torch.Tensor | torch.nn.utils.rnn.PackedSequence,
        meta: dict[str, torch.Tensor],
        *,
        dataset: str,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Call the model."""
        return model(data, meta, dataset=dataset, y=y)

    @beartype
    def _prepare_output(
        self,
        outputs_tensor: torch.Tensor,
        labels_tensor: torch.Tensor,
        meta_tensor: dict[str, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Map raw output to a suitable form."""
        # convert to numpy
        outputs = outputs_tensor.detach().cpu().numpy()
        labels = labels_tensor.detach().cpu().numpy()
        # loss attenuation as meta data
        meta = {key: value.detach().cpu().numpy() for key, value in meta_tensor.items()}
        if self.attenuation:
            outputs, meta["meta_sigma_2"] = outputs[:, :-1], outputs[:, -1]
        meta["y_scores"] = outputs
        if self.binary:
            # sigmoid (logits)
            outputs = outputs > 0
        elif self.nominal:
            # argmax
            outputs = np.argmax(outputs, axis=-1)
        return outputs, labels, meta

    @beartype
    def _fit(
        self,
        training: DataLoader,
        validation: DataLoader,
        metric_fun: MetricWrappedFun,
        partition_name: str,
        **kwargs: Any,
    ) -> tuple[Trainer, dict[str, float]]:
        self.device = device_if_possible(self.device)
        # get meta information
        if "final_activation" in kwargs:
            kwargs["final_activation"] = kwargs["final_activation"].copy()
        input_size, output_size, weights, x_names = self._get_properties(training)

        if not kwargs.get("minmax", True):
            self.minmax = {"min": None, "max": None}
        if not kwargs.get("sample_weight", True):
            self.sample_weight = None

        # extra output for loss attenuation
        self.attenuation = kwargs["attenuation"]
        if self.attenuation:
            output_size += 1

        match kwargs.get("final_activation", {"name": ""})["name"]:
            case "dwac":
                # DWAC needs to know the number of classes (and supports dependent
                # multivariate regressions/classifications)
                kwargs["final_activation"]["output_size"] = (output_size,)
                kwargs["final_activation"].setdefault("regression", not self.nominal)
                output_size = kwargs["final_activation"]["embedding_size"]
                output_size = kwargs["final_activation"]["embedding_size"]
            case "lme":
                # LME supports multiple independent regressions
                kwargs["final_activation"]["output_size"] = output_size
                kwargs["final_activation"]["number_of_cluster"] = len(
                    training.properties["meta_id_count"],
                )
            case "gpvfe":
                kwargs["final_activation"]["iterative_init"] = sum(
                    training.properties["meta_id_count"].values(),
                )
                output_size = kwargs["final_activation"]["embedding_size"]

        if kwargs.get("layers", 0) == -1 and "layer_sizes" in kwargs:
            kwargs["layer_sizes"] = None

        # weighted loss
        kwargs["loss_function"] = _weighted_loss_function(
            kwargs["loss_function"],
            weights if kwargs.get("class_weight", False) else None,
            self.device,
        )

        # validate kwargs: check that all keywords are expected
        unexpected = [
            key
            for key in kwargs
            if (key not in self.static_kwargs)
            and (key not in self.forward_names)
            and (not self.forward_prefix and key.startswith(self.forward_prefix))
        ]
        assert not unexpected, unexpected

        # get model fun
        kwargs = _convert_input_sizes(kwargs, x_names)
        model_fun = self._get_module_fun(
            kwargs,
            input_size,
            output_size,
            weights,
            training,
            x_names,
        )

        # initialize with existing model
        if kwargs.get("load_from_file", ""):
            model_fun = partial(
                load_model_state,
                model_fun,
                kwargs["load_from_file"],
                partition_name=partition_name,
                ignore=kwargs.get("ignore", ()),
            )

        # NME: add cluster information
        random_effects_keys = [
            key
            for key in kwargs
            if key.endswith("model_class")
            and issubclass(kwargs[key], NeuralMixedEffects)
        ]
        if random_effects_keys:
            generated_kwargs: dict[str, torch.Tensor] = {}
            clusters = torch.IntTensor(tuple(training.properties["meta_id_count"]))
            cluster_count = torch.Tensor(
                tuple(training.properties["meta_id_count"].values()),
            )
            for key in random_effects_keys:
                prefix = "_".join(key.split("_")[:-2])
                if prefix:
                    prefix += "_"
                generated_kwargs[f"{prefix}clusters"] = clusters
                generated_kwargs[f"{prefix}cluster_count"] = cluster_count
            assert len(model_fun.args) == 0
            model_fun = partial(
                model_fun.func,
                **model_fun.keywords,
                **generated_kwargs,
            )

        # trainer
        trainer = Trainer(
            model_fun=model_fun,
            metric_fun=metric_fun,
            metric=kwargs["metric"],
            metric_max=kwargs["metric_max"],
            device=self.device,
            optimizer_class=kwargs["optimizer"],
            epochs=kwargs["epochs"],
            early_stop=kwargs["early_stop"],
            run_model=self._run_model,
            lr=kwargs["lr"],
            lr_factor=kwargs["lr_factor"],
            lr_patience_ratio=kwargs["lr_patience_ratio"],
            weight_decay=kwargs.get("weight_decay", 0.0),
            exclude_parameters_prefixes=kwargs.get(
                "exclude_parameters_prefixes",
                ((),),
            ),
        )

        if self.minmax["min"] is not None:
            trainer.model.min = torch.from_numpy(self.minmax["min"])
            trainer.model.max = torch.from_numpy(self.minmax["max"])

        if kwargs.get("final_activation", {"name": ""})["name"] != "lme":
            output = trainer.fit(training, validation)
        else:
            # run EM
            iterations = kwargs["final_activation"]["iterations"]
            for iteration in range(kwargs["final_activation"]["iterations"]):
                # E-step: update fixed effects
                output = trainer.fit(training, validation)

                # M-step: update random effects
                meta = trainer.run(training, learn=False)[0]
                trainer.model.final_activation.estimate_random_effects(
                    torch.from_numpy(meta["y"])
                    - torch.from_numpy(meta["meta_y_hat_fixed"]),
                    torch.from_numpy(meta["meta_id"]),
                    torch.from_numpy(meta["meta_embedding"]),
                )

                if iteration + 1 == iterations:
                    break

                # reset all parameters except LME
                trainer.iexclude_parameters_prefix = 0
                trainer.metric_matrix = pd.DataFrame()
                trainer.save_state()
                trainer.saved_state["optimizer_state"] = {}
                trainer.saved_state["scheduler_state"] = {}
                for name in list(trainer.saved_state["model_state"]):
                    if "final_activation." in name or name.endswith(("min", "max")):
                        continue
                    del trainer.saved_state["model_state"][name]
                trainer.load_state()

        return trainer, output

    @beartype
    def _predict(
        self,
        model: Trainer,
        data: DataLoader,
        dataset: str,
    ) -> dict[str, np.ndarray]:
        data.curriculum_end()
        return model.run(data, learn=False)[0]

    def _load(self, saved_state: dict[str, Any], **kwargs: Any) -> Trainer:
        self.attenuation = kwargs["attenuation"]
        return Trainer(
            model_fun=None,
            device=self.device,
            saved_state=saved_state,
            run_model=self._run_model,
        )

    @beartype
    def _run_model(
        self,
        model: ModelType,
        item: tuple[
            torch.Tensor | torch.nn.utils.rnn.PackedSequence,
            torch.Tensor,
            dict[str, torch.Tensor],
        ],
        dataset: str,
    ) -> tuple[dict[str, np.ndarray], torch.Tensor | None]:
        # convert data
        features, labels, meta = item
        meta = _keep_fields_needed_by_loss(meta, dataset == "unknown")
        del item
        device = model.device()
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # model wrapper
        output, meta = self._model_wrapper(
            model,
            features,
            meta,
            y=labels,
            dataset=dataset,
        )
        meta = _keep_fields_needed_by_loss(meta, dataset == "unknown")

        # clip regression to training interval
        if model.min is not None:
            if model.min.device.type != output.device.type:
                model.min = model.min.to(output.device, non_blocking=True)
                model.max = model.max.to(output.device, non_blocking=True)
            for i in range(model.min.shape[0]):
                torch.clamp_(output[:, i], min=model.min[i], max=model.max[i])

        # get loss
        loss = None
        if dataset in ("training", "validation"):
            loss = self._intrinsic_loss(model, output, labels, meta)

        # prepare output
        output, labels, meta = self._prepare_output(output, labels, meta)
        output_dict = {"y_hat": output, "y": labels}
        output_dict.update(meta)

        return output_dict, loss

    @beartype
    def _get_transformation(
        self,
        item: dict[str, list[np.ndarray | list[int]]],
    ) -> tuple[
        torch.nn.utils.rnn.PackedSequence | torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Convert an item to x, y, meta form."""
        self.device = device_if_possible(self.device)

        features = pytorch_tools.prepare_tensor(
            item["x"],
            dtype=torch.float32,
            cuda_device=self.device,
        )

        labels = pytorch_tools.prepare_tensor(
            [item["y"][0]],
            dtype=self.label_dtype,
            cuda_device=self.device,
        )
        assert isinstance(labels, torch.Tensor)

        # remove sequence index from y and separate meta data
        meta = {
            key: torch.from_numpy(value[0].copy())
            for key, value in item.items()
            if key not in ("x", "y")
        }

        return features, labels, meta

    @beartype
    def _get_properties(
        self,
        data_loader: DataLoader,
    ) -> tuple[int, int, torch.Tensor | None, np.ndarray]:
        """Extract properties from the dataloader.

        Returns:
            Tuple of size three:
               0) Input dimension;
               1) Output dimension;
               2) Class weight vector.
        """
        # sizes
        input_size = len(data_loader.properties["x_names"])
        output_size = 1
        if len(data_loader.properties["y_names"]) > 2:
            output_size = len(data_loader.properties["y_names"])
        if self.nominal and output_size == 1:
            assert self.binary
        # weights
        weights = data_loader.properties["meta_y_inverse_weights"]
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float()
        # minmax after scaling
        if (
            "y" in data_loader.properties
            and "min_transformed" in data_loader.properties["y"]
        ):
            self.minmax = {
                "min": data_loader.properties["y"]["min_transformed"],
                "max": data_loader.properties["y"]["max_transformed"],
            }
        # prepare sample weighting
        self.sample_weight = data_loader.properties.get("meta_id_clip", None)

        return input_size, output_size, weights, data_loader.properties["x_names"]


class MLPModel(DeepModel):
    """Abstract Deep module.

    Requirements:
        - same as DeepModel
        - data has to be batched with lists
    """

    def _set_parameters(self) -> None:
        super()._set_parameters()

        # set parameters
        self.parameters["layers"] = [0, 1, 2, 3, 4]
        self.parameters["layer_sizes"] = [None]
        self.parameters["activation"] = [
            {"name": "linear"},
            {"name": "ReLU"},
        ]

        self.model_dict["name"] = "mlp"
        self.forward_names = (
            *self.forward_names,
            "layers",
            "activation",
            "final_activation",
            "dropout",
            "layer_sizes",
        )

    @beartype
    def _get_module_fun(
        self,
        kwargs: dict[str, Any],
        input_size: int,
        output_size: int,
        weights: torch.Tensor | None,
        training: DataLoader,
        x_names: np.ndarray,
    ) -> ModelFun:
        return partial(
            kwargs.get("model_class", neural.MLP),
            input_size=input_size,
            output_size=output_size,
            sample_weight=self.sample_weight,
            **{
                key: value for key, value in kwargs.items() if key in self.forward_names
            },
        )


class EnsembleModel(MLPModel):
    """Used for the FG'20 paper."""

    def _set_parameters(self) -> None:
        super()._set_parameters()

        # set parameters
        self.parameters["size"] = [10]

        self.model_dict["name"] = "ensemble"
        self.forward_names = (*self.forward_names, "model", "size", "final_activation")
        self.forward_prefix = "model_"

    @beartype
    def _get_module_fun(
        self,
        kwargs: dict[str, Any],
        input_size: int,
        output_size: int,
        weights: torch.Tensor | None,
        training: DataLoader,
        x_names: np.ndarray,
    ) -> ModelFun:
        return partial(
            kwargs.get("model_class", neural.Ensemble),
            input_size=input_size,
            output_size=output_size,
            sample_weight=self.sample_weight,
            **{
                key: value
                for key, value in kwargs.items()
                if (key.startswith(self.forward_prefix) and key != "model_class")
                or key in self.forward_names
            },
        )


class PoolModel(DeepModel):
    """Abstract Pooled CNN with MLP backend.

    Requirements"
        - same as MLPModel
    """

    def _set_parameters(self) -> None:
        super()._set_parameters()

        # set parameters
        self.parameters["model_before"] = [neural.MLP]
        self.parameters["model_after"] = [neural.MLP]
        self.parameters["pooling_size"] = [20]
        self.parameters["concat"] = [False, True]
        self.parameters["bootstrap"] = [False, True]
        self.parameters["pooling"] = [
            {"name": "mean", "dim": 1},
            {"name": "var", "dim": 1},
            {"name": "max", "dim": 1},
            {"name": "stats", "dim": 1},
        ]
        del self.parameters["dropout"]

        self.model_dict["name"] = "pool"
        self.forward_names = (
            *self.forward_names,
            "pooling",
            "pooling_size",
            "concat",
            "bootstrap",
            "final_activation",
        )

    @beartype
    def _get_module_fun(
        self,
        kwargs: dict[str, Any],
        input_size: int,
        output_size: int,
        weights: torch.Tensor | None,
        training: DataLoader,
        x_names: np.ndarray,
    ) -> ModelFun:
        return partial(
            kwargs.get("model_class", neural.NN_Pool_NN),
            input_size=input_size,
            output_size=output_size,
            sample_weight=self.sample_weight,
            **{
                key: value
                for key, value in kwargs.items()
                if (key.startswith(self.forward_prefix) and key != "model_class")
                or key in self.forward_names
            },
        )


class AttenuatedModalityExperts(DeepModel):
    """Used for the ICMI'21 paper."""

    def _set_parameters(self) -> None:
        super()._set_parameters()

        # set parameters
        self.parameters["model"] = [neural.MLP]
        self.parameters["competitive"] = [False, True]
        self.parameters["joint_attenuation"] = [None]
        del self.parameters["dropout"]

        self.model_dict["name"] = "AttenuatedModalityExperts"
        self.forward_names = (
            *self.forward_names,
            "competitive",
            "input_sizes",
            "final_activation",
            "model",
            "combinations",
            "joint_attenuation",
            "latent_gating",
        )

    @beartype
    def _get_module_fun(
        self,
        kwargs: dict[str, Any],
        input_size: int,
        output_size: int,
        weights: torch.Tensor | None,
        training: DataLoader,
        x_names: np.ndarray,
    ) -> ModelFun:
        return partial(
            kwargs.get("model_class", neural.Attenuated_Modality_Experts),
            output_size=output_size,
            sample_weight=self.sample_weight,
            **{
                key: value
                for key, value in kwargs.items()
                if (key.startswith("model_") or key in self.forward_names)
                and key != "model_class"
            },
        )


class GenericModel(DeepModel):
    """Generic model wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        """Create a generic model wrapper.

        Note:
            User needs to overwrite self.forward_names.

        Args:
            kwargs: Forwarded to DeepModel.
        """
        super().__init__(**kwargs)

        del self.parameters["dropout"]

        self.model_dict["name"] = "generic"

    @beartype
    def _get_module_fun(
        self,
        kwargs: dict[str, Any],
        input_size: int,
        output_size: int,
        weights: torch.Tensor | None,
        training: DataLoader,
        x_names: np.ndarray,
    ) -> ModelFun:
        return partial(
            kwargs["model_class"],
            input_size=input_size,
            output_size=output_size,
            sample_weight=self.sample_weight,
            **{
                key: value for key, value in kwargs.items() if key in self.forward_names
            },
        )
