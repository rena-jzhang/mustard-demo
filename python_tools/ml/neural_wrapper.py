#!/usr/bin/env python3
"""A wrapper to train a PyTorch model."""
import functools
import os
from collections import OrderedDict
from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import torch

from python_tools import caching, generic
from python_tools.generic import get_object
from python_tools.ml.metrics import concat_dicts
from python_tools.typing import DataLoader, LossModule, MetricWrappedFun, ModelFun


def load_model_state(
    model: LossModule | ModelFun,
    state_dict: dict[str, torch.Tensor] | Path,
    *,
    partition_name: str = "",
    ignore: tuple[str, ...] = (),
) -> LossModule:
    """Load parameters into a model."""
    # instantiate the model
    if not isinstance(model, LossModule):
        model = model()
    # load the statedict
    if isinstance(state_dict, Path):
        assert state_dict.is_file(), f"'{state_dict}' does not exist"
        # change partition, if available
        if partition_name:
            current_path_split = state_dict.name.split("partition_", 1)
            state_dict_ = state_dict.with_name(
                current_path_split[0]
                + f"partition_{partition_name}_"
                + current_path_split[1].split("_", 1)[1],
            )
            if state_dict_.is_file():
                state_dict = state_dict_
        state_dict = caching.read_pickle(state_dict)["model"]["model_state"]
    # ignore keys
    assert isinstance(state_dict, dict)
    state_dict = {key: value for key, value in state_dict.items() if key not in ignore}
    # remove _orig_mod prefix for non-compiled models
    if not hasattr(model, "_orig_mod"):
        state_dict = {
            key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
        }
    # non-parametric models might have parameters with a different size
    for key, value in model.state_dict().items():
        if state_dict.get(key, value).shape == value.shape:
            continue
        target = get_object(model, key.split("."))
        source = state_dict.pop(key)
        if target.dtype == source.dtype:
            # over-write inplace
            target.set_(source=source.to(device=target.device))
        else:
            # change object in parent
            parent = get_object(model, key.split(".")[:-1])
            setattr(parent, key.split(".")[-1], source.clone())
    # set parameters
    unexpected = model.load_state_dict(
        cast(OrderedDict[str, torch.Tensor], state_dict),
        strict=False,
    )[1]
    for key in unexpected:
        obj = generic.get_object(model, key.split(".")[:-1])
        if hasattr(obj, "register_buffer"):
            obj.register_buffer(key.split(".")[-1], state_dict[key])
    return model


def device_if_possible(device: str) -> str:
    """Disable cuda if there is no cuda device."""
    if "cuda" in device:
        if torch.cuda.device_count() == 0:
            return "cpu"
        gpu_id = os.environ.get("MY_CUDA_VISIBLE_DEVICES", "-1")
        if gpu_id != "-1":
            return f"cuda:{gpu_id}"
    return device


def set_random_state(device: str) -> None:
    """Try to be reproducible by setting the same random seeds.

    Args:
        device: Set the seed for a specific GPU.
    """
    torch.manual_seed(1)
    if "cuda" in device:
        with torch.cuda.device(torch.device(device)):
            torch.cuda.manual_seed(1)


class Trainer:
    """Encapsulates the training and validation process."""

    optimizer: torch.optim.Optimizer
    model: LossModule
    model_fun: ModelFun

    def __init__(
        self,
        *,
        model_fun: ModelFun | None,
        metric: str = "",
        metric_fun: MetricWrappedFun | None = None,
        metric_max: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        device: str = "cuda",
        saved_state: dict[str, Any] | None = None,
        epochs: int = 10,
        early_stop: int = 10,
        run_model: Callable[
            [torch.nn.Module, Any, str],
            tuple[dict[str, np.ndarray], torch.Tensor | None],
        ],
        lr_factor: float = 0.5,
        lr_patience_ratio: float = 4.0,
        exclude_parameters_prefixes: tuple[tuple[str, ...], ...] = ((),),
        **optimizer_args: Any,
    ) -> None:
        """Train neural networks and take care of all the small details.

        model_fun   Function to create the PyTorch model.
        metric      String representing the name of the metric used to do
                    early stopping.
        metric_fun  Function returning metrics in a dictionary.
        metric_max  Whether the maximum is the best  value of the metric.
        optimizer_class The class of the optimizer used for the training.
        device      String determining the device to use, e.g., 'cuda:0'.
        saved_state: A previously saved state from Trainer.save_state().
        epochs      Number of epochs to train for.
        early_stop  If the best metric is not within the last n iterations, stop.
        run_model   Function taking the model, one data item, and the dataset
                    name to return an output dict (with at least 'y' and 'y_hat')
                    and the loss.
        kwargs      Remaining parameters are passed to the Optimizer.
        """
        self.device = device_if_possible(device)
        self.early_stop = early_stop
        self.epochs = epochs
        self.exclude_parameters_prefixes = exclude_parameters_prefixes
        self.metric = metric
        self.metric_matrix = pd.DataFrame()
        self.metric_max = metric_max
        self.lr_factor = lr_factor
        self.lr_patience_ratio = lr_patience_ratio
        self.optimizer_args = optimizer_args
        self.optimizer_class = optimizer_class
        self.run_model = run_model
        self.iexclude_parameters_prefix = 0
        #  can be None if saved_state is provided
        if metric_fun is not None:
            self.metric_fun = metric_fun
        if model_fun is not None:
            self.model_fun = model_fun

        # save-able attributes
        self.attributes: Final = (
            "metric",
            "metric_max",
            "model_fun",
            "device",
            "run_model",
            "metric_fun",
            "optimizer_class",
            "optimizer_args",
            "metric_matrix",
            "early_stop",
            "epochs",
            "lr_factor",
            "lr_patience_ratio",
            "exclude_parameters_prefixes",
            "iexclude_parameters_prefix",
        )

        # init
        self.train_prefix = "training_"
        self.val_prefix = "validation_"
        self.saved_state = saved_state
        self.load_state()

    def run(
        self,
        iterator: Iterable[Any],
        *,
        learn: bool = False,
        dataset: str = "unknown",
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Evaluate the model and perform updates if requested.

        iterator    An iterable object.
        learn       Whether to perform updates.
        dataset     Can be 'training', 'validation', and 'unknown'.

        Returns:
            A tuple of size 2:
            0) a dictionary with 'y', 'y_hat', and fields returns by
               prepair_output;
            1) a dictionary containing the metrics.
        """
        metrics = {}
        if dataset in ("training", "validation"):
            metrics["optimization"] = 0.0
        outputs = []
        counter = -1
        self.model.zero_grad(set_to_none=True)

        # learn/predict
        if learn:
            self.model.train()
        else:
            self.model.eval()
        torch.set_grad_enabled(learn)

        # enable inference_mode when #60333 is fixed
        with torch.inference_mode(mode=False):
            for counter, data in enumerate(iterator, 1):
                # run model
                data, loss = self.run_model(self.model, data, dataset)
                outputs.append(data)

                if dataset in ("training", "validation"):
                    assert loss is not None
                    metrics["optimization"] += float(loss.detach().cpu().item())

                # learn
                if learn:
                    assert loss is not None
                    assert torch.isfinite(
                        loss,
                    ).all(), f"Loss is {loss} after batch {counter}"

                    loss.backward()
                    self.optimizer.step()
                    self.model.zero_grad(set_to_none=True)
                del loss

            # prepare output and calculate metrics
            if "optimization" in metrics:
                metrics["optimization"] /= counter
            output_dict = concat_dicts(outputs)
            metrics.update(
                self.metric_fun(
                    output_dict["y"],
                    output_dict["y_hat"],
                    ids=output_dict["meta_id"],
                    y_scores=output_dict["y_scores"],
                ),
            )

            return output_dict, metrics

    def load_state(self, *, iexclude_parameters_prefix: int = -1) -> None:
        """Load previously saved state.

        Args:
            iexclude_parameters_prefix: Whether to ignore the saved
                iexclude_parameters_prefix.
        """
        set_random_state(self.device)
        if self.saved_state is not None:
            # load attributes
            for key in self.attributes:
                if key == "device":
                    continue
                setattr(self, key, self.saved_state[key])
            self.model = load_model_state(
                self.model_fun,
                self.saved_state["model_state"],
            )
        else:
            self.model = self.model_fun()
        assert self.model is not None
        if iexclude_parameters_prefix == -1:
            iexclude_parameters_prefix = self.iexclude_parameters_prefix
        self.model.disable_gradient_for_excluded(
            self.exclude_parameters_prefixes[iexclude_parameters_prefix],
        )

        # try to JIT
        self.model.to(self.device, non_blocking=True)
        if self.model.can_jit():
            self.model = cast(LossModule, torch.jit.script(self.model))

        # optimizer
        parameters = list(self.model.parameters())
        self.optimizer = self.optimizer_class(parameters, **self.optimizer_args)

        # scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=self.lr_factor,
            mode="max" if self.metric_max else "min",
            patience=max(round(self.early_stop / self.lr_patience_ratio), 10),
        )

        # state of optimizer&scheduler
        if self.saved_state is not None:
            if self.saved_state["optimizer_state"]:
                self.optimizer.load_state_dict(self.saved_state["optimizer_state"])
            if self.saved_state["scheduler_state"]:
                self.scheduler.load_state_dict(self.saved_state["scheduler_state"])

    def save_state(self) -> None:
        """Save the entire state (including the metrics) to memory."""
        # save internal state
        self.saved_state = {key: getattr(self, key) for key in self.attributes}

        # do not save load_model_state
        if (
            isinstance(self.saved_state["model_fun"], functools.partial)
            and self.saved_state["model_fun"].func is load_model_state
        ):
            self.saved_state["model_fun"] = self.saved_state["model_fun"].args[0]

        self.saved_state["optimizer_state"] = deepcopy(self.optimizer.state_dict())
        self.saved_state["scheduler_state"] = deepcopy(self.scheduler.state_dict())
        if "cpu" in self.device:
            self.saved_state["model_state"] = deepcopy(self.model.state_dict())
        else:
            # avoid using additional GPU memory
            self.saved_state["model_state"] = {
                name: parameter.cpu()
                for name, parameter in self.model.state_dict().items()
            }

    def stop(self) -> bool:
        """Return whether to stop."""
        decision = False
        if self.early_stop > 0 and self.metric_matrix.shape[0] > 0:
            column = self.metric_matrix[self.val_prefix + self.metric]
            # is maximum in recent epochs
            index = column.idxmax() if self.metric_max else column.idxmin()
            decision = (
                self.metric_matrix.shape[0] - index  # pyright: ignore[reportGeneralTypeIssues]
                > self.early_stop
            )
            # is there a change in the recent epochs
            if self.metric_matrix.shape[0] > self.early_stop:
                decision |= column.iloc[-self.early_stop :].std().item() < 1e-8
        return decision

    def append_metrics(
        self,
        train_metric: dict[str, float],
        val_metric: dict[str, float],
    ) -> None:
        """Append new metrics to the metric matrix."""
        train_metric = {
            self.train_prefix + key: train_metric[key] for key in train_metric
        }
        train_metric["iexclude_parameters_prefix"] = self.iexclude_parameters_prefix
        train_metric.update(
            {self.val_prefix + key: val_metric[key] for key in val_metric},
        )

        self.metric_matrix = pd.concat(
            (self.metric_matrix, pd.DataFrame(train_metric, index=[0])),
            axis=0,
            ignore_index=True,
            sort=False,
        )

    def fit(
        self,
        train_iterator: DataLoader,
        val_iterator: DataLoader,
    ) -> dict[str, float]:
        """Train the neural network. Also supports curriculum learning.

        Args:
            train_iterator: Training data.
            val_iterator: Validation data.

        Returns:
            Metrics about the trained model.
        """
        outputs = {}
        metric_matrix = self.metric_matrix
        # data curriculum
        for step in range(train_iterator.curriculum_steps()):
            train_iterator.curriculum_set(step)
            val_iterator.curriculum_set(step)

            # fine-tuning 'curriculum'
            for isubset in range(len(self.exclude_parameters_prefixes)):
                self.iexclude_parameters_prefix = isubset

                output = self._fit(train_iterator, val_iterator)

                prefix = f"{step}_{isubset}"
                outputs.update(
                    {f"{prefix}_{key}": value for key, value in output.items()},
                )

                # reset to best model and load next stage
                metric_matrix = self.metric_matrix
                iexclude_parameters_prefix = isubset + 1
                if isubset + 1 == len(self.exclude_parameters_prefixes):
                    # last gradient step
                    iexclude_parameters_prefix = (
                        # one more data step: restart gradient
                        0
                        if step + 1 < train_iterator.curriculum_steps()
                        # no more data step: load best model
                        else -1
                    )
                self.load_state(iexclude_parameters_prefix=iexclude_parameters_prefix)

        self.metric_matrix = metric_matrix

        return outputs

    def _fit(
        self,
        train_iterator: DataLoader,
        val_iterator: DataLoader,
    ) -> dict[str, float]:
        start = 0

        # use existing metrics
        if self.metric_matrix.shape[0] > 0:
            start = self.metric_matrix.shape[0] + 1
        best_epoch = start

        epoch = -1
        val_metric = {"optimization": -1.0}
        train_metric = {"optimization": -1.0}
        for epoch in range(start, self.epochs):
            # early stopping (computationally)
            if self.stop():
                break

            # training
            train_metric = self.run(train_iterator, learn=True, dataset="training")[1]

            # validate
            if self.model.training_validation:
                train_metric = self.run(train_iterator, dataset="training")[1]
            val_metric = self.run(val_iterator, dataset="validation")[1]
            val_metric["epoch"] = epoch
            train_metric["epoch"] = epoch
            self.append_metrics(train_metric, val_metric)

            # update scheduler
            self.scheduler.step(val_metric[self.metric])

            # increment epoch counter
            self.model.epoch += 1

            # save best model
            metrics = self.metric_matrix[self.val_prefix + self.metric]
            current_metric = metrics.iloc[-1]
            if (
                self.metric_matrix.shape[0] == 1
                or (not self.metric_max and current_metric < metrics.iloc[:-1].min())
                or (self.metric_max and current_metric > metrics.iloc[:-1].max())
            ):
                self.save_state()
                best_epoch = epoch

        return {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "lr": self.optimizer.param_groups[0]["lr"],
            "optimization": train_metric["optimization"],
        }
