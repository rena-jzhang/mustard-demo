#!/usr/bin/env python3
"""Shared typing definitions."""
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator
from typing import (
    Any,
    Literal,
    Protocol,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import torch
from beartype import beartype
from torch._dynamo.eval_frame import OptimizedModule


class LossModule(torch.nn.Module, ABC):
    epoch: torch.Tensor

    @beartype
    def __init__(
        self,
        *,
        loss_function: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            | Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
            | str
        ) = "MSELoss",
        attenuation: Literal["", "gaussian", "horserace"] = "",
        attenuation_lambda: float = 0.0,
        sample_weight: dict[int, float] | None = None,
        training_validation: bool = False,
    ) -> None:
        super().__init__()

    @abstractmethod
    @torch.jit.export  # type: ignore[misc]
    def loss(
        self,
        scores: torch.Tensor,
        ground_truth: torch.Tensor,
        meta: dict[str, torch.Tensor],
        take_mean: bool = ...,
        loss: torch.Tensor | None = ...,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    @torch.jit.ignore  # type: ignore[misc] # pyright: ignore[reportGeneralTypeIssues]
    def device(self) -> torch.device:
        ...

    @abstractmethod
    @torch.jit.ignore  # type: ignore[misc] # pyright: ignore[reportGeneralTypeIssues]
    def disable_gradient_for_excluded(
        self,
        exclude_parameters_prefix: tuple[str, ...],
    ) -> None:
        ...

    @abstractmethod
    @beartype
    def can_jit(self) -> bool:
        ...


class TransformDictX(TypedDict, total=False):
    mean: np.ndarray
    std: np.ndarray
    keep: np.ndarray


class TransformDict(TypedDict, total=False):
    x: TransformDictX
    y: dict[str, np.ndarray]
    meta_id_count: dict[int, int]
    meta_id: dict[int, float]
    meta_id_clip: dict[int, float]
    meta_batches: int
    meta_x_names: np.ndarray
    meta_y_names: np.ndarray
    meta_y_inverse_weights: np.ndarray | None


@runtime_checkable
class DataLoader(Protocol):
    properties: dict[str, Any]
    iterator: Iterable[Any] | None

    def __init__(
        self,
        iterator: Collection[Any] | Callable[[], Iterable[Any]],
        *,
        properties: dict[str, Any] | None = None,
        prefetch: int = 10,
        steps: int = -1,
    ) -> None:
        ...

    def add_transform(
        self,
        function: Callable[[Any], Any],
        *,
        optimizable: bool = False,
    ) -> None:
        ...

    def curriculum_steps(self) -> int:
        ...

    def curriculum_set(self, state: int) -> None:
        ...

    def curriculum_end(self) -> None:
        ...

    def __iter__(self) -> Iterator[Any]:
        ...


DataLoaderT = TypeVar("DataLoaderT", bound=DataLoader)


@runtime_checkable
class IteratorCurriculumSet(Protocol):
    def curriculum_set(self, state: int) -> None:
        ...


@runtime_checkable
class IteratorCurriculumEnd(Protocol):
    def curriculum_end(self) -> None:
        ...


@runtime_checkable
class MetricWrappedFun(Protocol):
    def __call__(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        *,
        which: tuple[str, ...] = (),
        names: tuple[str, ...] = (),
        y_scores: np.ndarray | None = None,
        ids: np.ndarray | None = None,
        clustering: bool = False,
    ) -> dict[str, float]:
        ...


@runtime_checkable
class MetricFun(Protocol):
    def __call__(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        *,
        which: tuple[str, ...] = (),
        names: tuple[str, ...] = (),
        y_scores: np.ndarray | None = None,
    ) -> dict[str, float]:
        ...


@runtime_checkable
class MetricGenericFun(Protocol):
    def __call__(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        **kwargs: np.ndarray,
    ) -> dict[str, float]:
        ...


@runtime_checkable
class ModelFun(Protocol):
    def __call__(self) -> LossModule:
        ...


@runtime_checkable
class PredictFun(Protocol):
    def __call__(
        self,
        model: Any,
        data: DataLoader,
        dataset: str,
    ) -> dict[str, np.ndarray]:
        ...


@runtime_checkable
class ModelFitFun(Protocol):
    def __call__(
        self,
        training: DataLoader,
        validation: DataLoader,
        metric_fun: MetricWrappedFun,
        partition_name: str,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        ...


@runtime_checkable
class TransformRevertFun(Protocol):
    def __call__(
        self,
        predict: dict[str, np.ndarray],
        *,
        transform: TransformDict,
    ) -> dict[str, np.ndarray]:
        ...


@runtime_checkable
class TransformDefineFun(Protocol):
    def __call__(self, data: DataLoader, **kwargs: Any) -> TransformDict:
        ...


@runtime_checkable
class TransformSetFun(Protocol):
    def __call__(self, data: DataLoaderT, transform: TransformDict) -> DataLoaderT:
        ...


ModelType = LossModule | torch.jit.RecursiveScriptModule | OptimizedModule
