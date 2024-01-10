#!/usr/bin/env python3
"""Various small helper functions for PyTorch."""
import contextlib
import math
import warnings
from typing import Final, Union, cast

import numpy as np
import numpy.typing as npt
import torch

from python_tools.ml.metrics import scores_as_matrix

CPU_DEVICE: Final = torch.device("cpu")


def _find_batch_size(size: int, *, batch_size: int = 2048) -> int:
    if batch_size < 1:
        return size

    # determine similar but more balanced batch size
    candidates = np.arange(math.ceil(batch_size * 0.75), math.ceil(batch_size * 1.25))
    return candidates[np.argmax(np.mod(size, candidates) / candidates)].item()


def dict_to_batched_data(
    data: dict[str, np.ndarray],
    *,
    batch_size: int = 2048,
    shuffle: bool = False,
) -> list[dict[str, list[np.ndarray]]]:
    """Convert {'x': array, 'y': array, ...} to batched data format.

    Returns:
        List with number of batches each having a dict.
    """
    float64_keys = [key for key, value in data.items() if value.dtype == np.float64]
    if float64_keys:
        warnings.warn(
            f"Converting float64 to float32 for {float64_keys}",
            RuntimeWarning,
            stacklevel=2,
        )
        for key in float64_keys:
            data[key] = data[key].astype(np.float32)

    size = math.ceil(
        data["x"].shape[0]
        / _find_batch_size(data["x"].shape[0], batch_size=batch_size),
    )
    index: npt.NDArray[np.intp] = np.array([])
    if shuffle:
        index = np.arange(data["x"].shape[0])
        np.random.default_rng(1).shuffle(index)
    shuffeled = {
        key: np.array_split(
            scores_as_matrix(datum)[0][index]
            if shuffle
            else scores_as_matrix(datum)[0],
            size,
            axis=0,
        )
        for key, datum in data.items()
    }

    return [
        {key: [value[i]] for key, value in shuffeled.items()}
        for i in range(len(shuffeled["x"]))
    ]


def dict_list_to_batched_data(
    data: dict[str, list[np.ndarray]],
    *,
    batch_size: int = 2048,
) -> list[dict[str, list[np.ndarray | list[int]]]]:
    """Convert to batched data format. Meant for temporal data only!.

    data    A dictionary in the format of {'x': [...], 'y': [...], ...}. Each
            attribute needs to have the same list length. If items are not 2
            dimensional ndarrays, the are converted to it.
    batch_size Entries per batch (less batch might be different). Negative
            value indicate to put everything in one batch. It might choose to
            take 25% smaller/larger batches to fill the last batch.

    Returns:
        List with number of batches each having a dict.

    """
    for key, value in data.items():
        assert len(value) == len(data["x"]), key
    batch_size = _find_batch_size(len(data["x"]), batch_size=batch_size)
    size = math.ceil(len(data["x"]) / batch_size)
    result: list[dict[str, list[np.ndarray | list[int]]]] = [
        {key: [] for key in data} for _ in range(size)
    ]
    for key in data:
        # tuple format
        batch_index = []
        batch_data = []
        count = 0
        batch_count = 0
        for i, item in enumerate(scores_as_matrix(x)[0] for x in data[key]):
            batch_index.extend([count] * item.shape[0])
            batch_data.append(item)
            count += 1
            # finished collecting batch
            if len(batch_data) == batch_size or i == len(data["x"]) - 1:
                result[batch_count][key] = [
                    np.concatenate(batch_data, axis=0),
                ]
                result[batch_count][key].append(batch_index)
                batch_index = []
                batch_data = []
                count = 0
                batch_count += 1
    return result


def list_to_packedsequence(
    data: list[torch.Tensor | list[int]],
) -> torch.nn.utils.rnn.PackedSequence:
    """Convert a list of size two to a packed sequence.

    data[0] contains a matrix (observations * unrolling x features);
    data[1] contains a numpy array of indices representing a sequence (values
    between 0 and n).
    """
    index = torch.as_tensor(data[1])
    tensor: torch.Tensor = cast(torch.Tensor, data[0])
    return torch.nn.utils.rnn.pack_sequence(
        [tensor[index == i, :] for i in range(int(index.max()) + 1)],
        enforce_sorted=False,
    )


def packedsequence_to_list(
    sequence: torch.nn.utils.rnn.PackedSequence,
    *,
    batch_first: bool = True,
) -> list[torch.Tensor]:
    """Convert a packed sequence to a list of tensors.

    sequence    The sequence to be unpacked.
    reorder_index Numpy array which was used to sort the original sequence.
    batch_first Whether batches are the first dimension.

    Returns:
        A list of the length of the number of sequences.

    """
    return [
        output[:length]
        for output, length in zip(
            *torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=batch_first),
            strict=True,
        )
    ]


def prepare_tensor(
    tensor_list: list[np.ndarray | list[int]],
    *,
    device: torch.device = CPU_DEVICE,
    cuda_device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | torch.nn.utils.rnn.PackedSequence:
    """Move a tensor to the correct device.

    Note:
        Pins the tensor if the `cuda_device` is a cuda device.
    """
    tensor: torch.Tensor | torch.nn.utils.rnn.PackedSequence = (
        torch.from_numpy(tensor_list[0]).to(dtype=dtype).contiguous()
    )

    if len(tensor_list) == 2:
        # PackedSequence
        tensor = list_to_packedsequence(
            cast(list[Union[torch.Tensor, list[int]]], [tensor])
            + cast(list[Union[torch.Tensor, list[int]]], tensor_list[1:]),
        )

    # type and device
    if "cuda" in cuda_device and torch.cuda.is_available():
        with torch.cuda.device(torch.device(cuda_device)), contextlib.suppress(
            RuntimeError,
        ):
            tensor = tensor.pin_memory()
    return tensor.to(device=device, non_blocking=True)
