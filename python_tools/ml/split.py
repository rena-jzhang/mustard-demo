#!/usr/bin/env python3
"""A function to split data into stratified folds."""
import numpy as np


def stratified_splits(groups: np.ndarray, proportions: np.ndarray) -> np.ndarray:
    """Create a approximately stratified split.

    Note:
        Stratification is achieved by sampling from a sorted list.
        This function is deterministic.

    Args:
        groups: Defines the groups (observations x labels).
        proportions: The proportion of each partition.

    Returns:
        Integer array indicating to which partition each element belongs.
    """
    # input validation
    assert np.all(proportions > 0)
    if groups.ndim == 1:
        groups = groups.reshape(-1, 1)

    # sorted labels hierarchically (can somewhat handle multiple labels)
    permutation = np.arange(groups.shape[0])
    for column in range(groups.shape[1] - 1, -1, -1):
        index = np.argsort(groups[:, column])
        permutation = permutation[index]
        groups = groups[index, :]

    # cope with odd ratios
    reminder = np.mod(proportions, 1)
    proportions = np.floor(proportions).astype(np.uint8)

    # stratify by sorting
    partition_index = np.full(groups.shape[0], -1)
    mask = np.concatenate(
        [np.full(x, i, dtype=np.uint8) for i, x in enumerate(proportions)],
    )
    rng = np.random.default_rng(1)
    i = 0
    while partition_index.min() == -1:
        # create mask
        mask = np.concatenate(
            [
                np.full(x + (rng.random() < y), i, dtype=np.uint8)
                for i, (x, y) in enumerate(zip(proportions, reminder, strict=True))
            ],
        )
        index = np.arange(0, mask.size)
        rng.shuffle(mask)

        # for last round
        index = index[i + index < groups.shape[0]]

        partition_index[i + index] = mask[index]
        i += index.size

    return partition_index[permutation]
