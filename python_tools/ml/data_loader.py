#!/usr/bin/env python3
"""Wrapper for the data."""
from collections.abc import Callable, Collection, Iterable, Iterator
from typing import Any

from python_tools import generic
from python_tools.typing import IteratorCurriculumEnd, IteratorCurriculumSet


class DataLoader:
    """Iterator over the data, apply transforms, and store meta data."""

    def __init__(
        self,
        iterator: Collection[Any] | Callable[[], Iterable[Any]],
        *,
        properties: dict[str, Any] | None = None,
        prefetch: int = 10,
        steps: int = 1,
    ) -> None:
        """Dataloader that also applies transformations to the data.

        Args:
            iterator: Pickleable iterator or a function that returns an iterator.
            properties: Dictionary with name-value pairs which can be requested.
            prefetch: How many batches should be prefetched.
            steps: How many data-driven curriculum steps there are.
        """
        properties = properties or {}

        self.prefetch = prefetch
        self.iterator: Iterable[Any] | None = None
        self.iterator_fun: Callable[[], Iterable[Any]] | None = None
        if isinstance(iterator, Iterable):
            self.iterator = iterator
            self.prefetch = min(prefetch, len(iterator))
        else:
            self.iterator_fun = iterator
        self.properties = properties
        self.transform_funs: list[Callable[[Any], Any]] = []
        self.steps = steps
        self.curriculum_state = 0

    def add_transform(
        self,
        function: Callable[[Any], Any],
        *,
        optimizable: bool = False,
    ) -> None:
        """Set a function to transform the to be returned items.

        Args:
            function: Function to be applied to the elements.
            optimizable: Whether to iterate through the data and apply the function.
        """
        self._load_iterator()
        self.transform_funs.append(function)

        # apply optimization, if it is the first one
        if (
            len(self.transform_funs) == 1
            and optimizable
            and isinstance(self.iterator, list)
        ):
            for i, item in enumerate(self.iterator):
                self.iterator[i] = self.transform_funs[0](item)
            self.transform_funs = []

    def curriculum_steps(self) -> int:
        """Get number of steps for curriculum.

        Returns:
            The number of states.
        """
        return self.steps

    def curriculum_set(self, state: int) -> None:
        """Start the next phase of the curriculum.

        Args:
            state: New state for the data-curriculum.
        """
        if isinstance(self.iterator, IteratorCurriculumSet):
            self.iterator.curriculum_set(state)
        self.curriculum_state = state

    def curriculum_end(self) -> None:
        """End the data-curriculum.

        Note:
            Should be called before final validation in case the curriculum
            influences the validation.
        """
        if isinstance(self.iterator, IteratorCurriculumEnd):
            self.iterator.curriculum_end()
        self.curriculum_state = -1

    def _load_iterator(self) -> None:
        """Create iterable from function."""
        if self.iterator is None:
            assert self.iterator_fun is not None
            self.iterator = self.iterator_fun()

    def __iter__(self) -> Iterator[Any]:
        """Yield an item of the iterable.

        Yields:
            A transformed item.
        """
        self._load_iterator()

        # set curriculum state
        assert self.iterator is not None
        if self.curriculum_state > -1:
            self.curriculum_set(self.curriculum_state)
        else:
            self.curriculum_end()

        iterator = self.iterator
        if self.prefetch > 1:
            iterator = generic.randomized_iterator(self.iterator, size=self.prefetch)

        for item in iterator:
            for transform in self.transform_funs:
                item = transform(item)
            yield item
