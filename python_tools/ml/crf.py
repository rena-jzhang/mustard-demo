#!/usr/bin/env python3
"""Neural CRF with an explicit transition matrix."""
import math
from typing import Final, Optional

import torch
from beartype import beartype

from python_tools.ml.neural import LossModule


class CRF(LossModule):
    """A CRF layer to find a likely sequence."""

    number_states: Final[int]

    @beartype
    def __init__(
        self,
        *,
        number_states: int,
        learn_sink: bool = True,
        learn_source: bool = True,
    ) -> None:
        """Initialize the CRF.

        Args:
            number_states: The number of states (input dimension).
            learn_sink: Whether to learn probabilities for the end states.
            learn_source: Whether to learn probabilities for the start states.
        """
        super().__init__()
        self.number_states = number_states
        uniform = math.log(1 / self.number_states)

        self.transition = torch.nn.Parameter(
            torch.ones(1, self.number_states, self.number_states) * uniform,
        )

        if learn_source:
            self.source = torch.nn.Parameter(
                torch.ones(1, self.number_states) * uniform,
            )
        else:
            self.register_buffer("source", torch.zeros(1, self.number_states))

        if learn_sink:
            self.sink = torch.nn.Parameter(torch.ones(1, self.number_states) * uniform)
        else:
            self.register_buffer("sink", torch.zeros(1, self.number_states))

    def log_partition(self, scores: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Forward algorithm.

        Args:
            scores: The emission probabilities.
            masks: Mask of valid scores.
        """
        score = scores[:, 0, :] + self.source
        for time in range(1, scores.shape[1]):
            mask = masks[:, time].unsqueeze(1)
            score = (
                score * (~mask)
                + torch.logsumexp(
                    score.unsqueeze(2) + scores[:, time, None] + self.transition,
                    1,
                )
                * mask
            )
        return torch.logsumexp(score + self.sink, 1).sum()

    def forward(
        self,
        x: torch.Tensor,
        meta: dict[str, torch.Tensor],
        y: Optional[torch.Tensor] = None,
        dataset: str = "",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run viterbi decoding."""
        with torch.no_grad():
            # skip viterbi for faster training
            if self.training:
                return x[0, 0, 0, None], {
                    "meta_embedding": x[:, 0, :],
                    "meta_states": torch.argmax(x, dim=-1),
                }

            # init
            masks = ~(x == -1).all(dim=2)
            batch, seq = x.shape[:2]
            alphas = torch.zeros(
                (batch, self.number_states),
                device=x.device,
                dtype=x.dtype,
            )
            best_children = -torch.ones(
                (batch, seq, self.number_states),
                dtype=torch.int64,
                device=x.device,
            )
            index = torch.arange(self.number_states)

            # forward pass
            for time in range(seq):
                mask = masks[:, time]
                possible_alphas = alphas[mask].unsqueeze(2) + x[mask, time, None]
                possible_alphas = (
                    possible_alphas + self.transition
                    if time != 0
                    else possible_alphas + self.source.unsqueeze(1)
                )
                ends = mask if time == seq - 1 else mask & ~masks[:, time + 1]
                if ends.any():
                    ends = ends[mask]
                    possible_alphas[ends] = possible_alphas[ends] + self.sink.unsqueeze(
                        1,
                    )
                best_children[mask, time, :] = torch.max(possible_alphas, dim=1).indices
                alphas[mask] = (
                    possible_alphas[:, best_children[mask, time], index].diagonal().T
                )

            # backward from best state
            length = masks.sum(dim=1) - 1
            batch_index = torch.arange(batch)
            states = -torch.ones((batch, seq), dtype=torch.int64, device=x.device)
            states[batch_index, length] = torch.max(alphas, dim=1).indices
            for time in range(seq - 1, 0, -1):
                mask = masks[:, time]
                states[mask, time - 1] = best_children[
                    mask,
                    time,
                    states[mask, time],
                ]

            return alphas[batch_index, states[batch_index, length]], {
                "meta_embedding": alphas,
                "meta_states": states,
            }

    @torch.jit.export  # type: ignore[misc]
    def loss(
        self,
        scores: torch.Tensor,
        ground_truth: torch.Tensor,
        meta: dict[str, torch.Tensor],
        take_mean: bool = True,
        loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Negative log-likelihood of the CRF."""
        mask = ~(scores == -1).all(dim=2)
        length = mask.sum(dim=1) - 1
        return self.log_partition(scores, mask) - self.log_score(
            scores,
            ground_truth,
            mask,
            length,
        )

    def log_score(
        self,
        scores: torch.Tensor,
        states: torch.Tensor,
        masks: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        """Scores the gold sequence given the actual emission probabilities."""
        loss = (
            scores[masks, states[masks]].sum()
            + self.source[:, states[:, 0]].sum()
            + self.sink[:, states[torch.arange(states.shape[0]), length]].sum()
        )
        end = states[:, 1:][masks[:, 1:]]
        masks = masks.clone()
        masks[torch.arange(states.shape[0]), length] = False
        start = states[masks]
        return loss + self.transition[:, start, end].sum()
