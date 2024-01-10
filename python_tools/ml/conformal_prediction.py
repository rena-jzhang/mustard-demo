#!/usr/bin/env python3
"""Inductive conformal prediction."""
import numpy as np


class ICP:
    """Standard Inductive Conformal Prediction."""

    def __init__(
        self,
        alpha_calibration: np.ndarray,
        *,
        epsilon: float = 0.05,
        conditional: np.ndarray | None = None,
    ) -> None:
        """Create an object to calculate the preditive set/interval.

        Args:
            alpha_calibration: The calibration alpha values (probability
                of ground truth, MAE for regression).
            epsilon: Targeted error rate.
            conditional: True labels to condition prediction set on.
        """
        if alpha_calibration.shape[1] == 1:
            # regression
            alpha_calibration = np.sort(alpha_calibration, axis=0)[::-1]
            assert conditional is None

        self.alpha_calibration = alpha_calibration
        self.epsilon = epsilon
        self.conditional = conditional

    def prediction_set(
        self,
        test: np.ndarray,
        *,
        normalization: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate the prediction set.

        Args:
            test: Alphas for the test set (y_hat for regression).
            normalization: Normalization (for regression).

        Returns:
            A binary matrix indicating the prediction set for classification or
            a Nx2 matrix indicating the prediction interval.
        """
        if normalization is None:
            normalization = np.ones(test.shape)

        if test.shape[1] == 1:
            # regression
            alpha_s = self.alpha_calibration[
                int(np.floor(self.alpha_calibration.size * self.epsilon + 1)),
                0,
            ]
            result = np.zeros((test.shape[0], 2))
            result[:, 0] = test[:, 0] - alpha_s * normalization[:, 0]
            result[:, 1] = test[:, 0] + alpha_s * normalization[:, 0]
        else:
            # classification
            test = test / normalization
            if self.conditional is None:
                counts = np.sum(
                    test[:, :, None] <= self.alpha_calibration[None, None, :, 0],
                    axis=2,
                )
                p_values = (counts + 1) / (self.alpha_calibration.shape[0] + 1)
            else:
                # for each class
                p_values = np.full(test.shape, -1.0)
                for ilabel, label in enumerate(np.unique(self.conditional)):
                    index = self.conditional == label
                    counts = np.sum(
                        test[:, ilabel, None] <= self.alpha_calibration[None, index],
                        axis=1,
                    )
                    p_values[:, ilabel] = (counts + 1.0) / (index.sum() + 1)
            result = p_values >= self.epsilon

        return result


def get_alpha(
    ground_truth_prediction: np.ndarray,
    guess: np.ndarray,
    *,
    normalization: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Calculate calibration alphas.

    Args:
        ground_truth_prediction: Ground truth for regression, all probabilities
            for classification.
        guess: The predictions.
        normalization: Normalization, usually for regression.

    Returns:
        The alphas for ICP.
    """
    if guess.shape[1] == 1:
        result = np.abs(ground_truth_prediction - guess)
    else:
        guess = guess.reshape(-1)
        result = ground_truth_prediction[range(guess.size), guess, None]
    return result / normalization
