"""Gradient reversal, a small protein-domain classifier and its schedule."""

import math

import torch
from torch import nn


class GradientReversal(torch.autograd.Function):
    """Pass features forward unchanged and reverse their gradients."""

    @staticmethod
    def forward(ctx, features, coefficient):
        """Return identity features and save the reversal coefficient.

        Args:
            ctx: Autograd context receiving the scalar coefficient.
            features: Encoder features of any shape.
            coefficient: Nonnegative gradient reversal strength.

        Returns:
            An identity view of features.
        """
        ctx.coefficient = coefficient
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        """Return the scaled, sign-reversed feature gradient.

        Args:
            ctx: Autograd context holding the coefficient.
            gradient: Upstream derivative with respect to the identity output.

        Returns:
            Negative coefficient times gradient, and no coefficient derivative.
        """
        return -ctx.coefficient * gradient, None


def reverse_gradient(features, coefficient):
    """Apply identity forward and gradient reversal backward.

    Args:
        features: Encoder output tensor.
        coefficient: Scalar gradient reversal strength.

    Returns:
        Identity features with the custom backward rule.
    """
    return GradientReversal.apply(features, coefficient)


class DomainDiscriminator(nn.Module):
    """Two small hidden layers predicting one unseen-domain logit."""

    def __init__(self, input_width, hidden_width=64, dropout=0.1):
        """Build the classifier using the run configuration's widths.

        Args:
            input_width: Protein width, or concatenated protein/drug width.
            hidden_width: Width of both hidden layers, default 64.
            dropout: Dropout probability following each hidden ReLU.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_width, hidden_width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, features):
        """Return one domain logit per row.

        Args:
            features: Tensor of shape (batch, input_width).

        Returns:
            Tensor of shape (batch,), without a sigmoid.
        """
        return self.layers(features).squeeze(-1)


def lambda_schedule(progress, lambda_max):
    """Return the bounded Ganin schedule for training progress.

    Args:
        progress: Fraction of the configured training horizon, from zero to one.
        lambda_max: Nonnegative upper bound on the coefficient.

    Returns:
        Scheduled coefficient, exactly zero at progress zero.
    """
    progress = min(1.0, max(0.0, progress))
    return min(lambda_max, lambda_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0))
