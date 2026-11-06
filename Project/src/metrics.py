"""Exact bounded-memory affinity metrics and the training-mean baseline."""

import numpy as np


def mse(y, prediction) -> float:
    """Compute mean squared error on the supplied score scale.

    Args:
        y: Nonempty one-dimensional true scores.
        prediction: Predictions aligned with y.

    Returns:
        Mean squared error, or NaN for an empty subset.
    """
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.mean((y - prediction) ** 2)) if y.size else float("nan")


def concordance_index(y, prediction, chunk_size: int = 256) -> float:
    """Compute exact CI over ordered pairs with strictly decreasing truth.

    Args:
        y: One-dimensional true scores; equal truths are excluded.
        prediction: Aligned scores; prediction ties count as one half.
        chunk_size: Maximum number of comparison rows allocated at once.

    Returns:
        Concordant fraction, or NaN when no comparable pairs exist.

    Raises:
        ValueError: Shapes differ, inputs are nonfinite, or chunk size is invalid.
    """
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or prediction.shape != y.shape or chunk_size < 1:
        raise ValueError("CI needs aligned 1D scores and a positive chunk size.")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("CI scores must be finite.")
    comparable = concordant = tied = 0
    for start in range(0, len(y), chunk_size):
        valid = y[start:start + chunk_size, None] > y[None, :]
        predicted = prediction[start:start + chunk_size, None]
        comparable += int(np.count_nonzero(valid))
        concordant += int(np.count_nonzero(valid & (predicted > prediction[None, :])))
        tied += int(np.count_nonzero(valid & (predicted == prediction[None, :])))
    return (concordant + 0.5 * tied) / comparable if comparable else float("nan")


def training_mean_baseline(train_scores, y) -> dict:
    """Evaluate a constant estimated from training scores only.

    Args:
        train_scores: Scores from all training pairs, including censored pairs.
        y: Evaluation scores, optionally an uncensored subset.

    Returns:
        Training mean, MSE and CI; undefined metrics are NaN.
    """
    mean = float(np.mean(np.asarray(train_scores, dtype=np.float64)))
    prediction = np.full(len(y), mean)
    return {"mean": mean, "mse": mse(y, prediction), "ci": concordance_index(y, prediction)}
