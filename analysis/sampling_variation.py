"""Simulate mean differences under an equal-means Gaussian null model."""

import numpy as np

from distributions.gaussian_mixture import GaussianMixture

from .descriptive import mean_comparison


def simulate_mean_differences(
    null_model: GaussianMixture,
    sample_size=2000,
    repetitions=5000,
    first_seed=1404,
) -> np.ndarray:
    """Return repeated ``avg(B) - avg(A)`` values generated under H0.

    Args:
        null_model: Gaussian mixture whose two theoretical means are equal.
        sample_size: Total number of A and B observations per repetition.
        repetitions: Number of independent seeded samples to generate.
        first_seed: Seed used for the first repetition; later seeds increment it.
    """
    if not np.isclose(null_model.mean_a, null_model.mean_b):
        raise ValueError("The null model must use equal means for A and B.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    count_a = int(sample_size * null_model.weight_a)
    count_b = sample_size - count_a
    if count_a == 0 or count_b == 0:
        raise ValueError("sample_size and weight_a must keep both groups non-empty.")

    differences = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        samples = null_model.sample(sample_size, seed=first_seed + index)
        comparison = mean_comparison(samples, reference="A", comparison="B")
        differences[index] = comparison["difference"]

    return differences
