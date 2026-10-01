"""Store Gaussian-mixture parameters and generate labelled values."""

import numpy as np
import pandas as pd

from . import DEFAULT_SEED


class GaussianMixture:
    """Keep each simulation's configuration and dataframe on its own object."""

    def __init__(self, mean_a, std_a, mean_b, std_b, weight_a=0.6):
        """Store generation parameters and mark the sample as not yet created.

        Args:
            mean_a: Theoretical mean for group A.
            std_a: Theoretical standard deviation for group A.
            mean_b: Theoretical mean for group B.
            std_b: Theoretical standard deviation for group B.
            weight_a: Fraction used to determine group A's sample size.
        """
        self.mean_a = mean_a
        self.std_a = std_a
        self.mean_b = mean_b
        self.std_b = std_b
        self.weight_a = weight_a
        self.n = None
        self.seed = None
        self.samples: pd.DataFrame | None = None

    @staticmethod
    def _combine_components(values_a, values_b) -> pd.DataFrame:
        """Return one dataframe whose rows retain their source group.

        Args:
            values_a: Numeric values drawn for group A.
            values_b: Numeric values drawn for group B.
        """
        return pd.DataFrame({
            "value": np.concatenate([values_a, values_b]),
            "component": ["A"] * len(values_a) + ["B"] * len(values_b),
        })

    def sample(self, n, seed=DEFAULT_SEED) -> pd.DataFrame:
        """Draw both groups, store the dataframe and return that same object.

        Args:
            n: Total number of observations to draw.
            seed: Seed passed to NumPy's random generator.

        Returns:
            Panda Dataframe
        """
        n_a = int(n * self.weight_a)
        n_b = n - n_a
        rng = np.random.default_rng(seed)
        values_a = rng.normal(self.mean_a, self.std_a, n_a)
        values_b = rng.normal(self.mean_b, self.std_b, n_b)
        self.samples = self._combine_components(values_a, values_b)
        self.n = n
        self.seed = seed
        return self.samples


def main():
    """Run a reproducible smoke check of the generation module."""
    model = GaussianMixture(100, 18, 106, 24, weight_a=0.6)
    samples = model.sample(2000, seed=DEFAULT_SEED)
    counts = samples["component"].value_counts().to_dict()

    if counts != {"A": 1200, "B": 800}:
        raise AssertionError(f"Unexpected group counts: {counts}")
    if model.n != 2000 or model.seed != DEFAULT_SEED:
        raise AssertionError("Sampling metadata was not preserved.")

    print("GaussianMixture sampling smoke check passed.")


if __name__ == "__main__":
    main()
