"""Compare the two simulated groups with descriptive statistics and Welch's test."""

from analysis import describe_groups, mean_comparison, welch_mean_test
from distributions import DEFAULT_SEED
from distributions.gaussian_mixture import GaussianMixture


def main():
    """Print descriptive and Welch results for the seeded reference sample."""
    model = GaussianMixture(100, 18, 106, 24, weight_a=0.6)
    samples = model.sample(2000, seed=DEFAULT_SEED)

    print("Descriptive statistics:")
    print(describe_groups(samples).round(3))

    descriptive = mean_comparison(samples)
    print(
        f"\nObserved mean difference avg(B) - avg(A): "
        f"{descriptive['difference']:.3f}"
    )
    print(
        "Standardized difference: "
        f"{descriptive['standardized_difference']:.3f}"
    )

    inference = welch_mean_test(samples)
    confidence = int(inference["confidence_level"] * 100)
    print(
        f"{confidence}% confidence interval: "
        f"[{inference['confidence_low']:.3f}, "
        f"{inference['confidence_high']:.3f}]"
    )
    print(f"Welch p-value: {inference['p_value']:.6g}")


if __name__ == "__main__":
    main()
