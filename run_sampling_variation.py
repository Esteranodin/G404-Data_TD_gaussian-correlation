"""Simulate and display the sampling variation shown on slide 30."""

import argparse
from pathlib import Path

from analysis import mean_comparison, simulate_mean_differences
from analysis.inference_plots import plot_sampling_variation
from distributions import DEFAULT_SEED
from distributions.gaussian_mixture import GaussianMixture

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
DEFAULT_PNG_OUTPUT = OUTPUT_DIR / "sampling_variation.png"
DEFAULT_HTML_OUTPUT = OUTPUT_DIR / "sampling_variation.html"


def parse_args():
    """Return command-line options for display and export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PNG_OUTPUT,
        help="PNG path; defaults to output/sampling_variation.png.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=DEFAULT_HTML_OUTPUT,
        help="HTML path; defaults to output/sampling_variation.html.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive figure.",
    )
    return parser.parse_args()


def main():
    """Build the observed and null samples, then display the Plotly figure."""
    args = parse_args()
    sample_size = 2000

    observed_model = GaussianMixture(100, 18, 106, 24, weight_a=0.6)
    observed_samples = observed_model.sample(sample_size, seed=DEFAULT_SEED)
    observed_difference = mean_comparison(observed_samples)["difference"]

    null_model = GaussianMixture(103, 18, 103, 24, weight_a=0.6)
    differences = simulate_mean_differences(
        null_model,
        sample_size=sample_size,
        repetitions=50000,
        first_seed=1404,
    )
    figure = plot_sampling_variation(
        differences,
        observed_difference,
        null_model,
        sample_size,
    )

    print(f"Simulated differences: {len(differences)}")
    print(f"Mean under H0: {differences.mean():.3f}")
    print(f"Observed avg(B) - avg(A): {observed_difference:.3f}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.html_output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(args.output)
    figure.write_html(args.html_output)
    print(f"PNG written to: {args.output}")
    print(f"HTML written to: {args.html_output}")
    if not args.no_show:
        figure.show()
    print("Sampling variation Plotly check passed.")


if __name__ == "__main__":
    main()
