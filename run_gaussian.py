"""Run one simulation, export every Plotly figure and display its dashboard."""

import argparse

from distributions import DEFAULT_SEED
from distributions.gaussian_mixture import GaussianMixture
from distributions.gaussian_plots import (
    OUTPUT_DIR,
    build_all_figures,
    save_figures,
)


def parse_args():
    """Return the option controlling the interactive dashboard display."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Export every figure without opening the dashboard.",
    )
    return parser.parse_args()


def main():
    """Create one model, export its six figures and show one dashboard."""
    args = parse_args()
    mean_a = 100
    std_a = 18

    mean_b = 106
    std_b = 24

    weight_a = 0.6

    n = 2000

    seed = DEFAULT_SEED

    model = GaussianMixture(mean_a, std_a, mean_b, std_b, weight_a)
    samples = model.sample(n, seed)

    print("Sample summary:")
    print(samples.groupby("component")["value"].agg(["count", "mean", "std"]))

    figures = build_all_figures(model)
    save_figures(figures)
    print(f"\nSix PNG and six HTML files written to: {OUTPUT_DIR}")

    # One browser view combines the four complementary readings.
    if not args.no_show:
        figures["dashboard"].show()


if __name__ == "__main__":
    main()
