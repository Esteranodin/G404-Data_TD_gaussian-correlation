"""Check the descriptive and inferential extension of the reference project."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from analysis import (
    describe_groups,
    mean_comparison,
    simulate_mean_differences,
    welch_mean_test,
)
from analysis.inference_plots import plot_sampling_variation
from distributions.gaussian_mixture import GaussianMixture
from run_sampling_variation import DEFAULT_HTML_OUTPUT, DEFAULT_PNG_OUTPUT


class TestAnalysis(unittest.TestCase):
    """Protect descriptive and Welch contracts on one seeded sample."""

    @classmethod
    def setUpClass(cls):
        """Create the shared seeded sample used by every analysis test."""
        model = GaussianMixture(100, 18, 106, 24, weight_a=0.6)
        cls.samples = model.sample(2000, seed=404)

    def test_descriptive_statistics_keep_both_groups(self):
        """Preserve A/B counts and means without mutating the source dataframe."""
        original = self.samples.copy()
        summary = describe_groups(self.samples)

        self.assertEqual(summary.index.tolist(), ["A", "B"])
        self.assertEqual(summary["count"].to_dict(), {"A": 1200, "B": 800})
        self.assertAlmostEqual(summary.loc["A", "mean"], 100.0692, places=4)
        self.assertAlmostEqual(summary.loc["B", "mean"], 105.9307, places=4)
        pd.testing.assert_frame_equal(self.samples, original)

    def test_mean_comparison_uses_b_minus_a(self):
        """Report the signed B-minus-A difference and its standardized value."""
        comparison = mean_comparison(self.samples)

        self.assertAlmostEqual(comparison["difference"], 5.8616, places=4)
        self.assertAlmostEqual(
            comparison["standardized_difference"], 0.2805, places=4
        )

    def test_welch_interval_excludes_zero(self):
        """Keep the seeded Welch interval above zero with a very small p-value."""
        result = welch_mean_test(self.samples)

        self.assertAlmostEqual(result["difference"], 5.8616, places=4)
        self.assertLess(result["confidence_low"], result["difference"])
        self.assertGreater(result["confidence_high"], result["difference"])
        self.assertGreater(result["confidence_low"], 0)
        self.assertLess(result["p_value"], 1e-8)

    def test_unknown_group_is_reported(self):
        """Reject a requested group when the dataframe has no such observations."""
        with self.assertRaisesRegex(ValueError, "Unknown or empty group"):
            mean_comparison(self.samples, comparison="C")

    def test_null_simulation_is_reproducible_and_uses_b_minus_a(self):
        """Keep deterministic seeds and the documented difference direction."""
        first_model = GaussianMixture(103, 18, 103, 24, weight_a=0.6)
        second_model = GaussianMixture(103, 18, 103, 24, weight_a=0.6)
        first = simulate_mean_differences(
            first_model, sample_size=200, repetitions=12, first_seed=900
        )
        second = simulate_mean_differences(
            second_model, sample_size=200, repetitions=12, first_seed=900
        )

        np.testing.assert_array_equal(first, second)
        expected_sample = GaussianMixture(103, 18, 103, 24, weight_a=0.6)
        expected = mean_comparison(expected_sample.sample(200, seed=900))["difference"]
        self.assertAlmostEqual(first[0], expected)

    def test_null_simulation_rejects_invalid_configuration(self):
        """Require equal means and non-empty repeated samples under H0."""
        unequal_model = GaussianMixture(100, 18, 106, 24, weight_a=0.6)
        with self.assertRaisesRegex(ValueError, "equal means"):
            simulate_mean_differences(unequal_model, repetitions=2)

        null_model = GaussianMixture(103, 18, 103, 24, weight_a=0.6)
        with self.assertRaisesRegex(ValueError, "repetitions must be positive"):
            simulate_mean_differences(null_model, repetitions=0)
        with self.assertRaisesRegex(ValueError, "both groups non-empty"):
            simulate_mean_differences(null_model, sample_size=1, repetitions=2)

    def test_sampling_variation_plot_keeps_counts_and_reference_lines(self):
        """Show one count histogram plus the null and observed markers."""
        null_model = GaussianMixture(103, 18, 103, 24, weight_a=0.6)
        differences = np.array([-1.0, -0.4, 0.1, 0.5, 1.2])
        figure = plot_sampling_variation(
            differences,
            observed_difference=5.86,
            null_model=null_model,
            sample_size=2000,
        )

        self.assertEqual(len(figure.data), 1)
        self.assertEqual(figure.data[0].type, "histogram")
        self.assertIsNone(figure.data[0].histnorm)
        self.assertEqual(len(figure.layout.shapes), 2)
        self.assertEqual(
            [shape.x0 for shape in figure.layout.shapes],
            [0, 5.86],
        )
        self.assertIn("n total = 2 000", figure.layout.title.text)

    def test_sampling_exports_use_the_project_output_folder(self):
        """Keep PNG and HTML outputs inside the ignored project folder."""
        project_dir = Path(__file__).resolve().parents[1]
        self.assertEqual(DEFAULT_PNG_OUTPUT, project_dir / "output" / "sampling_variation.png")
        self.assertEqual(DEFAULT_HTML_OUTPUT, project_dir / "output" / "sampling_variation.html")


if __name__ == "__main__":
    unittest.main()
