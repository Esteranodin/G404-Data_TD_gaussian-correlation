"""Check the same class imported by the launch script."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from scipy.stats import norm

from distributions import DEFAULT_SEED
from distributions import gaussian_mixture, gaussian_plots
from distributions.gaussian_mixture import GaussianMixture
from distributions.gaussian_plots import (
    build_all_figures,
    plot_box,
    plot_components,
    plot_dashboard,
    plot_ecdf,
    plot_mixture,
    plot_violin,
    save_figures,
)


class TestGaussianMixture(unittest.TestCase):
    """Protect generation, state-isolation and plotting contracts."""

    def test_sample_contains_the_expected_values_and_groups(self):
        """Return the stored labelled dataframe with seeded values and metadata."""
        model = GaussianMixture(100, 18, 106, 24)
        samples = model.sample(2000)

        self.assertIs(samples, model.samples)
        self.assertEqual(samples.shape, (2000, 2))
        self.assertEqual(list(samples.columns), ["value", "component"])
        self.assertEqual(samples["component"].value_counts().to_dict(),
                         {"A": 1200, "B": 800})
        expected_rng = np.random.default_rng(DEFAULT_SEED)
        expected_a = expected_rng.normal(100, 18, 1200)
        expected_b = expected_rng.normal(106, 24, 800)
        np.testing.assert_array_equal(
            samples.loc[samples.component == "A", "value"], expected_a
        )
        np.testing.assert_array_equal(
            samples.loc[samples.component == "B", "value"], expected_b
        )
        self.assertEqual(model.n, 2000)
        self.assertEqual(model.seed, DEFAULT_SEED)

    def test_seed_controls_reproducibility(self):
        """Reproduce one sample with the same seed and change it with another."""
        model = GaussianMixture(100, 18, 106, 24)
        original = model.sample(2000, seed=404).copy()
        pd.testing.assert_frame_equal(original, model.sample(2000, seed=404))
        self.assertFalse(original.equals(model.sample(2000, seed=405)))

    def test_instances_keep_independent_parameters_and_data(self):
        """Keep parameters and mutable sample data isolated between instances."""
        first = GaussianMixture(100, 18, 106, 24, weight_a=0.6)
        second = GaussianMixture(100, 18, 106, 30, weight_a=0.3)
        original = first.sample(2000).copy()
        second.sample(2000)

        self.assertIsNot(first.samples, second.samples)
        self.assertEqual(second.samples["component"].value_counts().to_dict(),
                         {"B": 1400, "A": 600})
        second.sample(100, seed=23)
        pd.testing.assert_frame_equal(first.samples, original)
        self.assertEqual((first.weight_a, first.std_b), (0.6, 24))

    def test_plots_explain_when_sample_has_not_run(self):
        """Reject every plot request until the model contains a sample."""
        model = GaussianMixture(100, 18, 106, 24)

        for plot_function in (
            plot_components,
            plot_mixture,
            plot_box,
            plot_ecdf,
            plot_violin,
            plot_dashboard,
        ):
            with self.subTest(function=plot_function.__name__):
                with self.assertRaisesRegex(RuntimeError, r"Call sample\(\)"):
                    plot_function(model)

    def test_figures_use_stored_data_without_redrawing_or_displaying(self):
        """Reuse stored data without drawing, displaying or mutating it."""
        model = GaussianMixture(100, 18, 106, 24)
        original = model.sample(2000).copy()

        with patch(
            "numpy.random.default_rng",
            side_effect=AssertionError("Unexpected draw"),
        ):
            with patch.object(
                Figure,
                "show",
                side_effect=AssertionError("Unexpected display"),
            ):
                figures = [
                    plot_components(model),
                    plot_mixture(model),
                    plot_box(model),
                    plot_ecdf(model),
                    plot_violin(model),
                    plot_dashboard(model),
                ]

        self.assertTrue(all(isinstance(figure, Figure) for figure in figures))
        self.assertEqual([len(figure.data) for figure in figures],
                         [4, 4, 3, 2, 2, 12])
        pd.testing.assert_frame_equal(model.samples, original)

    def test_component_plot_uses_common_density_bins_and_known_curves(self):
        """Use shared density bins and known normal curves for the A/B view."""
        model = GaussianMixture(100, 18, 106, 24)
        samples = model.sample(2000)
        figure = plot_components(model)
        histograms = list(figure.data[:2])
        curves = list(figure.data[2:])

        self.assertEqual([trace.type for trace in histograms],
                         ["histogram", "histogram"])
        self.assertTrue(all(trace.histnorm == "probability density"
                            for trace in histograms))
        first_bins = histograms[0].xbins
        second_bins = histograms[1].xbins
        self.assertEqual(
            (first_bins.start, first_bins.end, first_bins.size),
            (second_bins.start, second_bins.end, second_bins.size),
        )
        for trace, component in zip(histograms, ("A", "B"), strict=True):
            expected = samples.loc[samples.component == component, "value"]
            np.testing.assert_array_equal(trace.x, expected)
            edges = np.arange(
                trace.xbins.start,
                trace.xbins.end + trace.xbins.size,
                trace.xbins.size,
            )
            density, edges = np.histogram(trace.x, bins=edges, density=True)
            self.assertAlmostEqual(float(np.sum(density * np.diff(edges))), 1.0)

        for trace, mean, std in zip(
            curves,
            (model.mean_a, model.mean_b),
            (model.std_a, model.std_b),
            strict=True,
        ):
            np.testing.assert_allclose(trace.y, norm.pdf(trace.x, mean, std))
            self.assertAlmostEqual(float(np.trapezoid(trace.y, trace.x)), 1.0,
                                   places=3)
        self.assertIn("histogramme observé", histograms[0].name)
        self.assertIn("courbe normale théorique", curves[0].name)

    def test_mixture_plot_weights_components_and_sums_them_pointwise(self):
        """Weight theoretical components by model weights and sum them pointwise."""
        model = GaussianMixture(100, 18, 106, 24, weight_a=0.6)
        samples = model.sample(7)
        figure = plot_mixture(model)
        histogram, contribution_a, contribution_b, mixture = figure.data

        self.assertEqual(samples["component"].value_counts().to_dict(),
                         {"A": 4, "B": 3})
        self.assertEqual(histogram.type, "histogram")
        self.assertEqual(histogram.histnorm, "probability density")
        np.testing.assert_array_equal(histogram.x, samples["value"])
        np.testing.assert_allclose(
            contribution_a.y,
            model.weight_a * norm.pdf(contribution_a.x, model.mean_a, model.std_a),
        )
        np.testing.assert_allclose(
            contribution_b.y,
            (1 - model.weight_a)
            * norm.pdf(contribution_b.x, model.mean_b, model.std_b),
        )
        np.testing.assert_allclose(
            mixture.y,
            np.asarray(contribution_a.y) + np.asarray(contribution_b.y),
        )
        self.assertAlmostEqual(
            float(np.trapezoid(contribution_a.y, contribution_a.x)),
            model.weight_a,
            places=3,
        )
        self.assertAlmostEqual(
            float(np.trapezoid(contribution_b.y, contribution_b.x)),
            1 - model.weight_a,
            places=3,
        )
        self.assertIn("60%", contribution_a.name)
        self.assertIn("40%", contribution_b.name)

    def test_box_ecdf_and_violin_reuse_the_stored_group_values(self):
        """Reuse exact stored group values in the box, ECDF and violin views."""
        model = GaussianMixture(100, 18, 106, 24)
        samples = model.sample(2000)
        box = plot_box(model)
        ecdf = plot_ecdf(model)
        violin = plot_violin(model)

        self.assertEqual([trace.name for trace in box.data],
                         ["A", "B", "Mélange"])
        for trace in box.data:
            expected = (
                samples["value"] if trace.name == "Mélange"
                else samples.loc[samples.component == trace.name, "value"]
            )
            np.testing.assert_array_equal(trace.y, expected)

        for trace, component in zip(ecdf.data, ("A", "B"), strict=True):
            expected = np.sort(
                samples.loc[samples.component == component, "value"].to_numpy()
            )
            np.testing.assert_array_equal(trace.x, expected)
            self.assertTrue(np.all(np.diff(trace.x) >= 0))
            self.assertTrue(np.all(np.diff(trace.y) > 0))
            self.assertAlmostEqual(trace.y[-1], 1.0)

        for trace, component in zip(violin.data, ("A", "B"), strict=True):
            expected = samples.loc[samples.component == component, "value"]
            np.testing.assert_array_equal(trace.y, expected)
            self.assertEqual(trace.scalemode, "width")

    def test_dashboard_places_four_views_in_readable_panels(self):
        """Place four views on the expected axes with readable labels and legends."""
        model = GaussianMixture(100, 18, 106, 24)
        model.sample(2000)
        figure = plot_dashboard(model)

        self.assertEqual(len(figure.data), 12)
        self.assertTrue(all(trace.xaxis == "x" and trace.yaxis == "y"
                            for trace in figure.data[:4]))
        self.assertTrue(all(trace.xaxis == "x2" and trace.yaxis == "y2"
                            for trace in figure.data[4:8]))
        self.assertTrue(all(trace.xaxis == "x3" and trace.yaxis == "y3"
                            for trace in figure.data[8:10]))
        self.assertTrue(all(trace.xaxis == "x4" and trace.yaxis == "y4"
                            for trace in figure.data[10:]))
        self.assertTrue(all(trace.showlegend for trace in figure.data[:8]))
        self.assertTrue(all(not trace.showlegend for trace in figure.data[8:]))
        self.assertTrue(all(trace.type == "violin" for trace in figure.data[8:10]))
        self.assertEqual(figure.layout.height, 900)
        panel_titles = [annotation.text for annotation in figure.layout.annotations]
        self.assertEqual(
            panel_titles,
            [
                "A/B séparés : observé + modèle",
                "Mélange : contributions + somme",
                "Forme lissée et repères",
                "Proportions cumulées",
            ],
        )

    def test_module_mains_export_every_figure_and_show_one_dashboard(self):
        """Export six PNG/HTML pairs while displaying only the dashboard."""
        with patch("builtins.print") as print_mock:
            gaussian_mixture.main()

        print_mock.assert_called_once_with(
            "GaussianMixture sampling smoke check passed."
        )

        model = GaussianMixture(100, 18, 106, 24)
        model.sample(20)
        figures = build_all_figures(model)
        with TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(Figure, "write_image", autospec=True) as png_mock:
                with patch.object(Figure, "write_html", autospec=True) as html_mock:
                    paths = save_figures(figures, destination)

        expected_names = {
            "components",
            "mixture",
            "box",
            "ecdf",
            "violin",
            "dashboard",
        }
        self.assertEqual(set(paths), expected_names)
        self.assertEqual(png_mock.call_count, 6)
        self.assertEqual(html_mock.call_count, 6)
        self.assertEqual(
            {call.args[1] for call in png_mock.call_args_list},
            {destination / f"gaussian_{name}.png" for name in expected_names},
        )
        self.assertEqual(
            {call.args[1] for call in html_mock.call_args_list},
            {destination / f"gaussian_{name}.html" for name in expected_names},
        )

        with patch("builtins.print") as print_mock:
            with patch.object(gaussian_plots, "save_figures") as save_mock:
                with patch.object(Figure, "show", autospec=True) as show_mock:
                    gaussian_plots.main()

        save_mock.assert_called_once()
        self.assertEqual(show_mock.call_count, 1)
        shown_figure = show_mock.call_args.args[0]
        self.assertEqual(len(shown_figure.data), 12)
        self.assertEqual(print_mock.call_count, 2)
        print_mock.assert_any_call("Gaussian plots smoke check passed.")


if __name__ == "__main__":
    unittest.main()
