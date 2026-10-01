"""Vérifier les calculs et exports de convergence descriptive."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from plotly.graph_objects import Figure

from analysis.convergence import (
    DEFAULT_CONVERGENCE_SEED,
    MEAN_DIFFERENCE_SEEDS,
    MULTIPLE_CONVERGENCE_SEEDS,
    OUTPUT_DIR,
    SAMPLE_SIZES,
    build_convergence_figures,
    convergence_summary,
    mean_difference_convergence_summary,
    save_convergence_figures,
)


class TestConvergence(unittest.TestCase):
    """Protéger les cinq figures fondées sur des préfixes emboîtés."""

    def test_summary_uses_twenty_nested_sizes_and_seed(self):
        """Conserver 20 tailles, la reproductibilité et le résultat final."""
        first = convergence_summary(DEFAULT_CONVERGENCE_SEED)
        second = convergence_summary(DEFAULT_CONVERGENCE_SEED)
        other = convergence_summary(DEFAULT_CONVERGENCE_SEED + 1)

        self.assertEqual(len(first), 20)
        self.assertEqual(SAMPLE_SIZES[0], 10)
        self.assertEqual(SAMPLE_SIZES[-1], 5_000)
        np.testing.assert_array_equal(first["sample_size"], SAMPLE_SIZES)
        self.assertTrue(first.equals(second))
        self.assertFalse(first["sample_mean"].equals(other["sample_mean"]))
        self.assertAlmostEqual(first.iloc[-1]["sample_mean"], 100.15731312500668)
        self.assertAlmostEqual(first.iloc[-1]["sample_std"], 14.991408736900555)

        difference = mean_difference_convergence_summary(MEAN_DIFFERENCE_SEEDS[0])
        self.assertEqual(difference.iloc[14]["sample_size"], 2_000)
        self.assertEqual(difference.iloc[14]["sample_size_a"], 1_200)
        self.assertEqual(difference.iloc[14]["sample_size_b"], 800)
        self.assertAlmostEqual(
            difference.iloc[-1]["mean_difference"],
            0.26169856203551944,
        )

    def test_five_figures_show_single_and_multiple_seeds(self):
        """Construire les deux statistiques et les comparaisons de moyenne."""
        figures = build_convergence_figures()

        self.assertEqual(
            set(figures),
            {
                "sample_mean_convergence",
                "sample_std_convergence",
                "sample_mean_convergence_seed_407",
                "sample_mean_convergence_multiple_seeds",
                "sample_mean_difference_convergence_multiple_seeds",
            },
        )
        self.assertEqual(len(figures["sample_mean_convergence"].data), 2)
        self.assertEqual(len(figures["sample_std_convergence"].data), 2)
        self.assertEqual(len(figures["sample_mean_convergence_seed_407"].data), 2)
        self.assertEqual(
            len(figures["sample_mean_convergence_multiple_seeds"].data),
            len(MULTIPLE_CONVERGENCE_SEEDS) + 1,
        )
        self.assertEqual(
            len(figures["sample_mean_difference_convergence_multiple_seeds"].data),
            len(MEAN_DIFFERENCE_SEEDS) + 1,
        )
        for figure in figures.values():
            self.assertEqual(figure.layout.xaxis.type, "log")

    def test_exports_use_stable_names_in_output(self):
        """Exporter cinq PNG et cinq HTML dans le dossier demandé."""
        project_dir = Path(__file__).resolve().parents[1]
        self.assertEqual(OUTPUT_DIR, project_dir / "output")
        figures = build_convergence_figures()
        with TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(Figure, "write_image", autospec=True) as png_mock:
                with patch.object(Figure, "write_html", autospec=True) as html_mock:
                    exported = save_convergence_figures(figures, destination)

        self.assertEqual(png_mock.call_count, 5)
        self.assertEqual(html_mock.call_count, 5)
        for name in figures:
            self.assertEqual(exported[name]["png"], destination / f"{name}.png")
            self.assertEqual(exported[name]["html"], destination / f"{name}.html")


if __name__ == "__main__":
    unittest.main()
