"""Vérifier les cinq colonnes et leur export CSV."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure

from analysis import compare_correlations, purchase_gap_decomposition
from analysis.purchase_plots import (
    COLORS,
    MARKER_LINE_WIDTH,
    MARKER_SIZE,
    OUTPUT_DIR,
    PLOT_SPECS,
    POOLED_LINE_COLOR,
    TRANSPARENT_CENTER,
    build_purchase_xy_linear_fit_figures,
    build_purchase_xy_figures,
    save_purchase_xy_figures,
)
from distributions.age import add_age
from distributions.purchase_dataset import (
    add_checkout_wait_minutes,
    add_distance_to_store_km,
    add_purchase_amount,
    make_groups,
)
from export_purchase_dataset import (
    DEFAULT_OUTPUT,
    EXPECTED_COLUMNS,
    build_dataset,
    export_dataset,
)


class TestAgeAssociation(unittest.TestCase):
    """Protéger le pipeline fonctionnel et son fichier exporté."""

    @classmethod
    def setUpClass(cls):
        """Créer une seule fois les lignes A/B."""
        cls.groups = make_groups(2000, weight_a=0.6)

    def test_add_age_returns_a_copy_and_preserves_existing_data(self):
        """Ajouter seulement age sans modifier les groupes."""
        original = self.groups.copy()
        enriched = add_age(self.groups)

        self.assertIsNot(enriched, self.groups)
        self.assertEqual(list(enriched.columns), ["group", "age"])
        pd.testing.assert_frame_equal(enriched[["group"]], original)
        pd.testing.assert_frame_equal(self.groups, original)

    def test_age_is_reproducible_plausible_and_differs_by_group(self):
        """Reproduire des âges entiers compris entre 18 et 80 ans."""
        first = add_age(self.groups, seed=405)
        second = add_age(self.groups, seed=405)

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(pd.api.types.is_integer_dtype(first["age"]))
        self.assertTrue(first["age"].between(18, 80).all())
        means = first.groupby("group")["age"].mean()
        self.assertGreater(means["B"] - means["A"], 6.0)

    def test_purchase_amount_follows_the_rule_and_preserves_age(self):
        """Ajouter le montant sans modifier group ni age."""
        with_age = add_age(self.groups, seed=405)
        samples = add_purchase_amount(with_age, noise_std=0.0)
        expected = 80.0 + 0.5 * samples["age"] + 2.0 * samples["group"].eq("B")

        pd.testing.assert_frame_equal(samples[["group", "age"]], with_age)
        pd.testing.assert_series_equal(
            samples["purchase_amount"],
            expected.astype(float),
            check_names=False,
        )

    def test_purchase_dataset_has_expected_means_and_correlations(self):
        """Retrouver l'écart partiel et les associations dans les deux groupes."""
        samples = add_purchase_amount(add_age(self.groups, seed=405), seed=406)
        means = samples.groupby("group")["purchase_amount"].mean()
        correlations = compare_correlations(samples)
        decomposition = purchase_gap_decomposition(samples, 0.5, 2.0)

        self.assertAlmostEqual(means["A"], 100.084, places=3)
        self.assertAlmostEqual(means["B"], 105.959625, places=6)
        scaled_amounts = samples["purchase_amount"] * 10
        self.assertTrue(np.allclose(scaled_amounts, np.round(scaled_amounts)))
        self.assertGreater(correlations.loc["A+B", "correlation"], 0.50)
        self.assertGreater(correlations.loc["A", "correlation"], 0.45)
        self.assertGreater(correlations.loc["B", "correlation"], 0.35)
        self.assertAlmostEqual(decomposition["age_profile_part"], 3.879, places=3)
        self.assertAlmostEqual(decomposition["remaining_group_part"], 2.0)
        self.assertAlmostEqual(
            decomposition["observed_residual_gap"],
            -0.003208,
            places=6,
        )

    def test_checkout_wait_has_no_programmed_association(self):
        """Ajouter une attente reproductible sans lire les autres colonnes."""
        purchases = add_purchase_amount(add_age(self.groups), seed=406)
        first = add_checkout_wait_minutes(purchases, seed=407)
        second = add_checkout_wait_minutes(purchases, seed=407)

        self.assertIsNot(first, purchases)
        pd.testing.assert_frame_equal(
            first.drop(columns="checkout_wait_minutes"),
            purchases,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["checkout_wait_minutes"].between(1, 15).all())
        correlations = compare_correlations(
            first,
            predictor="checkout_wait_minutes",
        )
        self.assertLess(abs(correlations.loc["A+B", "correlation"]), 0.06)
        self.assertLess(abs(correlations.loc["A", "correlation"]), 0.06)
        self.assertLess(abs(correlations.loc["B", "correlation"]), 0.06)

    def test_distance_depends_on_group_but_not_purchase_within_groups(self):
        """Créer un effet de composition sans pente interne programmée."""
        purchases = add_purchase_amount(add_age(self.groups), seed=406)
        samples = add_distance_to_store_km(purchases, seed=408)
        means = samples.groupby("group")["distance_to_store_km"].mean()
        correlations = compare_correlations(
            samples,
            predictor="distance_to_store_km",
        )

        pd.testing.assert_frame_equal(
            samples.drop(columns="distance_to_store_km"),
            purchases,
        )
        self.assertGreater(means["B"] - means["A"], 6.0)
        self.assertGreater(correlations.loc["A+B", "correlation"], 0.20)
        self.assertLess(abs(correlations.loc["A", "correlation"]), 0.05)
        self.assertLess(abs(correlations.loc["B", "correlation"]), 0.05)

    def test_invalid_group_is_rejected(self):
        """Signaler une valeur de groupe non prévue."""
        invalid = pd.DataFrame({"group": ["A", "C"]})
        with self.assertRaisesRegex(ValueError, "uniquement A et B"):
            add_age(invalid)

    def test_csv_export_matches_the_complete_pipeline(self):
        """Relire exactement les cinq colonnes exportées sans colonne d'index."""
        expected = build_dataset()
        project_dir = Path(__file__).resolve().parents[1]
        self.assertEqual(DEFAULT_OUTPUT.parent, project_dir / "data")
        self.assertEqual(DEFAULT_OUTPUT.name, "purchase_transactions.csv")
        with TemporaryDirectory() as directory:
            output = Path(directory) / "purchase_transactions.csv"
            exported = export_dataset(output)
            loaded = pd.read_csv(output, float_precision="round_trip")
            purchase_text = [
                line.split(",")[2]
                for line in output.read_text(encoding="utf-8").splitlines()[1:]
            ]

        self.assertEqual(list(loaded.columns), EXPECTED_COLUMNS)
        self.assertEqual(loaded.shape, (2000, 5))
        self.assertFalse(loaded.isna().any().any())
        self.assertTrue(
            all(len(value.rsplit(".", 1)[1]) == 2 for value in purchase_text)
        )
        pd.testing.assert_frame_equal(exported, expected)
        pd.testing.assert_frame_equal(loaded, expected)

    def test_three_xy_figures_use_raw_data_and_one_shared_y_range(self):
        """Tracer les deux groupes avec le même style et la même échelle Y."""
        samples = build_dataset()
        figures = build_purchase_xy_figures(samples)
        y_ranges = {tuple(figure.layout.yaxis.range) for figure in figures.values()}

        self.assertEqual(set(figures), set(PLOT_SPECS))
        self.assertEqual(len(y_ranges), 1)
        for predictor, figure in figures.items():
            self.assertEqual(len(figure.data), 2)
            self.assertEqual([trace.name for trace in figure.data], ["Magasin A", "Magasin B"])
            self.assertTrue(all(trace.mode == "markers" for trace in figure.data))
            self.assertEqual(figure.layout.xaxis.title.text, PLOT_SPECS[predictor])
            self.assertEqual(figure.layout.yaxis.title.text, "Montant d'achat (€)")
            for trace, group in zip(figure.data, ("A", "B"), strict=True):
                expected = samples.loc[samples["group"] == group]
                self.assertEqual(trace.marker.color, TRANSPARENT_CENTER)
                self.assertEqual(trace.marker.size, MARKER_SIZE)
                self.assertEqual(trace.marker.symbol, "circle")
                self.assertEqual(trace.marker.opacity, 1.0)
                self.assertEqual(trace.marker.line.color, COLORS[group])
                self.assertEqual(trace.marker.line.width, MARKER_LINE_WIDTH)
                np.testing.assert_array_equal(trace.x, expected[predictor].to_numpy())
                np.testing.assert_array_equal(
                    trace.y,
                    expected["purchase_amount"].to_numpy(),
                )

    def test_three_analytical_xy_figures_add_pooled_and_group_linear_fits(self):
        """Limiter chaque droite descriptive à la plage X qui l'a produite."""
        samples = build_dataset()
        figures = build_purchase_xy_linear_fit_figures(samples)

        self.assertEqual(set(figures), set(PLOT_SPECS))
        for predictor, figure in figures.items():
            self.assertEqual(len(figure.data), 5)
            self.assertEqual(
                [trace.name for trace in figure.data],
                ["Magasin A", "Magasin B", "Droite A+B", "Droite A", "Droite B"],
            )
            self.assertTrue(all(trace.mode == "markers" for trace in figure.data[:2]))
            self.assertTrue(all(trace.mode == "lines" for trace in figure.data[2:]))
            self.assertEqual(figure.data[2].line.color, POOLED_LINE_COLOR)
            self.assertEqual(figure.data[2].line.dash, "dash")
            for trace, group in zip(figure.data[3:], ("A", "B"), strict=True):
                selected = samples.loc[samples["group"] == group, predictor]
                np.testing.assert_allclose(trace.x, [selected.min(), selected.max()])
                self.assertEqual(trace.line.color, COLORS[group])
                self.assertEqual(trace.line.dash, "solid")

    def test_three_xy_figures_export_png_and_html_to_output(self):
        """Utiliser six noms stables dans le dossier local output."""
        project_dir = Path(__file__).resolve().parents[1]
        self.assertEqual(OUTPUT_DIR, project_dir / "output")
        figures = build_purchase_xy_figures(build_dataset())
        with TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(Figure, "write_image", autospec=True) as png_mock:
                with patch.object(Figure, "write_html", autospec=True) as html_mock:
                    exported = save_purchase_xy_figures(figures, destination)

        self.assertEqual(png_mock.call_count, 3)
        self.assertEqual(html_mock.call_count, 3)
        for predictor in PLOT_SPECS:
            stem = f"{predictor}_vs_purchase_amount"
            self.assertEqual(exported[predictor]["png"], destination / f"{stem}.png")
            self.assertEqual(exported[predictor]["html"], destination / f"{stem}.html")

    def test_analytical_xy_figures_use_distinct_export_names(self):
        """Ne jamais remplacer les nuages bruts par les vues avec droites."""
        figures = build_purchase_xy_linear_fit_figures(build_dataset())
        with TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(Figure, "write_image", autospec=True) as png_mock:
                with patch.object(Figure, "write_html", autospec=True) as html_mock:
                    exported = save_purchase_xy_figures(
                        figures,
                        destination,
                        stem_suffix="_with_linear_fits",
                    )

        self.assertEqual(png_mock.call_count, 3)
        self.assertEqual(html_mock.call_count, 3)
        for predictor in PLOT_SPECS:
            stem = f"{predictor}_vs_purchase_amount_with_linear_fits"
            self.assertEqual(exported[predictor]["png"], destination / f"{stem}.png")
            self.assertEqual(exported[predictor]["html"], destination / f"{stem}.html")


if __name__ == "__main__":
    unittest.main()
