"""Construire le dataset canonique et exporter six vues XY."""

from analysis.purchase_plots import (
    build_purchase_xy_linear_fit_figures,
    build_purchase_xy_figures,
    save_purchase_xy_figures,
)
from export_purchase_dataset import build_dataset, validate_dataset


def main() -> None:
    """Exécuter la génération des données puis les six exports graphiques."""
    samples = build_dataset()
    validate_dataset(samples)
    raw_figures = build_purchase_xy_figures(samples)
    analytical_figures = build_purchase_xy_linear_fit_figures(samples)
    exports = (
        save_purchase_xy_figures(raw_figures),
        save_purchase_xy_figures(
            analytical_figures,
            stem_suffix="_with_linear_fits",
        ),
    )
    for exported in exports:
        for predictor, paths in exported.items():
            print(f"{predictor} : {paths['png']}")
            print(f"{predictor} : {paths['html']}")


if __name__ == "__main__":
    main()
