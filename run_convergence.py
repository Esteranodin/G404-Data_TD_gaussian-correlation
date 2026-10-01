"""Construire et exporter les figures de convergence descriptive."""

from analysis.convergence import (
    build_convergence_figures,
    save_convergence_figures,
)


def main() -> None:
    """Exécuter le calcul puis exporter les cinq figures."""
    figures = build_convergence_figures()
    exported = save_convergence_figures(figures)
    for name, paths in exported.items():
        print(f"{name} : {paths['png']}")
        print(f"{name} : {paths['html']}")


if __name__ == "__main__":
    main()
