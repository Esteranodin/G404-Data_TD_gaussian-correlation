"""Descriptive and inferential tools for comparing two groups."""

from .association import (
    compare_correlations,
    pearson_association,
    purchase_gap_decomposition,
    purchase_mean_test,
)
from .convergence import (
    build_convergence_figures,
    convergence_summary,
    mean_difference_convergence_summary,
    plot_multiple_mean_convergence,
    plot_multiple_mean_difference_convergence,
    plot_statistic_convergence,
    save_convergence_figures,
)
from .descriptive import describe_groups, mean_comparison
from .inference import welch_mean_test
from .purchase_plots import (
    build_purchase_xy_figures,
    build_purchase_xy_linear_fit_figures,
    plot_purchase_xy,
    plot_purchase_xy_with_linear_fits,
    save_purchase_xy_figures,
)
from .sampling_variation import simulate_mean_differences

__all__ = [
    "compare_correlations",
    "build_purchase_xy_linear_fit_figures",
    "build_purchase_xy_figures",
    "build_convergence_figures",
    "convergence_summary",
    "mean_difference_convergence_summary",
    "describe_groups",
    "mean_comparison",
    "pearson_association",
    "plot_purchase_xy",
    "plot_purchase_xy_with_linear_fits",
    "plot_multiple_mean_convergence",
    "plot_multiple_mean_difference_convergence",
    "plot_statistic_convergence",
    "purchase_gap_decomposition",
    "purchase_mean_test",
    "save_purchase_xy_figures",
    "save_convergence_figures",
    "simulate_mean_differences",
    "welch_mean_test",
]
