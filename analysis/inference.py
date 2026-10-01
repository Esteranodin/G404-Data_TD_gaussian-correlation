"""Estimate and test the difference between two independent group means."""

from scipy.stats import ttest_ind

from .descriptive import group_values


def welch_mean_test(
    samples,
    reference="A",
    comparison="B",
    confidence_level=0.95,
):
    """Return a Welch test and interval for comparison - reference.

    Args:
        samples: Dataframe with `value` and `component` columns.
        reference: Group label subtracted from `comparison`.
        comparison: Group label whose mean is compared with `reference`.
        confidence_level: Confidence level used for the returned interval.
    """
    reference_values = group_values(samples, reference)
    comparison_values = group_values(samples, comparison)

    result = ttest_ind(comparison_values, reference_values, equal_var=False)
    interval = result.confidence_interval(confidence_level=confidence_level)

    return {
        "reference": reference,
        "comparison": comparison,
        "difference": comparison_values.mean() - reference_values.mean(),
        "confidence_level": confidence_level,
        "confidence_low": interval.low,
        "confidence_high": interval.high,
        "t_statistic": result.statistic,
        "degrees_of_freedom": result.df,
        "p_value": result.pvalue,
    }
