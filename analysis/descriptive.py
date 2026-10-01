"""Describe two labelled groups without choosing a statistical test."""

import numpy as np


REQUIRED_COLUMNS = {"value", "component"}


def group_values(samples, label):
    """Return one group's numeric values after checking the dataframe contract.

    Args:
        samples: Dataframe with `value` and `component` columns.
        label: Group label to extract.
    """
    missing = REQUIRED_COLUMNS.difference(samples.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    values = samples.loc[samples["component"] == label, "value"]
    if values.empty:
        raise ValueError(f"Unknown or empty group: {label}")
    if values.isna().any():
        raise ValueError(f"Group {label} contains missing values")
    return values.to_numpy(dtype=float)


def describe_groups(samples):
    """Return centre and dispersion statistics for each observed group.

    Args:
        samples: Dataframe with `value` and `component` columns.
    """
    missing = REQUIRED_COLUMNS.difference(samples.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if samples["value"].isna().any():
        raise ValueError("The value column contains missing values")

    grouped = samples.groupby("component", sort=True)["value"]
    summary = grouped.agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
    )
    summary["q1"] = grouped.quantile(0.25)
    summary["q3"] = grouped.quantile(0.75)
    summary["iqr"] = summary["q3"] - summary["q1"]
    return summary


def mean_comparison(samples, reference="A", comparison="B"):
    """Return comparison - reference and its standardized difference.

    Args:
        samples: Dataframe with `value` and `component` columns.
        reference: Group label subtracted from `comparison`.
        comparison: Group label whose mean is compared with `reference`.
    """
    reference_values = group_values(samples, reference)
    comparison_values = group_values(samples, comparison)

    difference = comparison_values.mean() - reference_values.mean()
    pooled_variance = (
        (len(reference_values) - 1) * reference_values.var(ddof=1)
        + (len(comparison_values) - 1) * comparison_values.var(ddof=1)
    ) / (len(reference_values) + len(comparison_values) - 2)

    return {
        "reference": reference,
        "comparison": comparison,
        "difference": difference,
        "standardized_difference": difference / np.sqrt(pooled_variance),
    }
